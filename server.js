const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 10000;

// Enable CORS for all routes
app.use(cors());

// Parse JSON request bodies
app.use(express.json({ limit: '1mb' })); // Increase limit for larger prompts

// Simple health check endpoint
app.get('/', (req, res) => {
    res.json({
        status: 'Dermi Hugging Face Proxy API is running',
        currentModel: CURRENT_MODEL
    });
});

// Global variables to track model loading status
let modelLoadingInProgress = false;
let lastModelLoadAttempt = 0;
let modelSuccessfullyLoaded = false;
let consecutiveFailures = 0;
let CURRENT_MODEL = process.env.PRIMARY_MODEL || 'microsoft/phi-2';

// Define models in order of preference
const MODELS = [
    'microsoft/phi-2',           // Primary choice - 2.7B parameter model
    'google/flan-t5-small',      // Fallback #1 - smaller but reliable
    'facebook/opt-350m',         // Fallback #2 - very small but very reliable
    'distilbert/distilbert-base-uncased' // Last resort - not ideal for generation but should work
];

// Implement a basic rate limiter
const requestCounts = {};
const RATE_LIMIT_WINDOW = 60000; // 1 minute in milliseconds
const RATE_LIMIT = 10; // requests per minute

function rateLimiter(req, res, next) {
    const ip = req.ip;
    const now = Date.now();

    if (!requestCounts[ip]) {
        requestCounts[ip] = {
            count: 1,
            resetTime: now + RATE_LIMIT_WINDOW
        };
        return next();
    }

    if (now > requestCounts[ip].resetTime) {
        // Reset counter if the time window has passed
        requestCounts[ip] = {
            count: 1,
            resetTime: now + RATE_LIMIT_WINDOW
        };
        return next();
    }

    // Increment counter
    requestCounts[ip].count++;

    // Check if rate limit exceeded
    if (requestCounts[ip].count > RATE_LIMIT) {
        return res.status(429).json({
            error: 'Rate limit exceeded',
            message: 'Too many requests. Please try again later.'
        });
    }

    next();
}

// Format prompt based on model type
function formatPrompt(inputs, modelName) {
    // Different models may require different prompting formats
    if (modelName.includes('phi-2')) {
        // Microsoft phi-2 uses a simple User/Assistant format
        return `User: ${inputs}\nAssistant:`;
    } else if (modelName.includes('flan-t5')) {
        // T5 models work well with task prefixes
        return `Answer this medical question: ${inputs}`;
    } else if (modelName.includes('opt')) {
        // OPT models work with simple prompts
        return `Q: ${inputs}\nA:`;
    } else {
        // Default format
        return inputs;
    }
}

// Extract response from model format
function extractModelResponse(response, modelName) {
    // Handle different model output formats
    let cleanedResponse = response;

    if (modelName.includes('phi-2')) {
        // Clean up phi-2 responses
        cleanedResponse = cleanedResponse.replace(/\nUser:.*$/s, '');
        cleanedResponse = cleanedResponse.replace(/^Assistant:\s*/i, '');
        cleanedResponse = cleanedResponse.replace(/^Dermi:\s*/i, '');
    } else if (modelName.includes('flan-t5') || modelName.includes('opt')) {
        // These models typically return just the answer
        cleanedResponse = cleanedResponse.replace(/^A:\s*/i, '');
    }

    return cleanedResponse.trim();
}

// Exponential backoff for retries
async function wait(attemptNumber) {
    const baseDelay = 1000; // 1 second
    const maxDelay = 30000; // 30 seconds
    const delay = Math.min(baseDelay * Math.pow(2, attemptNumber), maxDelay);
    await new Promise(resolve => setTimeout(resolve, delay));
}

// Get model parameters based on model name
function getModelParameters(modelName) {
    if (modelName.includes('phi-2')) {
        return {
            max_new_tokens: 250,
            temperature: 0.5,
            top_p: 0.95,
            do_sample: true,
            return_full_text: false
        };
    } else if (modelName.includes('flan-t5')) {
        return {
            max_new_tokens: 150,
            temperature: 0.7,
            top_p: 0.9,
            do_sample: true
        };
    } else if (modelName.includes('opt')) {
        return {
            max_new_tokens: 100,
            temperature: 0.6,
            top_p: 0.9,
            do_sample: true,
            return_full_text: false
        };
    } else {
        // Default parameters for other models
        return {
            max_new_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            do_sample: true
        };
    }
}

// Hugging Face API calling function with better error handling and fallback
async function callHuggingFaceAPI(inputs, attemptNumber = 0, modelIndex = 0) {
    const HUGGING_FACE_API_KEY = process.env.HUGGINGFACE_API_KEY;

    if (!HUGGING_FACE_API_KEY) {
        console.error('Missing Hugging Face API key');
        throw new Error('Server configuration error: Missing API key');
    }

    // Get the model to try
    if (modelIndex >= MODELS.length) {
        throw new Error('All models failed, unable to process request');
    }

    const modelName = MODELS[modelIndex];
    CURRENT_MODEL = modelName;

    try {
        console.log(`Attempt ${attemptNumber + 1} with model ${modelName}`);
        console.log(`Input length: ${inputs.length} characters`);

        // Format the input correctly for the chosen model
        const formattedInput = formatPrompt(inputs, modelName);
        const parameters = getModelParameters(modelName);

        const response = await axios.post(
            `https://api-inference.huggingface.co/models/${modelName}`,
            {
                inputs: formattedInput,
                parameters: parameters
            },
            {
                headers: {
                    'Authorization': `Bearer ${HUGGING_FACE_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                timeout: 30000 + (attemptNumber * 10000) // Increase timeout with each retry
            }
        );

        console.log(`Received successful response from model ${modelName}`);
        console.log('Response preview:',
            JSON.stringify(response.data).substring(0, 150) + '...');

        modelLoadingInProgress = false;
        modelSuccessfullyLoaded = true;
        consecutiveFailures = 0; // Reset failure counter on success

        // Parse the response based on its format
        let rawResponse = '';

        if (Array.isArray(response.data)) {
            if (response.data.length > 0) {
                rawResponse = response.data[0].generated_text || '';
            }
        } else if (typeof response.data === 'object') {
            rawResponse = response.data.generated_text || '';
        } else if (typeof response.data === 'string') {
            rawResponse = response.data;
        }

        // Handle any other possible response formats
        if (!rawResponse && response.data) {
            rawResponse = JSON.stringify(response.data);
        }

        // Extract the actual response text
        const generatedText = extractModelResponse(rawResponse, modelName);

        // Return in consistent format
        return {
            generated_text: generatedText,
            model_used: modelName
        };
    } catch (error) {
        // Enhanced error logging
        console.error(`Error with model ${modelName}:`, {
            message: error.message,
            code: error.code,
            status: error.response?.status,
            statusText: error.response?.statusText
        });

        consecutiveFailures++;

        // Handle model loading errors
        if (error.response?.status === 503 ||
            (error.response?.data?.error && error.response.data.error.includes('loading'))) {

            console.log(`Model ${modelName} unavailable or loading`);
            modelLoadingInProgress = true;
            lastModelLoadAttempt = Date.now();

            // If we've had too many failures with this model, try the next one
            if (consecutiveFailures > 2) {
                console.log(`Switching to next model after ${consecutiveFailures} failures`);
                return callHuggingFaceAPI(inputs, 0, modelIndex + 1);
            }

            // If this is not the first attempt, wait with exponential backoff
            if (attemptNumber > 0) {
                await wait(attemptNumber);
            }

            // Retry with the same model (up to 3 times per model)
            if (attemptNumber < 2) {
                return callHuggingFaceAPI(inputs, attemptNumber + 1, modelIndex);
            } else {
                // After 3 attempts, try the next model
                return callHuggingFaceAPI(inputs, 0, modelIndex + 1);
            }
        }

        // Special handling for timeouts
        if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
            console.log(`Timeout with model ${modelName}`);

            // Try the next model immediately on timeout
            return callHuggingFaceAPI(inputs, 0, modelIndex + 1);
        }

        // For any other error, try the next model
        console.log(`Unhandled error with model ${modelName}, trying next model`);
        return callHuggingFaceAPI(inputs, 0, modelIndex + 1);
    }
}

// Apply rate limiter to the API endpoint
app.post('/api/huggingface', rateLimiter, async (req, res) => {
    try {
        console.log('Received request to /api/huggingface');

        const { inputs } = req.body;

        if (!inputs) {
            return res.status(400).json({
                error: 'Bad Request',
                message: 'The "inputs" field is required'
            });
        }

        // Check input length to avoid token limit issues
        if (inputs.length > 2000) {
            return res.status(400).json({
                error: 'Input too long',
                message: 'The input text exceeds the maximum allowed length'
            });
        }

        // Make the actual request to Hugging Face API with fallback support
        try {
            const data = await callHuggingFaceAPI(inputs);
            res.json(data);
        } catch (apiError) {
            console.log('All models failed:', apiError.message);

            // If all models failed, return a clear error
            res.status(503).json({
                error: 'Service Unavailable',
                message: 'All AI models are currently unavailable. Please try again later.',
                retry: true
            });
        }
    } catch (error) {
        console.error('Unhandled error:', error);

        // Provide detailed error response
        res.status(500).json({
            error: 'Failed to process request',
            message: error.message || 'An unexpected error occurred',
            retry: true
        });
    }
});

// Add a pre-warming endpoint that can be called by a scheduler
app.get('/api/warmup', async (req, res) => {
    try {
        const result = await callHuggingFaceAPI('You are Dermi, a friendly dermatology assistant. Provide brief, helpful answers about skin health.');
        res.json({
            status: 'success',
            message: `Model ${result.model_used} warmed up successfully`
        });
    } catch (error) {
        res.status(500).json({
            status: 'error',
            message: 'Failed to warm up all models',
            error: error.message
        });
    }
});

// Add a status endpoint to check which model is currently active
app.get('/api/status', (req, res) => {
    res.json({
        status: 'online',
        current_model: CURRENT_MODEL,
        model_loaded: modelSuccessfullyLoaded,
        consecutive_failures: consecutiveFailures
    });
});

// Periodically clean up the rate limiter data
setInterval(() => {
    const now = Date.now();
    Object.keys(requestCounts).forEach(ip => {
        if (now > requestCounts[ip].resetTime) {
            delete requestCounts[ip];
        }
    });
}, RATE_LIMIT_WINDOW);

app.listen(PORT, () => {
    console.log(`Proxy server running on port ${PORT}`);
    console.log(`Starting with model: ${CURRENT_MODEL}`);

    // Try warming up the model on startup
    setTimeout(async () => {
        console.log('Attempting initial model warm-up');
        try {
            const result = await callHuggingFaceAPI(
                'You are Dermi, a dermatology assistant. Your job is to provide helpful information about skin health and the Dermi app. Keep responses brief and professional.'
            );
            console.log(`Initial model ${result.model_used} warm-up successful!`);
        } catch (error) {
            console.error('All warm-up attempts failed:', error.message);
        }
    }, 2000); // Wait 2 seconds after startup before warming up
});