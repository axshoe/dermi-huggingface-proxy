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
const MAX_CONSECUTIVE_FAILURES = 5;
let serviceRecoveryAttempts = 0;
let CURRENT_MODEL = process.env.PRIMARY_MODEL || 'facebook/opt-350m';

// Define models in order of preference - prioritizing reliable models first
const MODELS = [
    'facebook/opt-350m',         // Primary choice - very small but very reliable
    'google/flan-t5-small',      // Fallback #1 - smaller but reliable
    'microsoft/phi-2',           // Fallback #2 - 2.7B parameter model but less reliable
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

// IMPROVED: Better prompt formatting based on model type
function formatPrompt(inputs, modelName) {
    // First, try to extract just the user's question when possible
    let userQuestion = "";
    const userMatch = inputs.match(/User:\s*(.*?)(\n|$)/);
    if (userMatch && userMatch[1]) {
        userQuestion = userMatch[1].trim();
    }

    // If we couldn't extract it, use the full input
    if (!userQuestion) {
        userQuestion = inputs;
    }

    // Extract system instructions if present (for context)
    let systemInstructions = "";
    if (inputs.includes("You are Dermi")) {
        const systemMatch = inputs.match(/You are Dermi.*?(?=\n\n|$)/s);
        if (systemMatch) {
            systemInstructions = systemMatch[0];
        }
    }

    // Format differently based on model type
    if (modelName.includes('phi-2')) {
        // For phi-2, use a structured prompt format
        return `<INST>You are Dermi, a friendly dermatology AI assistant. 
Always respond about skin health topics concisely.
Never say you can diagnose conditions - always recommend seeing a doctor.

${userQuestion}</INST>`;
    } else if (modelName.includes('flan-t5')) {
        // T5 models work well with task prefixes
        return `Answer this dermatology question concisely as a helpful skin health assistant: ${userQuestion}`;
    } else if (modelName.includes('opt')) {
        // OPT models work with simple prompts but need clear instruction
        return `Q: ${userQuestion}\nA: [As a skin health assistant, remember to never claim you can diagnose and keep your answer brief]`;
    } else {
        // Default format
        return `You are Dermi, a skin health assistant. Answer this question briefly: ${userQuestion}`;
    }
}

// IMPROVED: Better response extraction
function extractModelResponse(response, modelName) {
    // First, clean up any wrapper text
    let cleanedResponse = response;

    // Handle different model output formats
    if (modelName.includes('phi-2')) {
        // Clean up phi-2 responses
        cleanedResponse = cleanedResponse.replace(/\nUser:.*$/s, '');
        cleanedResponse = cleanedResponse.replace(/^Assistant:\s*/i, '');
        cleanedResponse = cleanedResponse.replace(/^Dermi:\s*/i, '');
        cleanedResponse = cleanedResponse.replace(/<\/?INST>/, '');
    } else if (modelName.includes('flan-t5') || modelName.includes('opt')) {
        // These models typically return just the answer
        cleanedResponse = cleanedResponse.replace(/^A:\s*(\[.*?\]\s*)?/i, '');
        cleanedResponse = cleanedResponse.replace(/^Q:.*$/gm, ''); // Remove question reflections
    }

    // Further clean up any system instructions that might have leaked
    const systemMarkers = [
        "You are Dermi",
        "friendly dermatology AI",
        "ABCDE rule",
        "skin health assistant",
        "never claim you can diagnose",
        "keep your answer brief"
    ];

    // Go through and remove any lines containing system instructions
    const lines = cleanedResponse.split('\n');
    const filteredLines = lines.filter(line => {
        // Keep the line if it doesn't contain any system markers
        return !systemMarkers.some(marker =>
            line.toLowerCase().includes(marker.toLowerCase())
        );
    });

    cleanedResponse = filteredLines.join('\n').trim();

    // If we still have system instructions, try a more aggressive approach
    // by finding the start of the actual response
    if (cleanedResponse.includes("dermatology") &&
        (cleanedResponse.includes("assistant") || cleanedResponse.includes("ABCDE"))) {

        const possibleStartPatterns = [
            "Hi", "Hello", "I'd", "I can", "Welcome", "The Dermi app",
            "This app", "No,", "Yes,", "I'm", "To use"
        ];

        for (const pattern of possibleStartPatterns) {
            const index = cleanedResponse.indexOf(pattern);
            if (index > 0) {
                cleanedResponse = cleanedResponse.substring(index);
                break;
            }
        }
    }

    return cleanedResponse.trim();
}

// Improved exponential backoff for retries
async function wait(attemptNumber) {
    const baseDelay = 2000; // 2 seconds (increased from 1)
    const maxDelay = 30000; // 30 seconds
    const delay = Math.min(baseDelay * Math.pow(2, attemptNumber), maxDelay);
    console.log(`Waiting ${delay}ms before retry`);
    await new Promise(resolve => setTimeout(resolve, delay));
}

// Get model parameters based on model name
function getModelParameters(modelName) {
    if (modelName.includes('phi-2')) {
        return {
            max_new_tokens: 150,
            temperature: 0.5,
            top_p: 0.95,
            do_sample: true,
            return_full_text: false
        };
    } else if (modelName.includes('flan-t5')) {
        return {
            max_new_tokens: 120,
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

// Service recovery function
async function attemptServiceRecovery() {
    if (serviceRecoveryAttempts > 3) {
        console.log("Too many recovery attempts, pausing recovery for 5 minutes");
        setTimeout(() => {
            serviceRecoveryAttempts = 0;
            attemptServiceRecovery();
        }, 300000); // 5 minutes
        return;
    }

    serviceRecoveryAttempts++;
    console.log(`Recovery attempt ${serviceRecoveryAttempts}: Trying to warm up a model`);

    // Try all models in sequence until one works
    for (const modelName of MODELS) {
        try {
            console.log(`Trying to recover with model ${modelName}`);
            const response = await axios.post(
                `https://api-inference.huggingface.co/models/${modelName}`,
                {
                    inputs: "You are Dermi, a helpful assistant.",
                    parameters: getModelParameters(modelName)
                },
                {
                    headers: {
                        'Authorization': `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
                        'Content-Type': 'application/json'
                    },
                    timeout: 15000
                }
            );

            if (response.status === 200) {
                console.log(`Recovery successful with model ${modelName}`);
                CURRENT_MODEL = modelName;
                modelSuccessfullyLoaded = true;
                consecutiveFailures = 0;
                serviceRecoveryAttempts = 0;
                return true;
            }
        } catch (error) {
            console.log(`Recovery attempt with ${modelName} failed: ${error.message}`);
        }
    }

    console.log("All recovery attempts failed");
    return false;
}

// IMPROVED: Hugging Face API calling function with better error handling and fallback
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

        // Check if response is too short and provide a better response
        if (generatedText.length < 20) {
            return {
                generated_text: "I'm here to help with skin health questions. Could you please ask about a specific skin condition or how to use the Dermi app?",
                model_used: modelName
            };
        }

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

        // Check if we need to attempt service recovery
        if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
            console.log(`${consecutiveFailures} consecutive failures detected, attempting service recovery`);
            attemptServiceRecovery();
        }

        // Handle model loading errors
        if (error.response?.status === 503 ||
            (error.response?.data?.error && error.response.data.error.includes('loading'))) {

            console.log(`Model ${modelName} unavailable or loading`);
            modelLoadingInProgress = true;
            lastModelLoadAttempt = Date.now();

            // Wait longer between retries for 503 errors
            await wait(attemptNumber + 1);

            // Try more times before switching models
            if (attemptNumber < 3) {
                return callHuggingFaceAPI(inputs, attemptNumber + 1, modelIndex);
            } else {
                // After multiple attempts, try the next model
                return callHuggingFaceAPI(inputs, 0, modelIndex + 1);
            }
        }

        // Special handling for timeouts
        if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
            console.log(`Timeout with model ${modelName}`);

            // Wait before trying the next model
            await wait(0);
            return callHuggingFaceAPI(inputs, 0, modelIndex + 1);
        }

        // For any other error, try the next model
        console.log(`Unhandled error with model ${modelName}, trying next model`);
        await wait(0);
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

            // If all models failed, return a default response instead of error
            res.json({
                generated_text: "I'm having trouble connecting to my knowledge base, but I'm here to help with skin health questions. Could you try again in a few moments?",
                model_used: "fallback_response"
            });

            // Try recovery in background
            attemptServiceRecovery();
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
                'You are Dermi, an AI dermatology assistant on the Dermi skin health scanning app. Your job is to provide helpful information about skin health and the Dermi app. Keep responses brief and professional.'
            );
            console.log(`Initial model ${result.model_used} warm-up successful!`);
        } catch (error) {
            console.error('All warm-up attempts failed:', error.message);
            // Attempt service recovery on startup failure
            attemptServiceRecovery();
        }
    }, 2000); // Wait 2 seconds after startup before warming up
});