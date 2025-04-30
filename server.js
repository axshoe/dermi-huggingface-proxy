const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Enable CORS for all routes
app.use(cors());

// Parse JSON request bodies
app.use(express.json({ limit: '1mb' })); // Increase limit for larger prompts

// Simple health check endpoint
app.get('/', (req, res) => {
    res.json({ status: 'Dermi Hugging Face Proxy API is running' });
});

// Global variables to track model loading status
let modelLoadingInProgress = false;
let lastModelLoadAttempt = 0;
let modelSuccessfullyLoaded = false;

// Define the model to use - USING A FASTER LOADING MODEL
const MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'; // Much faster than phi-2

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

// Improved function to format TinyLlama inputs correctly
function formatPromptForTinyLlama(inputs) {
    // TinyLlama uses a specific format for chat
    return `<|im_start|>system
${inputs}
<|im_end|>
<|im_start|>assistant
`;
}

// Extract response from TinyLlama format
function extractTinyLlamaResponse(response) {
    // Try to extract text from TinyLlama's standard format first
    const assistantMatch = response.match(/<\|im_start\|>assistant\n([\s\S]*?)(?:<\|im_end\|>|$)/);
    if (assistantMatch && assistantMatch[1]) {
        return assistantMatch[1].trim();
    }

    // If we can't find the standard format, try to extract just the assistant response
    // Look for Dermi: pattern as a fallback
    const dermiMatch = response.match(/Dermi:([\s\S]*?)(?:\nUser:|$)/i);
    if (dermiMatch && dermiMatch[1]) {
        return dermiMatch[1].trim();
    }

    // Last resort - just return whatever is there, trimmed
    return response.trim();
}

// Hugging Face API calling function with better error handling
async function callHuggingFaceAPI(inputs) {
    const HUGGING_FACE_API_KEY = process.env.HUGGINGFACE_API_KEY;

    if (!HUGGING_FACE_API_KEY) {
        console.error('Missing Hugging Face API key');
        throw new Error('Server configuration error: Missing API key');
    }

    try {
        console.log(`Sending request to Hugging Face API for model: ${MODEL_NAME}`);
        console.log(`Input length: ${inputs.length} characters`);

        // Format the input correctly for TinyLlama
        const formattedInput = formatPromptForTinyLlama(inputs);

        const response = await axios.post(
            `https://api-inference.huggingface.co/models/${MODEL_NAME}`,
            {
                inputs: formattedInput,
                parameters: {
                    max_new_tokens: 400,
                    temperature: 0.7,
                    top_p: 0.9,
                    do_sample: true,
                    return_full_text: false  // Only return new text
                }
            },
            {
                headers: {
                    'Authorization': `Bearer ${HUGGING_FACE_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                timeout: 45000 // 45 seconds
            }
        );

        console.log('Received successful response from Hugging Face API');
        console.log('Response preview:',
            JSON.stringify(response.data).substring(0, 150) + '...');

        modelLoadingInProgress = false;
        modelSuccessfullyLoaded = true;

        // Parse the response based on its format
        let rawResponse = '';

        console.log('Response data type:', typeof response.data);
        if (typeof response.data === 'object') {
            console.log('Response data keys:', Object.keys(response.data));
        }

        if (Array.isArray(response.data)) {
            console.log('Response is an array of length:', response.data.length);
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

        // Now extract the actual response text
        const generatedText = extractTinyLlamaResponse(rawResponse);

        console.log('Extracted response:', generatedText.substring(0, 100) + '...');

        // Return in consistent format
        return { generated_text: generatedText };
    } catch (error) {
        // Enhanced error logging
        console.error('Hugging Face API Error Details:', {
            message: error.message,
            code: error.code,
            status: error.response?.status,
            statusText: error.response?.statusText,
            data: error.response?.data,
        });

        // Handle model loading errors
        if (error.response && error.response.data &&
            error.response.data.error &&
            (error.response.data.error.includes('loading') ||
                error.response.data.error.includes('currently loading'))) {

            console.log(`Model ${MODEL_NAME} loading in progress at Hugging Face`);
            modelLoadingInProgress = true;
            lastModelLoadAttempt = Date.now();
            throw { code: 'MODEL_LOADING', message: error.response.data.error };
        }

        // Special handling for timeouts with clear error code
        if (error.code === 'ECONNABORTED') {
            throw {
                code: 'TIMEOUT',
                message: 'Request to AI model timed out. Try a shorter message or try again later.'
            };
        }

        // For any other error, rethrow with additional context
        throw error;
    }
}

// Apply rate limiter to the API endpoint
app.post('/api/huggingface', rateLimiter, async (req, res) => {
    try {
        console.log('Received request to /api/huggingface', {
            contentType: req.headers['content-type'],
            bodySize: JSON.stringify(req.body).length,
            hasInputs: !!req.body.inputs
        });

        const { inputs } = req.body;

        if (!inputs) {
            console.log('Missing inputs field in request body', req.body);
            return res.status(400).json({
                error: 'Bad Request',
                message: 'The "inputs" field is required'
            });
        }

        // Check input length to avoid token limit issues
        if (inputs.length > 2500) { // Reduced from 4000 to improve reliability
            return res.status(400).json({
                error: 'Input too long',
                message: 'The input text exceeds the maximum allowed length'
            });
        }

        // Check if we need to wait because a model loading is in progress
        const now = Date.now();
        if (modelLoadingInProgress && (now - lastModelLoadAttempt) < 10000) {
            return res.status(503).json({
                error: 'Model loading in progress',
                message: `The AI model ${MODEL_NAME} is currently loading. Please try again in a moment.`,
                retry: true
            });
        }

        // Reset the model loading flag if it's been a while since the last attempt
        if (modelLoadingInProgress && (now - lastModelLoadAttempt) >= 10000) {
            modelLoadingInProgress = false;
        }

        // If the model has never been successfully loaded, we might need a "warm-up" call
        if (!modelSuccessfullyLoaded) {
            try {
                // Make a simple warming request to load the model
                console.log('Attempting model warm-up');
                await callHuggingFaceAPI('You are Dermi, a dermatology assistant. Keep responses brief.');
                console.log('Model warm-up successful');
            } catch (warmupError) {
                console.error('Warm-up error:', warmupError);
                // If it's a loading error, we can continue with the actual request
                if (warmupError.code !== 'MODEL_LOADING') {
                    throw warmupError;
                }
            }
        }

        // Make the actual request to Hugging Face API
        try {
            const data = await callHuggingFaceAPI(inputs);

            // Ensure we're sending a consistent response format for the frontend
            if (typeof data === 'string') {
                // If the response is a string, wrap it in an object
                res.json({ generated_text: data });
            } else {
                // Otherwise, just send the object as is
                res.json(data);
            }
        } catch (apiError) {
            console.log('API Error caught:', apiError.code, apiError.message);

            if (apiError.code === 'MODEL_LOADING') {
                return res.status(503).json({
                    error: 'Model loading',
                    message: `The AI model ${MODEL_NAME} is still warming up. Please try again in a moment.`,
                    retry: true
                });
            }

            // Handle timeouts specifically
            if (apiError.code === 'TIMEOUT' || apiError.code === 'ECONNABORTED') {
                return res.status(504).json({
                    error: 'Gateway Timeout',
                    message: 'The request to the AI model timed out. Please try again with a shorter message.',
                    retry: true
                });
            }

            // For any other error, return appropriate status code and error details
            const statusCode = apiError.response?.status || 500;
            res.status(statusCode).json({
                error: 'Failed to process request',
                message: apiError.message || 'An unexpected error occurred',
                details: apiError.response?.data || null,
                retry: true // Add retry flag to encourage client to retry
            });
        }

    } catch (error) {
        console.error('Unhandled error:', error);

        // Provide detailed error response
        res.status(500).json({
            error: 'Failed to process request',
            message: error.message || 'An unexpected error occurred',
            stack: process.env.NODE_ENV === 'development' ? error.stack : undefined,
            retry: true // Add retry flag to encourage client to retry
        });
    }
});

// Add a pre-warming endpoint that can be called by a scheduler
app.get('/api/warmup', async (req, res) => {
    try {
        await callHuggingFaceAPI('You are Dermi, a friendly dermatology assistant. Provide brief, helpful answers about skin health.');
        res.json({ status: 'success', message: `Model ${MODEL_NAME} warmed up successfully` });
    } catch (error) {
        if (error.code === 'MODEL_LOADING') {
            res.status(202).json({
                status: 'pending',
                message: `Model ${MODEL_NAME} is loading, warm-up request acknowledged`
            });
        } else {
            res.status(500).json({
                status: 'error',
                message: 'Failed to warm up model',
                error: error.message
            });
        }
    }
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
    console.log(`Using model: ${MODEL_NAME}`);

    // Try warming up the model on startup with multiple retries
    setTimeout(async () => {
        console.log('Attempting initial model warm-up');
        // Try up to 3 times with increasing delays
        for (let i = 0; i < 3; i++) {
            try {
                await callHuggingFaceAPI('You are Dermi, a dermatology assistant. Your job is to provide helpful information about skin health and the Dermi app. Keep responses brief and professional.');
                console.log(`Initial model ${MODEL_NAME} warm-up successful!`);
                break; // Exit loop on success
            } catch (error) {
                console.log(`Warm-up attempt ${i+1} failed, waiting before retrying...`);
                if (i < 2) { // Don't wait after the last attempt
                    await new Promise(resolve => setTimeout(resolve, 5000 * (i+1))); // Increasing wait
                }
            }
        }
    }, 2000); // Wait 2 seconds after startup before warming up
});