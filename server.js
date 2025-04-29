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
// Add a flag to track if model has been successfully loaded at least once
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

// Hugging Face API calling function with better error handling
async function callHuggingFaceAPI(inputs) {
    const HUGGING_FACE_API_KEY = process.env.HUGGINGFACE_API_KEY;

    if (!HUGGING_FACE_API_KEY) {
        console.error('Missing Hugging Face API key');
        throw new Error('Server configuration error: Missing API key');
    }

    try {
        console.log(`Sending request to Hugging Face API for model: ${MODEL_NAME}`);

        const response = await axios.post(
            `https://api-inference.huggingface.co/models/${MODEL_NAME}`,
            {
                inputs,
                parameters: {
                    max_new_tokens: 256,
                    temperature: 0.7,
                    top_p: 0.9,
                    do_sample: true
                }
            },
            {
                headers: {
                    'Authorization': `Bearer ${HUGGING_FACE_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                timeout: 20000 // 20 second timeout (faster model needs less time)
            }
        );

        console.log('Received successful response from Hugging Face API');
        modelLoadingInProgress = false;
        modelSuccessfullyLoaded = true;
        return response.data;
    } catch (error) {
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

        // For any other error, log it and rethrow
        console.error('Error from Hugging Face API:', error.message);
        throw error;
    }
}

// Apply rate limiter to the API endpoint
app.post('/api/huggingface', rateLimiter, async (req, res) => {
    try {
        const { inputs } = req.body;

        if (!inputs) {
            return res.status(400).json({
                error: 'Bad Request',
                message: 'The "inputs" field is required'
            });
        }

        // Check input length to avoid token limit issues
        if (inputs.length > 4000) {
            return res.status(400).json({
                error: 'Input too long',
                message: 'The input text exceeds the maximum allowed length'
            });
        }

        // Check if we need to wait because a model loading is in progress
        const now = Date.now();
        if (modelLoadingInProgress && (now - lastModelLoadAttempt) < 10000) { // Reduced wait time for faster model
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
                await callHuggingFaceAPI('Hello');
                console.log('Model warm-up successful');
            } catch (warmupError) {
                // If it's a loading error, we can continue with the actual request
                // Otherwise, we'll handle the error below
                if (warmupError.code !== 'MODEL_LOADING') {
                    throw warmupError;
                }
            }
        }

        // Make the actual request to Hugging Face API
        try {
            const data = await callHuggingFaceAPI(inputs);
            res.json(data);
        } catch (apiError) {
            if (apiError.code === 'MODEL_LOADING') {
                return res.status(503).json({
                    error: 'Model loading',
                    message: `The AI model ${MODEL_NAME} is still warming up. Please try again in a moment.`,
                    retry: true
                });
            }

            // Handle timeouts specifically
            if (apiError.code === 'ECONNABORTED') {
                return res.status(504).json({
                    error: 'Gateway Timeout',
                    message: 'The request to the AI model timed out. Please try again with a shorter message.'
                });
            }

            // For any other error
            throw apiError;
        }

    } catch (error) {
        console.error('Unhandled error:', error.message);

        // Generic error handler
        res.status(error.response?.status || 500).json({
            error: 'Failed to process request',
            message: error.message || 'An unexpected error occurred'
        });
    }
});

// Add a pre-warming endpoint that can be called by a scheduler
app.get('/api/warmup', async (req, res) => {
    try {
        await callHuggingFaceAPI('Hello, I am warming up the model.');
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
                await callHuggingFaceAPI('Hello, this is a warm-up message.');
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