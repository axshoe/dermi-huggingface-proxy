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

// Track model loading status
let modelLoadingInProgress = false;
let lastModelLoadAttempt = 0;

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
        if (modelLoadingInProgress && (now - lastModelLoadAttempt) < 20000) {
            return res.status(503).json({
                error: 'Model loading in progress',
                message: 'The AI model is currently loading. Please try again in a moment.',
                retry: true
            });
        }

        // Make request to Hugging Face API with increased timeout
        console.log('Sending request to Hugging Face API');
        const response = await axios.post(
            'https://api-inference.huggingface.co/models/microsoft/phi-2',
            { inputs, parameters: { max_new_tokens: 256 } }, // Limit output tokens
            {
                headers: {
                    'Authorization': `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                timeout: 60000 // 60 second timeout
            }
        );

        console.log('Received response from Hugging Face API');
        // Reset model loading flag on successful response
        modelLoadingInProgress = false;

        // Return the data from Hugging Face
        res.json(response.data);

    } catch (error) {
        console.error('Error calling Hugging Face API:', error.message);

        // Handle model loading errors (common with Hugging Face)
        if (error.response && error.response.data &&
            error.response.data.error &&
            (error.response.data.error.includes('loading') ||
                error.response.data.error.includes('currently loading'))) {

            modelLoadingInProgress = true;
            lastModelLoadAttempt = Date.now();

            return res.status(503).json({
                error: 'Model loading',
                message: 'The AI model is still warming up. Please try again in a moment.',
                retry: true
            });
        }

        // Handle timeouts specifically
        if (error.code === 'ECONNABORTED') {
            return res.status(504).json({
                error: 'Gateway Timeout',
                message: 'The request to the AI model timed out. Please try again with a shorter message.'
            });
        }

        // Generic error handler
        res.status(error.response?.status || 500).json({
            error: 'Failed to process request',
            message: error.message
        });
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
});