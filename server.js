const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 10000;

// Message keys for internationalization
const MESSAGE_KEYS = {
    // Error message keys
    RATE_LIMIT_EXCEEDED: 'RATE_LIMIT_EXCEEDED',
    INPUT_REQUIRED: 'INPUT_REQUIRED',
    INPUT_TOO_LONG: 'INPUT_TOO_LONG',
    SERVER_ERROR: 'SERVER_ERROR',
    API_KEY_MISSING: 'API_KEY_MISSING',
    ALL_MODELS_FAILED: 'ALL_MODELS_FAILED',

    // Default response keys
    EMPTY_RESPONSE_FALLBACK: 'EMPTY_RESPONSE_FALLBACK',
    CONNECTION_TROUBLE: 'CONNECTION_TROUBLE',
    ASK_DIFFERENTLY: 'ASK_DIFFERENTLY',
    HELP_WITH_SKIN: 'HELP_WITH_SKIN',
    HELP_WITH_SKIN_SPECIFIC: 'HELP_WITH_SKIN_SPECIFIC'
};

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
let CURRENT_MODEL = process.env.PRIMARY_MODEL || 'mistralai/Mistral-7B-Instruct-v0.1';

// Define models in order of preference - prioritizing more capable models first
const MODELS = [
    'facebook/opt-1.3b',                   // Fallback #2 - slightly larger OPT model
    'facebook/opt-350m',                   // Fallback #3 - small but reliable
    'google/flan-t5-small',                // Last resort - smaller but reliable
    'mistralai/Mistral-7B-Instruct-v0.1',
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
            messageKey: MESSAGE_KEYS.RATE_LIMIT_EXCEEDED,
            fallbackMessage: 'Too many requests. Please try again later.'
        });
    }

    next();
}

// IMPROVED: Better prompt formatting based on model type
function formatPrompt(inputs, modelName, language = 'en') {
    // Extract the user's question when possible
    let userQuestion = "";
    const userMatch = inputs.match(/User:\s*(.*?)(\n|$)/);
    if (userMatch && userMatch[1]) {
        userQuestion = userMatch[1].trim();
    } else {
        // Try to get the last line if there's no User: prefix
        const lines = inputs.split('\n');
        for (let i = lines.length - 1; i >= 0; i--) {
            if (lines[i].trim() && !lines[i].includes('Assistant:') && !lines[i].includes('Dermi:')) {
                userQuestion = lines[i].trim();
                break;
            }
        }
    }

    // If we still couldn't extract it, use the full input
    if (!userQuestion) {
        userQuestion = inputs;
    }

    // Format differently based on model type
    if (modelName.includes('mistral')) {
        // For Mistral, use chat format
        const systemPrompt = getSystemPromptForLanguage(language);
        return `<s>[INST] ${systemPrompt}

User question: ${userQuestion} [/INST]</s>`;
    } else if (modelName.includes('phi-2')) {
        // For phi-2, use a structured prompt format
        return `<|USER|>You are Dermi, a friendly dermatology AI assistant.
Answer this question about skin health: ${userQuestion}<|ASSISTANT|>`;
    } else if (modelName.includes('flan-t5')) {
        // T5 models work well with task prefixes
        return `Answer this dermatology question concisely as a helpful skin health assistant: ${userQuestion}`;
    } else if (modelName.includes('opt')) {
        // OPT models work with simple prompts but need clear instruction
        return `Human: ${userQuestion}\n\nDermi (a dermatology assistant):`;
    } else {
        // Default format
        return `Q: ${userQuestion}\nA:`;
    }
}

function getSystemPromptForLanguage(language) {
    switch (language) {
        case 'es':
            return `Eres Dermi, un asistente amigable de salud de la piel en una aplicación de dermatología.
Pautas:
1. Solo discute temas de salud de la piel
2. Nunca diagnostiques - siempre recomienda ver a un médico
3. Mantén las respuestas breves y útiles (2-4 oraciones)
4. Proporciona información médica precisa`;
        case 'zh':
            return `你是Dermi，皮肤科应用程序中的友好皮肤健康助手。
指导原则：
1. 只讨论皮肤健康话题
2. 永远不要诊断 - 总是建议看医生
3. 保持答案简洁有用（2-4句话）
4. 提供准确的医疗信息`;
        case 'en':
        default:
            return `You are Dermi, a friendly skin health assistant in a dermatology app.
Guidelines:
1. Only discuss skin health topics
2. Never diagnose - always recommend seeing a doctor
3. Keep answers brief and helpful (2-4 sentences)
4. Provide accurate medical information`;
    }
}

// IMPROVED: Better response extraction with message keys for fallbacks
function extractModelResponse(response, modelName) {
    // First check if we received anything
    if (!response || response.length < 1) {
        return {
            messageKey: MESSAGE_KEYS.ASK_DIFFERENTLY,
            fallbackText: "I apologize, but I couldn't generate a proper response. Could you try asking your question differently?"
        };
    }

    // Handle different model output formats
    let cleanedResponse = response;

    console.log("Raw model response:", response.substring(0, 150));

    if (modelName.includes('mistral')) {
        // Clean up Mistral responses
        cleanedResponse = cleanedResponse.replace(/<\/?s>/g, '');
        cleanedResponse = cleanedResponse.replace(/\[INST\].*?\[\/INST\]/s, '');
    } else if (modelName.includes('phi-2')) {
        // Clean up phi-2 responses
        cleanedResponse = cleanedResponse.replace(/<\|USER\|>.*?<\|ASSISTANT\|>/s, '');
        cleanedResponse = cleanedResponse.replace(/<\|.*?\|>/g, '');
    } else if (modelName.includes('flan-t5')) {
        // Clean up flan-t5 responses
        // T5 typically doesn't need much cleaning
    } else if (modelName.includes('opt')) {
        // Clean up OPT responses
        cleanedResponse = cleanedResponse.replace(/^Human:.*?\n\nDermi.*?:/s, '');
        cleanedResponse = cleanedResponse.replace(/^Human:.*$/gm, '');
    }

    // Generic cleaning for all models
    cleanedResponse = cleanedResponse.replace(/^(Assistant|Dermi|A):\s*/i, '');
    cleanedResponse = cleanedResponse.replace(/^Q:.*$/gm, '');

    // Remove any remaining system prompts or instructions
    // Note: Be less aggressive with filtering to avoid removing content
    const systemInstructionPhrases = [
        "You are Dermi, a dermatology assistant",
        "You are a dermatology assistant",
        "I am Dermi, a dermatology assistant",
        "never claim you can diagnose",
        "never diagnose - always recommend seeing a doctor",
        "keep answers brief"
    ];

    // Only remove complete system instructions, not parts of valid responses
    for (const phrase of systemInstructionPhrases) {
        cleanedResponse = cleanedResponse.replace(new RegExp(phrase, 'i'), '');
    }

    cleanedResponse = cleanedResponse.trim();

    // If we have an empty response after cleaning, provide a fallback with message key
    if (!cleanedResponse || cleanedResponse.length < 10) {
        return {
            messageKey: MESSAGE_KEYS.HELP_WITH_SKIN,
            fallbackText: "I'm here to help with your skin health questions. Could you provide more details about your question?"
        };
    }

    // Return successful response (no message key needed)
    return {
        messageKey: null,
        text: cleanedResponse
    };
}

// Improved exponential backoff for retries
async function wait(attemptNumber) {
    const baseDelay = 2000; // 2 seconds
    const maxDelay = 30000; // 30 seconds
    const delay = Math.min(baseDelay * Math.pow(2, attemptNumber), maxDelay);
    console.log(`Waiting ${delay}ms before retry`);
    await new Promise(resolve => setTimeout(resolve, delay));
}

// Get model parameters based on model name
function getModelParameters(modelName) {
    if (modelName.includes('mistral')) {
        return {
            max_new_tokens: 150,
            temperature: 0.7,
            top_p: 0.95,
            do_sample: true,
            return_full_text: false
        };
    } else if (modelName.includes('phi-2')) {
        return {
            max_new_tokens: 150,
            temperature: 0.6,
            top_p: 0.92,
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
            temperature: 0.7,
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
async function callHuggingFaceAPI(inputs, attemptNumber = 0, modelIndex = 0, language = 'en') {
    const HUGGING_FACE_API_KEY = process.env.HUGGINGFACE_API_KEY;

    if (!HUGGING_FACE_API_KEY) {
        console.error('Missing Hugging Face API key');
        throw new Error(MESSAGE_KEYS.API_KEY_MISSING);
    }

    // Get the model to try
    if (modelIndex >= MODELS.length) {
        throw new Error(MESSAGE_KEYS.ALL_MODELS_FAILED);
    }

    const modelName = MODELS[modelIndex];
    CURRENT_MODEL = modelName;

    try {
        console.log(`Attempt ${attemptNumber + 1} with model ${modelName}`);
        console.log(`Input length: ${inputs.length} characters`);

        // Format the input correctly for the chosen model
        const formattedInput = formatPrompt(inputs, modelName, language);
        const parameters = getModelParameters(modelName);

        console.log(`Formatted prompt for ${modelName}: ${formattedInput.substring(0, 100)}...`);

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
        const extractedResponse = extractModelResponse(rawResponse, modelName);

        // Check if we got a message key response (fallback)
        if (extractedResponse.messageKey) {
            return {
                generated_text: {
                    messageKey: extractedResponse.messageKey,
                    fallbackText: extractedResponse.fallbackText
                },
                model_used: modelName
            };
        }

        // Check if response is too short and provide a better response
        if (extractedResponse.text && extractedResponse.text.length < 15) {
            console.log("Response too short, returning default response");
            return {
                generated_text: {
                    messageKey: MESSAGE_KEYS.HELP_WITH_SKIN_SPECIFIC,
                    fallbackText: "I'm here to help with skin health questions. Could you please ask about a specific skin condition or how to use the Dermi app?"
                },
                model_used: modelName
            };
        }

        // Return successful response
        return {
            generated_text: extractedResponse.text,
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
            if (attemptNumber < 2) {
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

        const { inputs, language = 'en' } = req.body;

        if (!inputs) {
            return res.status(400).json({
                error: 'Bad Request',
                messageKey: MESSAGE_KEYS.INPUT_REQUIRED,
                fallbackMessage: 'The "inputs" field is required'
            });
        }

        // Check input length to avoid token limit issues
        if (inputs.length > 2000) {
            return res.status(400).json({
                error: 'Input too long',
                messageKey: MESSAGE_KEYS.INPUT_TOO_LONG,
                fallbackMessage: 'The input text exceeds the maximum allowed length'
            });
        }

        // Make the actual request to Hugging Face API with fallback support
        try {
            const data = await callHuggingFaceAPI(inputs, 0, 0, language);

            // Check if we got a message key response (fallback/error case)
            if (data.generated_text && typeof data.generated_text === 'object' && data.generated_text.messageKey) {
                console.log('Returning message key response:', data.generated_text.messageKey);
                res.json({
                    messageKey: data.generated_text.messageKey,
                    fallbackText: data.generated_text.fallbackText,
                    model_used: data.model_used
                });
            } else {
                // Normal successful response
                console.log('Final response sent to client:', data.generated_text.substring(0, 150));
                res.json(data);
            }

        } catch (apiError) {
            console.log('All models failed:', apiError.message);

            // Check if error message is a message key
            if (Object.values(MESSAGE_KEYS).includes(apiError.message)) {
                res.json({
                    messageKey: apiError.message,
                    fallbackText: apiError.message === MESSAGE_KEYS.API_KEY_MISSING
                        ? "Service temporarily unavailable."
                        : "Unable to process your request right now.",
                    model_used: "fallback_response"
                });
            } else {
                // If all models failed, return a default response with message key
                res.json({
                    messageKey: MESSAGE_KEYS.CONNECTION_TROUBLE,
                    fallbackText: "I'm having trouble connecting to my knowledge base, but I'm here to help with skin health questions. Could you try asking in a different way?",
                    model_used: "fallback_response"
                });
            }

            // Try recovery in background
            attemptServiceRecovery();
        }
    } catch (error) {
        console.error('Unhandled error:', error);

        // Provide detailed error response with message key
        res.status(500).json({
            error: 'Failed to process request',
            messageKey: MESSAGE_KEYS.SERVER_ERROR,
            fallbackMessage: error.message || 'An unexpected error occurred',
            retry: true
        });
    }
});

// Add a pre-warming endpoint that can be called by a scheduler
app.get('/api/warmup', async (req, res) => {
    try {
        const result = await callHuggingFaceAPI('You are Dermi, a friendly dermatology assistant. What can you tell me about sunscreen?');
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

// Debug endpoint to test a specific prompt directly
app.post('/api/debug', async (req, res) => {
    try {
        const { inputs, model_name } = req.body;

        if (!inputs) {
            return res.status(400).json({
                error: 'Inputs are required',
                messageKey: MESSAGE_KEYS.INPUT_REQUIRED,
                fallbackMessage: 'Inputs are required'
            });
        }

        // Allow specifying a model for testing
        let modelIndex = 0;
        if (model_name) {
            modelIndex = MODELS.findIndex(m => m.includes(model_name));
            if (modelIndex === -1) modelIndex = 0; // Default to first if not found
        }

        const result = await callHuggingFaceAPI(inputs, 0, modelIndex);
        res.json({
            ...result,
            prompt_used: formatPrompt(inputs, result.model_used)
        });
    } catch (error) {
        res.status(500).json({
            error: 'Debug request failed',
            messageKey: MESSAGE_KEYS.SERVER_ERROR,
            fallbackMessage: error.message
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
    console.log(`Starting with model: ${CURRENT_MODEL}`);

    // Try warming up the model on startup
    setTimeout(async () => {
        console.log('Attempting initial model warm-up');
        try {
            const result = await callHuggingFaceAPI(
                'What can you tell me about sun exposure and skin health?'
            );
            console.log(`Initial model ${result.model_used} warm-up successful!`);
        } catch (error) {
            console.error('All warm-up attempts failed:', error.message);
            // Attempt service recovery on startup failure
            attemptServiceRecovery();
        }
    }, 2000); // Wait 2 seconds after startup before warming up
});