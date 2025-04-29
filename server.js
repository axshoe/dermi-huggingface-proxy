const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Enable CORS for all routes
app.use(cors());

// Parse JSON request bodies
app.use(express.json());

// Simple health check endpoint
app.get('/', (req, res) => {
    res.json({ status: 'Dermi Hugging Face Proxy API is running' });
});

app.post('/api/huggingface', async (req, res) => {
    try {
        const { inputs } = req.body;

        // Make request to Hugging Face API
        const response = await axios.post(
            'https://api-inference.huggingface.co/models/microsoft/phi-2',
            { inputs },
            {
                headers: {
                    'Authorization': `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                timeout: 30000
            }
        );

        // Return the data from Hugging Face
        res.json(response.data);
    } catch (error) {
        console.error('Error calling Hugging Face API:', error);
        res.status(500).json({
            error: 'Failed to process request',
            message: error.message
        });
    }
});

app.listen(PORT, () => {
    console.log(`Proxy server running on port ${PORT}`);
});