import axios from 'axios';

// Create an axios instance with default configuration
const api = axios.create({
  baseURL: 'http://localhost:5000',
  timeout: 600000, // 10 minutes timeout for large file uploads
  headers: {
    'Content-Type': 'application/json',
  }
});

export default api; 