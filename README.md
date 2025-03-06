# Opik RAG Example with Weaviate

This project demonstrates a Retrieval-Augmented Generation (RAG) system using Opik for tracing and Weaviate as the vector database.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export WEAVIATE_CLUSTER_URL="your-weaviate-cluster-url"
export WEAVIATE_API_KEY="your-weaviate-api-key"
```

3. Run the application:
```bash
python app.py
```

## Features

- Opik integration for tracing LLM calls
- Weaviate vector database integration for semantic search
- OpenAI API integration for generating responses
- Book recommendation system based on user queries

## Project Structure

- `app.py`: Main application file containing the RAG implementation
- `requirements.txt`: Project dependencies
- `README.md`: This file

## Notes

- Make sure you have an Opik account and API key from [Comet](https://www.comet.com/site/products/opik/)
- The Weaviate database should have a "Book" collection set up with appropriate schema
- The application uses the o3-mini model from OpenAI
