# API Documentation

## Base URL

All endpoints are relative to the base URL: `http://localhost:8000/v1/api`

## API Documentation URL

You can find the full documentation of the API at: `http://localhost:8000/docs` or `http://localhost:8000/redoc`

## Endpoints

### 1. FAISS-based Recommendations

**Endpoint:** `/recommend/faiss`

**Method:** POST

**Request Body:**
```json
{
  "description": "string",
  "num_recommendations": "int"
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "title": "string",
      "authors": ["string"],
      "description": "string",
      "similarity": "float"
    }
  ]
}
```

**Description:** This endpoint uses FAISS (Facebook AI Similarity Search) to find books similar to the provided description. It processes the input description using Langchain, generates an embedding, and then uses FAISS to find the most similar books in the database.

### 2. Cosine Similarity-based Recommendations

**Endpoint:** `/recommend/cosine`

**Method:** POST

**Request Body:**
```json
{
  "description": "string",
  "num_recommendations": "int"
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "title": "string",
      "authors": ["string"],
      "description": "string",
      "similarity": "float"
    }
  ]
}
```

**Description:** This endpoint uses Cosine Similarity to find books similar to the provided description. It processes the input description using Langchain, generates an embedding, and then calculates the cosine similarity between this embedding and the embeddings of all books in the database to find the most similar ones.

## Error Handling

In case of any errors, the API will return an appropriate HTTP status code along with a JSON response containing the error details. Common errors include:

- `400 Bad Request`: The request was invalid. This can happen due to missing or invalid parameters.
- `422 Unprocessable Entity`: The server understands the content type of the request entity, and the syntax of the request entity is correct, but it was unable to process the contained instructions.
- `500 Internal Server Error`: An unexpected error occurred on the server.

Example error response:
```json
{
  "detail": "Error message"
}
```

For a more detailed explanation of error codes and troubleshooting steps, please refer to the API documentation at `http://localhost:8000/docs`.