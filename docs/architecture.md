# System Architecture

The Book Recommendation System is built using FastAPI and follows a modular architecture. Here's an overview of the main components and their interactions:

## Components

1. **FastAPI Application (`app/main.py`)**: 
   - Entry point of the application
   - Configures and starts the FastAPI server
   - Includes API routes

2. **API Endpoints (`app/api/endpoints.py`)**: 
   - Defines the API routes
   - Handles incoming requests and returns responses
   - Interacts with the RecommendationService

3. **Book Service (`app/services/book_service.py`)**: 
   - Handles book data operations
   - Loads and cleans book data
   - Creates and manages the Pandas DataFrame

4. **Processing Chain (`app/services/processing_chain.py`)**: 
   - Implements the Langchain processing flow
   - Creates embeddings for book descriptions
   - Possibility of processing user queries

5. **Recommendation Service (`app/services/recommendation_service.py`)**: 
   - Manages the recommendation logic
   - Uses FAISS and Cosine Similarity for recommendations
   - Interacts with the Book Service and Processing Chain

6. **Book Models (`app/models/book.py`)**: 
   - Defines the data models used for book recommendations.

7. **Configuration (`app/core/config.py`)**: 
   - Manages application settings
   - Loads environment variables

8. **Logging (`app/core/logger.py`)**: 
   - Configures loggers for different parts of the application (API, services, and main application flow).
   - Log files are stored in the `logs/` directory, making it easy to access and review them.
   - Each logger can be configured with different levels and formats to suit the needs of various components.

9. **Data Population Script (`populate_data.py`)**:
   - Script to populate the initial data for the application
   - Fetches and processes book data from Google Books API
   - Inserts processed data into data/ folder (books.json, Books DataFrame, FAISS Index and embeddings.npy)

## Data Flow

The data flow within the Book Recommendation System is designed to efficiently process and recommend books based on user queries. Here's a step-by-step overview:

1. **Data Collection**: The system can start by collecting book data from the Google Books API. This initial dataset includes various attributes of books, including descriptions.

2. **Data Preparation**: Once the data is collected, it undergoes a cleaning process using Pandas and NLTK for NLP tasks such as tokenization, stopword removal, and lemmatization. This step ensures that the data is in a suitable format for further processing.

3. **Embedding Generation**: The cleaned book descriptions are then processed by the OpenAI API to generate embeddings. These embeddings capture the semantic essence of the book descriptions, making them suitable for similarity comparisons.

4. **FAISS Index Creation**: With the embeddings generated, a FAISS index is created for efficient similarity search. This index allows the system to quickly find books with descriptions similar to a user query.

5. **User Query Processing**: When a user submits a query, it can be first processed through Langchain to capture the essence of the request. This involves rephrasing the query to align more closely with the context of book descriptions.

6. **Recommendation Generation**: Based on the processed query / description, the system then uses either the FAISS index or Cosine Similarity to find and recommend books that match the user's interests.

7. **Response**: The recommended books are returned to the user through the FastAPI application, completing the data flow cycle.

## External Dependencies

- **OpenAI API**: Used for generating embeddings and processing natural language.
- **Google Books API**: Used to populate initial book data.
- **FAISS**: Used for efficient similarity search.
- **Pandas**: Used for data manipulation and storage.
- **NLTK**: Used for natural language processing tasks in data cleaning.

## Scalability and Performance Considerations

- The FAISS index allows for efficient similarity search, even with a large number of books.
- Embeddings are cached to disk to avoid unnecessary API calls to OpenAI.
- The Pandas DataFrame is also cached to disk for faster loading.
- The modular architecture allows for easy scaling of individual components as needed.

## Future Improvements

- Implement a database for more efficient data storage and retrieval.
- Add a caching layer for frequently requested recommendations.
- Implement user feedback and collaborative filtering for improved recommendations.
- Add more advanced NLP techniques for better understanding of book descriptions and user queries.