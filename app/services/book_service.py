import json
import os
import pandas as pd
import nltk
import requests
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from app.core.config import settings

class BookService:
    def __init__(self):
        self.data_dir = Path(settings.DATA_DIR)
        self.books_file = os.path.join(settings.DATA_DIR, "books.json")
        self.df_file = os.path.join(settings.DATA_DIR, "books_df.pkl")
        
        # Downloading necessary NLTK data for NLP preprocessing
        if not os.path.exists(os.path.join(nltk.data.find('corpora'), 'stopwords')):
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def collect_books(self, query: str = "", max_results: int = 40) -> dict:
        books_data = {}
        if max_results > 40:
            for i in range(0, max_results, 40):
                url = "https://www.googleapis.com/books/v1/volumes"
                params = {
                    "q": query if query else "*",
                    "maxResults": 40,
                    "startIndex": i,
                    "key": settings.GOOGLE_BOOKS_API_KEY
                }
                response = requests.get(url, params=params)
                response.raise_for_status()
                if i == 0:
                    books_data = response.json()
                else:
                    books_data['items'].extend(response.json().get('items', []))
        else:
            url = "https://www.googleapis.com/books/v1/volumes"
            params = {
                "q": query if query else "*",
                "maxResults": max_results,
                "key": settings.GOOGLE_BOOKS_API_KEY
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            books_data = response.json()
        
        return books_data

    def load_books(self):
        with open(self.books_file, 'r') as f:
            return json.load(f)

    def clean_data(self, books):
        cleaned_books = []
        for book in books['items']:
            if 'description' in book['volumeInfo']:
                cleaned_book = {
                    'id': book['id'],
                    'title': book['volumeInfo'].get('title', '') + (' - ' + book['volumeInfo'].get('subtitle', '') if book['volumeInfo'].get('subtitle') else ''),
                    'authors': book['volumeInfo'].get('authors', []),
                    'processed_description': self._clean_text(book['volumeInfo'].get('description', '')),
                    'description': book['volumeInfo'].get('description', '')
                }
                cleaned_books.append(cleaned_book)
        return cleaned_books

    def _clean_text(self, text):
        # Tokenize the text
        tokens = word_tokenize(text.lower())

        # Remove stopwords and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        
        # Lemmatize the tokens
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join the tokens back into a string
        cleaned_text = ' '.join(lemmatized_tokens)
        
        return cleaned_text

    def create_dataframe(self):
        books = self.load_books()
        cleaned_books = self.clean_data(books)
        df = pd.DataFrame(cleaned_books)
        df.to_pickle(self.df_file)
        return df

    def load_dataframe(self):
        if os.path.exists(self.df_file):
            return pd.read_pickle(self.df_file)
        else:
            return self.create_dataframe()