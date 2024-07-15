import os
import json
import numpy as np
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.book_service import BookService
from app.services.processing_chain import ProcessingChain
from app.core.config import settings

book_service = BookService()
processing_chain = ProcessingChain()

predefined_genres = ["science-fiction", "non-fiction", "science", "history", "fantasy", "mystery", "romance", "horror", "biography", "self-help", "technology", "python",
                    "rest-api", "programming", "web-development", "data-science", "machine-learning", "deep-learning", "artificial-intelligence", "cloud-computing"]

def fetch_books(genres):
    all_books = []
    books_per_genre = 2400 // len(genres)
    for genre in genres:
        print(f"Collecting books for genre: {genre}")
        books_data = book_service.collect_books(query=genre, max_results=books_per_genre)
        all_books.extend(books_data.get('items', []))
    return all_books

def append_or_save_books(all_books):
    file_path = 'data/books.json'
    if not os.path.exists(file_path):
        data = {'items': all_books}
    else:
        with open(file_path, 'r') as file:
            data = json.load(file)
            data['items'].extend(all_books)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def save_dataframe_and_embeddings():
    df = book_service.create_dataframe()
    embeddings = processing_chain.create_embeddings(df['processed_description'].tolist())
    df['embeddings'] = embeddings
    np.save(os.path.join(settings.DATA_DIR, "embeddings.npy"), embeddings)
    return df

def save_faiss_index():
    df = book_service.create_dataframe()
    faiss_index = processing_chain.create_faiss_index(df)
    faiss_index.save_local(os.path.join(settings.DATA_DIR, "faiss_index"))
    return faiss_index

if __name__ == "__main__":
    start_time = time.time()
    all_books = fetch_books(predefined_genres)
    print(f"Total books collected: {len(all_books)}")
    append_or_save_books(all_books)
    print("Books saved successfully!")
    df = save_dataframe_and_embeddings()
    print("Dataframe and embeddings saved successfully!")
    faiss_index = save_faiss_index()
    print("FAISS index saved successfully!")
    print("Data population completed successfully!")
    time_spent = time.time() - start_time
    print(f"Time taken: {time_spent:.2f} seconds")
