import os
import numpy as np
from pathlib import Path
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from app.services.book_service import BookService
from app.services.processing_chain import ProcessingChain
from app.core.config import settings
from app.core.logger import service_logger
from app.models import RecommendedBook

class RecommendationService:
    def __init__(self):
        self.book_service = BookService()
        self.processing_chain = ProcessingChain()
        self.data_dir = Path(settings.DATA_DIR)
        self.df = self.book_service.load_dataframe()
        self.faiss_index = self._load_or_create_faiss_index()
        self.embeddings = self._load_or_create_embeddings()

    def _load_or_create_faiss_index(self):
        faiss_index_path = os.path.join(settings.DATA_DIR, "faiss_index")
        try:
            if os.path.exists(faiss_index_path):
                service_logger.info("Loading existing FAISS index")
                return FAISS.load_local(faiss_index_path, self.processing_chain.embeddings, allow_dangerous_deserialization=True)
            else:
                service_logger.info("Creating new FAISS index")
                faiss_index = self.processing_chain.create_faiss_index(self.df)
                faiss_index.save_local(faiss_index_path)
                return faiss_index
        except Exception as e:
            service_logger.error(f"Error in _load_or_create_faiss_index: {str(e)}", exc_info=True)
            raise

    def _load_or_create_embeddings(self):
        embeddings_path = os.path.join(settings.DATA_DIR, "embeddings.npy")
        try:
            if os.path.exists(embeddings_path):
                service_logger.info("Loading existing embeddings")
                return np.load(embeddings_path)
            else:
                service_logger.info("Creating new embeddings")
                texts = self.df['processed_description'].tolist()
                embeddings = self.processing_chain.create_embeddings(texts)
                np.save(embeddings_path, embeddings)
                return embeddings
        except Exception as e:
            service_logger.error(f"Error in _load_or_create_embeddings: {str(e)}", exc_info=True)
            raise

    @lru_cache(maxsize=100)
    def recommend_books_faiss(self, description, k=5):
        if k == 0:
            return []
        if k <= 0:
            raise ValueError("Number of recommendations must be greater than 0")
        try:
            # processed_query = self.processing_chain.process_query(description)
            clean_description = self.book_service._clean_text(description)
            similar_docs = self.faiss_index.similarity_search_with_relevance_scores(clean_description, k=k)

            return [
                RecommendedBook(
                    id=doc.metadata['id'], 
                    title=doc.metadata['title'], 
                    authors=doc.metadata['authors'], 
                    description=doc.metadata['description'],
                    similarity=score
                    ) for doc, score in similar_docs
                ]
        
        except Exception as e:
            service_logger.error(f"Error in recommend_books_faiss: {str(e)}", exc_info=True)
            raise

    @lru_cache(maxsize=100)
    def recommend_books_cosine(self, description, k=5):
        if k == 0:
            return []
        if k <= 0:
            raise ValueError("Number of recommendations must be greater than 0")
        try:
            # processed_query = self.processing_chain.process_query(description)
            query_embedding = self.processing_chain.embeddings.embed_query(description)
            cosine_similarities = cosine_similarity([query_embedding], self.embeddings).flatten()
            similar_indices = cosine_similarities.argsort()[-k:][::-1]

            return [
                RecommendedBook(
                    id=self.df.iloc[idx]['id'],
                    title=self.df.iloc[idx]['title'],
                    authors=self.df.iloc[idx]['authors'],
                    description=self.df.iloc[idx]['description'],
                    similarity=cosine_similarities[idx]
                ) for idx in similar_indices
            ]
        except Exception as e:
            service_logger.error(f"Error in recommend_books_cosine: {str(e)}", exc_info=True)
            raise