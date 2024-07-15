import pytest
from app.services.recommendation_service import RecommendationService
from app.models import RecommendedBook
from pathlib import Path

@pytest.fixture
def recommendation_service():
    return RecommendationService()

# Test recommend_books_faiss method
@pytest.mark.parametrize("k", [5, 0, -1])
def test_recommend_books_faiss(recommendation_service: RecommendationService, k: int):
    description = "A science fiction novel"
    if k == 0:
        assert recommendation_service.recommend_books_faiss(description, k) == []
    elif k < 0:
        with pytest.raises(ValueError):
            recommendation_service.recommend_books_faiss(description, k)
    else:
        recommendations = recommendation_service.recommend_books_faiss(description, k)
        assert len(recommendations) == k
        for recommendation in recommendations:
            assert isinstance(recommendation, RecommendedBook)
            assert recommendation.similarity > 0

# Test recommend_books_cosine method
@pytest.mark.parametrize("k", [5, 0, -1])
def test_recommend_books_cosine(recommendation_service: RecommendationService, k: int):
    description = "A science fiction novel"
    if k == 0:
        assert recommendation_service.recommend_books_cosine(description, k) == []
    elif k < 0:
        with pytest.raises(ValueError):
            recommendation_service.recommend_books_cosine(description, k)
    else:
        recommendations = recommendation_service.recommend_books_cosine(description, k)
        assert len(recommendations) == k
        for recommendation in recommendations:
            assert isinstance(recommendation, RecommendedBook)
            assert recommendation.similarity > 0