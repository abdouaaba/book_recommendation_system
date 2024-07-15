from fastapi import APIRouter, HTTPException, Depends
from app.models.book import BookRecommendationRequest, BookRecommendationResponse
from app.services.recommendation_service import RecommendationService
from app.core.logger import api_logger

router = APIRouter()

@router.post("/recommend/faiss", response_model=BookRecommendationResponse)
async def recommend_books_faiss(
    request: BookRecommendationRequest, 
    recommendation_service: RecommendationService = Depends(RecommendationService)
):
    api_logger.info(f"Received FAISS recommendation request for description: {request.description[:50]}...")
    try:
        recommendations = recommendation_service.recommend_books_faiss(request.description, k=request.num_recommendations)
        api_logger.info(f"Successfully generated {len(recommendations)} FAISS recommendations")
        return BookRecommendationResponse(recommendations=recommendations)
    except ValueError as ve:
        api_logger.error(f"ValueError in FAISS recommendation: {str(ve)}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        api_logger.error(f"Unexpected error in FAISS recommendation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

@router.post("/recommend/cosine", response_model=BookRecommendationResponse)
async def recommend_books_cosine(
    request: BookRecommendationRequest, 
    recommendation_service: RecommendationService = Depends(RecommendationService)
):
    api_logger.info(f"Received Cosine recommendation request for description: {request.description[:50]}...")
    try:
        recommendations = recommendation_service.recommend_books_cosine(request.description, k=request.num_recommendations)
        api_logger.info(f"Successfully generated {len(recommendations)} Cosine recommendations")
        return BookRecommendationResponse(recommendations=recommendations)
    except ValueError as ve:
        api_logger.error(f"ValueError in Cosine recommendation: {str(ve)}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        api_logger.error(f"Unexpected error in Cosine recommendation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")