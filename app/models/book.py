from pydantic import BaseModel, StringConstraints, Field
from typing import Annotated, List

class RecommendedBook(BaseModel):
    id: str
    title: str
    authors: List[str]
    description: str
    similarity: float

class BookRecommendationRequest(BaseModel):
    description: Annotated[str, StringConstraints(min_length=1)]
    num_recommendations: int = Field(5, ge=0)

class BookRecommendationResponse(BaseModel):
    recommendations: List[RecommendedBook]
