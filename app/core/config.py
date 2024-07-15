from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Book Recommendation System"
    PROJECT_VERSION: str = "1.0.0"
    V1_STR: str = "/api/v1"
    OPENAI_API_KEY: str
    GOOGLE_BOOKS_API_KEY: str
    DATA_DIR: str = "data"

    class Config:
        env_file = ".env"

settings = Settings()