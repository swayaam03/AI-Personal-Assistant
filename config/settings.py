import os
from typing import List
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application Settings and Environment Configuration."""

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Comma-separated list of models to try in order of preference.
    # The agent will cascade through them if quota is exhausted on one.
    GEMINI_MODELS: str = os.getenv(
        "GEMINI_MODELS",
        "gemini-2.0-flash,gemini-2.0-flash-lite,gemini-2.5-flash,gemini-2.5-pro"
    )

    # Legacy single-model setting (used as primary if GEMINI_MODELS is not set)
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_model_cascade(self) -> List[str]:
        """Returns the ordered list of models to attempt, deduped."""
        models = [m.strip() for m in self.GEMINI_MODELS.split(",") if m.strip()]
        if not models:
            models = [self.GEMINI_MODEL]
        return models


settings = Settings()
