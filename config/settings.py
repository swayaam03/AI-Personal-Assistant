import os
from typing import List, Tuple
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application Settings and Environment Configuration."""

    # ─── Google Gemini Provider ───
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODELS: str = os.getenv(
        "GEMINI_MODELS",
        "gemini-2.0-flash,gemini-2.0-flash-lite"
    )
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # ─── OpenRouter Provider (for Qwen, Llama, Mistral, etc.) ───
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODELS: str = os.getenv(
        "OPENROUTER_MODELS",
        "qwen/qwen3-235b-a22b:free,qwen/qwen-2.5-72b-instruct:free"
    )

    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_model_cascade(self) -> List[Tuple[str, str]]:
        """
        Returns the ordered list of (provider, model_name) tuples to attempt.
        Tries Gemini models first, then OpenRouter models.
        
        Returns:
            List of tuples: [("gemini", "gemini-2.0-flash"), ("openrouter", "qwen/qwen3-235b-a22b:free"), ...]
        """
        cascade = []

        # Add Gemini models if API key is set
        if self.GEMINI_API_KEY and self.GEMINI_API_KEY != "your_gemini_api_key_here":
            gemini_models = [m.strip() for m in self.GEMINI_MODELS.split(",") if m.strip()]
            for model in gemini_models:
                cascade.append(("gemini", model))

        # Add OpenRouter models if API key is set
        if self.OPENROUTER_API_KEY and self.OPENROUTER_API_KEY != "your_openrouter_api_key_here":
            or_models = [m.strip() for m in self.OPENROUTER_MODELS.split(",") if m.strip()]
            for model in or_models:
                cascade.append(("openrouter", model))

        # Fallback: if no keys configured, try Gemini anyway (will fail to SimulatedModel)
        if not cascade:
            cascade.append(("gemini", self.GEMINI_MODEL))

        return cascade


settings = Settings()
