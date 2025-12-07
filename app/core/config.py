from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable loading"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application Settings
    app_title: str = Field(default="GPT Shopping", alias="APP_TITLE")
    environment: Literal["development", "production"] = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    
    # CORS Settings
    cors_origins: str = Field(
        default="http://localhost:3000",
        alias="CORS_ORIGINS"
    )
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Convert CORS_ORIGINS string to list"""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    # Supabase Configuration
    supabase_url: str = Field(..., alias="SUPABASE_URL")
    supabase_key: str = Field(..., alias="SUPABASE_KEY")
    supabase_service_key: str = Field(default="", alias="SUPABASE_SERVICE_KEY")
    database_url: str = Field(..., alias="DATABASE_URL")

    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    
    # Redis Configuration
    upstash_redis_url: str = Field(default="", alias="UPSTASH_REDIS_URL")
    upstash_redis_token: str = Field(default="", alias="UPSTASH_REDIS_TOKEN")
    redis_enabled: bool = Field(default=False, alias="REDIS_ENABLED")
    cache_ttl: int = Field(default=3600, alias="CACHE_TTL")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to avoid re-reading .env file on every call.
    """
    return Settings()


# Global settings instance
settings = get_settings()
