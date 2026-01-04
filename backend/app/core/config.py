"""
Configuration management for FastAPI application
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Environment
    environment: str = "development"  # development, staging, production

    # API Configuration
    app_name: str = "DOE Drone Design Optimizer API"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # CORS Configuration
    # In production, set CORS_ORIGINS env var to your domain(s)
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # Model Configuration
    models_dir: str = "data/models"
    xgb_model_name: str = "xgboost_v1.pkl"
    nn_model_name: str = "neural_v1.pt"
    feature_engineer_name: str = "feature_engineer.pkl"

    # Optimization defaults
    default_population_size: int = 200
    default_n_generations: int = 100
    default_n_designs: int = 50
    max_optimization_time: int = 120  # seconds

    # Security
    show_error_details: bool = False  # Set to False in production

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton settings instance
settings = Settings()
