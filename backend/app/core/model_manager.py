"""
Model Manager for loading and caching ML models
Provides singleton access to ensemble model and feature engineer
"""

import joblib
from pathlib import Path
import logging
from typing import Optional

from app.models.ensemble import EnsembleDroneModel
from app.models.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages loading and caching of ML models
    Provides thread-safe access to models for API endpoints
    """

    def __init__(self):
        self.ensemble_model: Optional[EnsembleDroneModel] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.is_loaded = False

    def load_models(self, models_dir: Path) -> None:
        """
        Load trained models from disk

        Args:
            models_dir: Directory containing model files
        """
        models_dir = Path(models_dir)

        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        logger.info(f"Loading models from {models_dir}")

        # Load ensemble model
        self.ensemble_model = EnsembleDroneModel()
        self.ensemble_model.load_models(
            xgb_model_path=models_dir / "xgboost_v1.pkl",
            nn_model_path=models_dir / "neural_v1.pt",
            input_dim=17  # Engineered features
        )

        # Load feature engineer
        engineer_path = models_dir / "feature_engineer.pkl"
        self.feature_engineer = joblib.load(engineer_path)

        self.is_loaded = True
        logger.info("All models loaded successfully")

    def get_ensemble_model(self) -> EnsembleDroneModel:
        """Get ensemble model"""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.ensemble_model

    def get_feature_engineer(self) -> FeatureEngineer:
        """Get feature engineer"""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.feature_engineer

    def get_model_info(self) -> dict:
        """Get information about loaded models"""
        if not self.is_loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "ensemble": {
                "xgb_weight": self.ensemble_model.xgb_weight,
                "nn_weight": self.ensemble_model.nn_weight
            },
            "feature_engineer": {
                "n_features": len(self.feature_engineer.get_feature_names()),
                "feature_names": self.feature_engineer.get_feature_names()
            }
        }
