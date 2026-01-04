"""
Model Manager for loading and caching ML models
Provides singleton access to ensemble model and feature engineer
Supports both variable-span (7 inputs) and fixed-span (6 inputs) models
Supports bootstrap ensemble for uncertainty quantification
"""

import joblib
import json
from pathlib import Path
import logging
from typing import Optional, Union, List
import numpy as np

from app.models.ensemble import EnsembleDroneModel
from app.models.feature_engineering import FeatureEngineer
from app.models.xgboost_model import XGBoostDroneModel
from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages loading and caching of ML models
    Provides thread-safe access to models for API endpoints

    Supports two model types:
    - Fixed-span (12ft): 6 inputs, span is constant at 144"
    - Variable-span: 7 inputs including span

    Supports bootstrap ensemble for robust uncertainty quantification
    """

    def __init__(self):
        self.ensemble_model: Optional[EnsembleDroneModel] = None
        self.feature_engineer = None  # Can be FeatureEngineer or FixedSpanFeatureEngineer
        self.scaler_X = None  # Feature scaler for NN
        self.is_loaded = False
        self.fixed_span: Optional[float] = None  # Set if using fixed-span model
        self.bootstrap_models: Optional[List[XGBoostDroneModel]] = None  # Bootstrap ensemble

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

        # Determine model type from directory name or config
        is_fixed_span = "fixed_span" in str(models_dir) or settings.fixed_span_inches is not None

        if is_fixed_span:
            self.fixed_span = settings.fixed_span_inches or 144.0
            input_dim = 21  # 6 raw + 15 engineered (including structural features) for fixed-span
            logger.info(f"Loading FIXED-SPAN model (span={self.fixed_span} inches)")
        else:
            self.fixed_span = None
            input_dim = 22  # 7 raw + 15 engineered (including structural features) for variable-span
            logger.info("Loading VARIABLE-SPAN model")

        # Load ensemble model
        self.ensemble_model = EnsembleDroneModel()
        self.ensemble_model.load_models(
            xgb_model_path=models_dir / "xgboost_v1.pkl",
            nn_model_path=models_dir / "neural_v1.pt",
            input_dim=input_dim
        )

        # Load feature engineer
        engineer_path = models_dir / "feature_engineer.pkl"
        self.feature_engineer = joblib.load(engineer_path)
        logger.info(f"Loaded feature engineer with {len(self.feature_engineer.get_feature_names())} features")

        # Load feature scaler for NN (if exists)
        scaler_path = models_dir / "scaler_X_engineered.pkl"
        if scaler_path.exists():
            self.scaler_X = joblib.load(scaler_path)
            logger.info("Loaded feature scaler for neural network")
        else:
            self.scaler_X = None
            logger.warning("No feature scaler found - NN predictions may be affected")

        # Load bootstrap ensemble if available (for robust uncertainty)
        self._load_bootstrap_ensemble(models_dir)

        self.is_loaded = True
        logger.info("All models loaded successfully")

    def _load_bootstrap_ensemble(self, models_dir: Path) -> None:
        """Load bootstrap ensemble models if available"""
        bootstrap_metadata_path = models_dir / "bootstrap_ensemble_metadata.json"

        if bootstrap_metadata_path.exists():
            try:
                with open(bootstrap_metadata_path, 'r') as f:
                    metadata = json.load(f)

                self.bootstrap_models = []
                for model_file in metadata['model_files']:
                    model_path = models_dir / model_file
                    if model_path.exists():
                        model = XGBoostDroneModel.load(model_path)
                        self.bootstrap_models.append(model)

                logger.info(f"Loaded bootstrap ensemble with {len(self.bootstrap_models)} models")
            except Exception as e:
                logger.warning(f"Failed to load bootstrap ensemble: {e}")
                self.bootstrap_models = None
        else:
            logger.info("No bootstrap ensemble found - using single model for uncertainty")
            self.bootstrap_models = None

    def predict_with_bootstrap_uncertainty(self, X: np.ndarray) -> tuple:
        """
        Predict using bootstrap ensemble for uncertainty quantification

        Args:
            X: Engineered features (n_samples, n_features)

        Returns:
            Tuple of (mean_predictions, uncertainty_std)
        """
        if self.bootstrap_models is None or len(self.bootstrap_models) == 0:
            # Fall back to ensemble model
            return self.ensemble_model.predict(X)

        all_predictions = []
        for model in self.bootstrap_models:
            pred = model.predict(X)
            all_predictions.append(pred)

        # Stack: (n_models, n_samples, n_outputs)
        stacked = np.stack(all_predictions, axis=0)

        # Compute mean and std
        mean_pred = np.mean(stacked, axis=0)
        std_pred = np.std(stacked, axis=0)

        return mean_pred, std_pred

    def has_bootstrap_ensemble(self) -> bool:
        """Check if bootstrap ensemble is available"""
        return self.bootstrap_models is not None and len(self.bootstrap_models) > 0

    def is_fixed_span_model(self) -> bool:
        """Check if using fixed-span model"""
        return self.fixed_span is not None

    def get_fixed_span(self) -> Optional[float]:
        """Get fixed span value (or None for variable-span model)"""
        return self.fixed_span

    def get_ensemble_model(self) -> EnsembleDroneModel:
        """Get ensemble model"""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.ensemble_model

    def get_feature_engineer(self):
        """Get feature engineer"""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.feature_engineer

    def get_feature_scaler(self):
        """Get feature scaler for NN (may be None)"""
        return self.scaler_X

    def get_model_info(self) -> dict:
        """Get information about loaded models"""
        if not self.is_loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_type": "fixed_span_12ft" if self.fixed_span else "variable_span",
            "fixed_span_inches": self.fixed_span,
            "ensemble": {
                "xgb_weight": self.ensemble_model.xgb_weight,
                "nn_weight": self.ensemble_model.nn_weight
            },
            "bootstrap_ensemble": {
                "available": self.has_bootstrap_ensemble(),
                "n_models": len(self.bootstrap_models) if self.bootstrap_models else 0
            },
            "feature_engineer": {
                "n_features": len(self.feature_engineer.get_feature_names()),
                "feature_names": self.feature_engineer.get_feature_names()
            }
        }
