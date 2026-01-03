"""
Ensemble Model: Combines XGBoost + Neural Network with Uncertainty Quantification
Provides robust predictions with confidence estimates
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import logging
import json

from app.models.xgboost_model import XGBoostDroneModel
from app.models.neural_model import NeuralNetworkDroneModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleDroneModel:
    """
    Ensemble model combining XGBoost and Neural Network

    Strategy: Weighted average
    - XGBoost weight: 0.6 (fast, reliable)
    - Neural Network weight: 0.4 (captures non-linearities)

    Uncertainty quantification: Standard deviation between model predictions
    """

    def __init__(
        self,
        xgb_weight: float = 0.6,
        nn_weight: float = 0.4
    ):
        """
        Initialize ensemble model

        Args:
            xgb_weight: Weight for XGBoost predictions (default: 0.6)
            nn_weight: Weight for Neural Network predictions (default: 0.4)
        """
        if abs(xgb_weight + nn_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {xgb_weight + nn_weight}")

        self.xgb_weight = xgb_weight
        self.nn_weight = nn_weight

        self.xgb_model = None
        self.nn_model = None
        self.is_fitted = False

    def set_models(
        self,
        xgb_model: XGBoostDroneModel,
        nn_model: NeuralNetworkDroneModel
    ) -> 'EnsembleDroneModel':
        """
        Set trained models

        Args:
            xgb_model: Trained XGBoost model
            nn_model: Trained Neural Network model

        Returns:
            self
        """
        if not xgb_model.is_fitted:
            raise ValueError("XGBoost model must be fitted")
        if not nn_model.is_fitted:
            raise ValueError("Neural Network model must be fitted")

        self.xgb_model = xgb_model
        self.nn_model = nn_model
        self.is_fitted = True

        logger.info(f"Ensemble configured: XGBoost={self.xgb_weight:.1%}, "
                   f"Neural Net={self.nn_weight:.1%}")

        return self

    def predict(
        self,
        X: np.ndarray,
        return_uncertainty: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with ensemble model and uncertainty quantification

        Args:
            X: Input features (n_samples, n_features)
            return_uncertainty: If True, return uncertainty estimates

        Returns:
            Tuple of (predictions, uncertainty)
            - predictions: (n_samples, 4) weighted average
            - uncertainty: (n_samples, 4) standard deviation between models
        """
        if not self.is_fitted:
            raise ValueError("Models must be set before prediction")

        # Get predictions from both models
        y_pred_xgb = self.xgb_model.predict(X)
        y_pred_nn = self.nn_model.predict(X)

        # Weighted average
        y_pred_ensemble = (
            self.xgb_weight * y_pred_xgb +
            self.nn_weight * y_pred_nn
        )

        if return_uncertainty:
            # Uncertainty: standard deviation between model predictions
            # Stack predictions along new axis
            predictions_stack = np.stack([y_pred_xgb, y_pred_nn], axis=0)
            uncertainty = np.std(predictions_stack, axis=0)

            return y_pred_ensemble, uncertainty
        else:
            return y_pred_ensemble, None

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        output_names: list = None
    ) -> Dict[str, Any]:
        """
        Evaluate ensemble model performance

        Args:
            X: Input features
            y_true: True target values
            output_names: Names of output variables

        Returns:
            Dictionary with evaluation metrics for ensemble and individual models
        """
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        if output_names is None:
            output_names = ['Range', 'Endurance', 'MTOW', 'Cost']

        # Get predictions
        y_pred_ensemble, uncertainty = self.predict(X, return_uncertainty=True)
        y_pred_xgb = self.xgb_model.predict(X)
        y_pred_nn = self.nn_model.predict(X)

        metrics = {
            'ensemble': {'overall': {}, 'per_output': {}},
            'xgboost': {'overall': {}, 'per_output': {}},
            'neural_network': {'overall': {}, 'per_output': {}},
            'uncertainty': {}
        }

        # Evaluate ensemble
        for model_name, y_pred in [
            ('ensemble', y_pred_ensemble),
            ('xgboost', y_pred_xgb),
            ('neural_network', y_pred_nn)
        ]:
            # Overall metrics
            metrics[model_name]['overall']['r2'] = float(r2_score(y_true, y_pred))
            metrics[model_name]['overall']['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics[model_name]['overall']['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))

            # Per-output metrics
            for i, name in enumerate(output_names):
                metrics[model_name]['per_output'][name] = {
                    'r2': float(r2_score(y_true[:, i], y_pred[:, i])),
                    'mae': float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
                    'rmse': float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
                    'mape': float(np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / (y_true[:, i] + 1e-6))) * 100)
                }

        # Uncertainty statistics
        for i, name in enumerate(output_names):
            metrics['uncertainty'][name] = {
                'mean': float(np.mean(uncertainty[:, i])),
                'median': float(np.median(uncertainty[:, i])),
                'std': float(np.std(uncertainty[:, i])),
                'max': float(np.max(uncertainty[:, i]))
            }

        return metrics

    def save(self, output_dir: Path) -> None:
        """
        Save ensemble configuration

        Args:
            output_dir: Directory to save ensemble metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            'model_type': 'Ensemble (XGBoost + Neural Network)',
            'xgb_weight': self.xgb_weight,
            'nn_weight': self.nn_weight,
            'n_outputs': 4,
            'output_names': ['Range', 'Endurance', 'MTOW', 'Cost'],
            'is_fitted': self.is_fitted
        }

        metadata_path = output_dir / "ensemble_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved ensemble metadata to {output_dir}")

    def load_models(
        self,
        xgb_model_path: Path,
        nn_model_path: Path,
        input_dim: int = 17
    ) -> None:
        """
        Load trained models from disk

        Args:
            xgb_model_path: Path to XGBoost model file
            nn_model_path: Path to Neural Network model file
            input_dim: Number of input features (for Neural Network)
        """
        # Load XGBoost model
        self.xgb_model = XGBoostDroneModel()
        self.xgb_model.load(xgb_model_path)

        # Load Neural Network model
        self.nn_model = NeuralNetworkDroneModel(input_dim=input_dim)
        self.nn_model.load(nn_model_path)

        self.is_fitted = True

        logger.info(f"Loaded ensemble models from disk")


def create_ensemble_model(
    xgb_model: XGBoostDroneModel,
    nn_model: NeuralNetworkDroneModel,
    xgb_weight: float = 0.6,
    nn_weight: float = 0.4
) -> EnsembleDroneModel:
    """
    Convenience function to create and configure ensemble model

    Args:
        xgb_model: Trained XGBoost model
        nn_model: Trained Neural Network model
        xgb_weight: Weight for XGBoost (default: 0.6)
        nn_weight: Weight for Neural Network (default: 0.4)

    Returns:
        Configured ensemble model
    """
    ensemble = EnsembleDroneModel(xgb_weight=xgb_weight, nn_weight=nn_weight)
    ensemble.set_models(xgb_model, nn_model)

    return ensemble


def optimize_ensemble_weights(
    xgb_model: XGBoostDroneModel,
    nn_model: NeuralNetworkDroneModel,
    X_val: np.ndarray,
    y_val: np.ndarray,
    weights_to_try: list = None
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Find optimal ensemble weights using validation set

    Args:
        xgb_model: Trained XGBoost model
        nn_model: Trained Neural Network model
        X_val: Validation features
        y_val: Validation targets
        weights_to_try: List of (xgb_weight, nn_weight) tuples to try

    Returns:
        Tuple of (best_xgb_weight, best_nn_weight, best_metrics)
    """
    from sklearn.metrics import r2_score

    if weights_to_try is None:
        # Default weights to try
        weights_to_try = [
            (1.0, 0.0),   # Pure XGBoost
            (0.9, 0.1),
            (0.8, 0.2),
            (0.7, 0.3),
            (0.6, 0.4),
            (0.5, 0.5),
            (0.4, 0.6),
            (0.3, 0.7),
            (0.2, 0.8),
            (0.1, 0.9),
            (0.0, 1.0)    # Pure Neural Network
        ]

    logger.info("Optimizing ensemble weights...")

    # Get predictions from both models
    y_pred_xgb = xgb_model.predict(X_val)
    y_pred_nn = nn_model.predict(X_val)

    best_r2 = -np.inf
    best_weights = None
    best_metrics = None

    results = []

    for xgb_w, nn_w in weights_to_try:
        # Weighted ensemble prediction
        y_pred = xgb_w * y_pred_xgb + nn_w * y_pred_nn

        # Overall R² score
        r2 = r2_score(y_val, y_pred)

        results.append({
            'xgb_weight': xgb_w,
            'nn_weight': nn_w,
            'r2_score': r2
        })

        logger.info(f"Weights ({xgb_w:.1f}, {nn_w:.1f}): R² = {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_weights = (xgb_w, nn_w)
            best_metrics = {'r2': r2}

    logger.info(f"\nBest weights: XGBoost={best_weights[0]:.1%}, "
               f"Neural Net={best_weights[1]:.1%}")
    logger.info(f"Best validation R²: {best_r2:.4f}")

    return best_weights[0], best_weights[1], best_metrics


if __name__ == "__main__":
    # Test ensemble model
    from app.models.data_loader import load_doe_data
    from app.models.feature_engineering import engineer_features
    from app.models.xgboost_model import train_xgboost_model
    from app.models.neural_model import train_neural_network_model

    print("Loading DOE data...")
    loader, _, _, _, _, _, _ = load_doe_data()

    # Engineer features
    print("\nEngineering features...")
    X_train_eng, engineer = engineer_features(loader.X_train, loader.get_feature_names(), return_engineer=True)
    X_val_eng = engineer.transform(loader.X_val)
    X_test_eng = engineer.transform(loader.X_test)

    # Train XGBoost
    print("\nTraining XGBoost...")
    xgb_model, _ = train_xgboost_model(X_train_eng, loader.y_train, X_val_eng, loader.y_val)

    # Scale for Neural Network
    X_train_scaled, y_train_scaled = loader.transform(X_train_eng, loader.y_train)
    X_val_scaled, y_val_scaled = loader.transform(X_val_eng, loader.y_val)
    X_test_scaled, y_test_scaled = loader.transform(X_test_eng, loader.y_test)

    # Train Neural Network
    print("\nTraining Neural Network...")
    nn_model, _ = train_neural_network_model(
        X_train_scaled, y_train_scaled,
        X_val_scaled, y_val_scaled,
        input_dim=X_train_scaled.shape[1],
        epochs=50  # Reduced for testing
    )

    # Optimize ensemble weights
    print("\nOptimizing ensemble weights...")
    best_xgb_w, best_nn_w, _ = optimize_ensemble_weights(
        xgb_model, nn_model,
        X_val_eng, loader.y_val
    )

    # Create ensemble with optimized weights
    print("\nCreating ensemble model...")
    ensemble = create_ensemble_model(xgb_model, nn_model, best_xgb_w, best_nn_w)

    # Evaluate ensemble on test set
    print("\nEvaluating ensemble on test set...")
    test_metrics = ensemble.evaluate(X_test_eng, loader.y_test, loader.get_output_names())

    print(f"\n{'='*60}")
    print("ENSEMBLE MODEL RESULTS")
    print(f"{'='*60}")

    for model_name in ['ensemble', 'xgboost', 'neural_network']:
        print(f"\n{model_name.upper()}:")
        print(f"  Overall R²: {test_metrics[model_name]['overall']['r2']:.4f}")
        print(f"  Overall MAE: {test_metrics[model_name]['overall']['mae']:.4f}")

    print(f"\nPer-output R² scores:")
    print(f"{'Output':<15} {'Ensemble':<12} {'XGBoost':<12} {'Neural Net':<12}")
    print("-" * 60)
    for name in loader.get_output_names():
        ensemble_r2 = test_metrics['ensemble']['per_output'][name]['r2']
        xgb_r2 = test_metrics['xgboost']['per_output'][name]['r2']
        nn_r2 = test_metrics['neural_network']['per_output'][name]['r2']
        print(f"{name:<15} {ensemble_r2:<12.4f} {xgb_r2:<12.4f} {nn_r2:<12.4f}")

    print(f"\nUncertainty Statistics:")
    for name in loader.get_output_names():
        unc = test_metrics['uncertainty'][name]
        print(f"\n{name}:")
        print(f"  Mean: {unc['mean']:.4f}")
        print(f"  Median: {unc['median']:.4f}")
        print(f"  Max: {unc['max']:.4f}")
