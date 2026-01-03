"""
XGBoost Multi-Output Regressor for DOE Drone Design Optimization
Predicts: Range, Endurance, MTOW, Material Cost
"""

import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostDroneModel:
    """
    XGBoost model for predicting drone performance metrics

    Predicts 4 outputs:
    - Range (nm)
    - Endurance (hr)
    - MTOW (lbm)
    - Material Cost ($)
    """

    def __init__(self, **xgb_params):
        """
        Initialize XGBoost model

        Args:
            **xgb_params: XGBoost hyperparameters
        """
        # Default hyperparameters optimized for this problem
        default_params = {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,      # L1 regularization
            'reg_lambda': 1.0,     # L2 regularization
            'random_state': 42,
            'n_jobs': -1,          # Use all CPU cores
            'early_stopping_rounds': 50,
            'verbose': False
        }

        # Update with user-provided parameters
        default_params.update(xgb_params)

        # Store early stopping separately
        self.early_stopping_rounds = default_params.pop('early_stopping_rounds', 50)

        # Create base XGBoost regressor
        base_estimator = xgb.XGBRegressor(**default_params)

        # Wrap in MultiOutputRegressor for multiple targets
        self.model = MultiOutputRegressor(base_estimator)

        self.is_fitted = False
        self.feature_importances_ = None
        self.training_history_ = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> 'XGBoostDroneModel':
        """
        Train XGBoost model

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples, 4)
            X_val: Validation features (optional, for early stopping)
            y_val: Validation targets (optional, for early stopping)

        Returns:
            self
        """
        logger.info("Training XGBoost model...")
        logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

        # Train model (MultiOutputRegressor trains one model per output)
        # Note: MultiOutputRegressor doesn't support eval_set directly,
        # so we'll train without early stopping for now
        self.model.fit(X_train, y_train)

        self.is_fitted = True

        # Extract feature importances (average across output models)
        importances = []
        for estimator in self.model.estimators_:
            importances.append(estimator.feature_importances_)
        self.feature_importances_ = np.mean(importances, axis=0)

        logger.info("XGBoost training complete")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict drone performance metrics

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Predictions (n_samples, 4): [Range, Endurance, MTOW, Cost]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict(X)

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        output_names: list = None
    ) -> Dict[str, Any]:
        """
        Evaluate model performance

        Args:
            X: Input features
            y_true: True target values
            output_names: Names of output variables

        Returns:
            Dictionary with evaluation metrics
        """
        if output_names is None:
            output_names = ['Range', 'Endurance', 'MTOW', 'Cost']

        y_pred = self.predict(X)

        metrics = {
            'overall': {},
            'per_output': {}
        }

        # Overall metrics
        metrics['overall']['r2'] = float(r2_score(y_true, y_pred))
        metrics['overall']['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['overall']['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        # Per-output metrics
        for i, name in enumerate(output_names):
            metrics['per_output'][name] = {
                'r2': float(r2_score(y_true[:, i], y_pred[:, i])),
                'mae': float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
                'rmse': float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
                'mape': float(np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / (y_true[:, i] + 1e-6))) * 100)
            }

        return metrics

    def get_feature_importance(
        self,
        feature_names: list = None,
        top_n: int = None
    ) -> Dict[str, float]:
        """
        Get feature importances

        Args:
            feature_names: List of feature names
            top_n: Return only top N features (None = all)

        Returns:
            Dictionary of {feature_name: importance}
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importances_))]

        importance_dict = dict(zip(feature_names, self.feature_importances_))

        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        if top_n is not None:
            importance_dict = dict(list(importance_dict.items())[:top_n])

        return importance_dict

    def save(self, output_path: Path) -> None:
        """
        Save model to disk

        Args:
            output_path: Path to save model file (.pkl)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, output_path)
        logger.info(f"Saved XGBoost model to {output_path}")

        # Save metadata
        metadata = {
            'model_type': 'XGBoost',
            'n_outputs': 4,
            'output_names': ['Range', 'Endurance', 'MTOW', 'Cost'],
            'is_fitted': self.is_fitted
        }

        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load(self, input_path: Path) -> None:
        """
        Load model from disk

        Args:
            input_path: Path to model file (.pkl)
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Model file not found: {input_path}")

        self.model = joblib.load(input_path)
        self.is_fitted = True

        # Extract feature importances
        importances = []
        for estimator in self.model.estimators_:
            importances.append(estimator.feature_importances_)
        self.feature_importances_ = np.mean(importances, axis=0)

        logger.info(f"Loaded XGBoost model from {input_path}")


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    **xgb_params
) -> Tuple[XGBoostDroneModel, Dict[str, Any]]:
    """
    Convenience function to train and evaluate XGBoost model

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        **xgb_params: XGBoost hyperparameters

    Returns:
        Tuple of (trained_model, validation_metrics)
    """
    model = XGBoostDroneModel(**xgb_params)
    model.fit(X_train, y_train, X_val, y_val)

    # Evaluate on validation set
    if X_val is not None and y_val is not None:
        metrics = model.evaluate(X_val, y_val)
        logger.info(f"\nValidation Metrics:")
        logger.info(f"Overall R²: {metrics['overall']['r2']:.4f}")
        logger.info(f"Overall MAE: {metrics['overall']['mae']:.4f}")
        logger.info(f"\nPer-output R² scores:")
        for name, output_metrics in metrics['per_output'].items():
            logger.info(f"  {name}: {output_metrics['r2']:.4f}")
    else:
        metrics = None

    return model, metrics


if __name__ == "__main__":
    # Test XGBoost model
    from app.models.data_loader import load_doe_data
    from app.models.feature_engineering import engineer_features

    print("Loading DOE data...")
    loader, X_train, X_val, X_test, y_train, y_val, y_test = load_doe_data()

    # Engineer features
    print("\nEngineering features...")
    X_train_eng, engineer = engineer_features(loader.X_train, loader.get_feature_names(), return_engineer=True)
    X_val_eng = engineer.transform(loader.X_val)
    X_test_eng = engineer.transform(loader.X_test)

    # Train XGBoost model
    print("\nTraining XGBoost model...")
    model, val_metrics = train_xgboost_model(
        X_train_eng, loader.y_train,
        X_val_eng, loader.y_val
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = model.evaluate(X_test_eng, loader.y_test, loader.get_output_names())

    print(f"\nTest Set Results:")
    print(f"Overall R²: {test_metrics['overall']['r2']:.4f}")
    print(f"Overall MAE: {test_metrics['overall']['mae']:.4f}")
    print(f"\nPer-output metrics:")
    for name, metrics in test_metrics['per_output'].items():
        print(f"\n{name}:")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")

    # Feature importance
    print("\nTop 10 Most Important Features:")
    importance = model.get_feature_importance(engineer.get_feature_names(), top_n=10)
    for i, (feat, imp) in enumerate(importance.items(), 1):
        print(f"  {i}. {feat}: {imp:.4f}")
