"""
Specialized Training Script for Wingtip Deflection Prediction
Uses log-transform and sample weighting to handle extreme data skew
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import logging
import json
import joblib
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeflectionSpecialistModel:
    """
    Specialized XGBoost model for wingtip deflection prediction.

    Key techniques:
    1. Log-transform target: Handles extreme skew (0.01 to 10000+ range)
    2. Sample weighting: Upweight high-deflection cases for balanced learning
    3. Structural features: Add physics-informed features
    """

    # Epsilon for log transform (avoids log(0))
    LOG_EPSILON = 1e-4

    # Deflection threshold for weighting (inches)
    HIGH_DEFLECTION_THRESHOLD = 0.5

    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.feature_names = None

    def _log_transform(self, y: np.ndarray) -> np.ndarray:
        """Transform deflection to log scale"""
        return np.log(y + self.LOG_EPSILON)

    def _inverse_log_transform(self, y_log: np.ndarray) -> np.ndarray:
        """Transform back from log scale"""
        return np.exp(y_log) - self.LOG_EPSILON

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Compute sample weights to upweight rare high-deflection cases.

        Strategy: Weight inversely proportional to bin frequency
        """
        # Create bins based on deflection values
        bins = [0, 0.1, 0.5, 1.0, 5.0, 100.0, float('inf')]
        bin_indices = np.digitize(y, bins)

        # Count samples per bin
        bin_counts = np.bincount(bin_indices, minlength=len(bins)+1)

        # Weight inversely proportional to count (with smoothing)
        weights = np.zeros(len(y))
        for i in range(len(y)):
            bin_idx = bin_indices[i]
            # Use sqrt to moderate the weighting
            weights[i] = np.sqrt(len(y) / (bin_counts[bin_idx] + 1))

        # Normalize to mean=1
        weights = weights / weights.mean()

        return weights

    def add_structural_features(self, X: np.ndarray, feature_names: list) -> tuple:
        """
        Add physics-informed structural features for deflection.

        Key insight: Deflection scales with:
        - Wing loading (weight / area)
        - Span cubed (bending moment)
        - Sweep angle (changes lift distribution)
        """
        features = [X]
        new_names = list(feature_names)

        # Find indices of key features
        try:
            loa_idx = feature_names.index('loa')
            le_sweep_p1_idx = feature_names.index('le_sweep_p1')
            le_sweep_p2_idx = feature_names.index('le_sweep_p2')
            te_sweep_p1_idx = feature_names.index('te_sweep_p1')
            te_sweep_p2_idx = feature_names.index('te_sweep_p2')
            panel_break_idx = feature_names.index('panel_break')
        except ValueError:
            logger.warning("Some expected features not found, using available features only")
            return X, feature_names

        loa = X[:, loa_idx]
        le_sweep_p1 = X[:, le_sweep_p1_idx]
        le_sweep_p2 = X[:, le_sweep_p2_idx]
        te_sweep_p1 = X[:, te_sweep_p1_idx]
        te_sweep_p2 = X[:, te_sweep_p2_idx]
        panel_break = X[:, panel_break_idx]

        # Fixed span for 12ft model
        span = 144.0

        # 1. Sweep differential magnitude (larger = more complex loading)
        sweep_diff_total = np.abs(te_sweep_p1 - le_sweep_p1) + np.abs(te_sweep_p2 - le_sweep_p2)
        features.append(sweep_diff_total.reshape(-1, 1))
        new_names.append('sweep_diff_total')

        # 2. Max absolute sweep (extreme sweeps cause structural issues)
        max_sweep = np.maximum.reduce([
            np.abs(le_sweep_p1), np.abs(le_sweep_p2),
            np.abs(te_sweep_p1), np.abs(te_sweep_p2)
        ])
        features.append(max_sweep.reshape(-1, 1))
        new_names.append('max_abs_sweep')

        # 3. TE sweep severity (negative TE sweep = forward sweep = flutter prone)
        te_sweep_severity = np.minimum(te_sweep_p1, te_sweep_p2)
        features.append(te_sweep_severity.reshape(-1, 1))
        new_names.append('te_sweep_severity')

        # 4. Aspect ratio effect (higher AR = more deflection)
        aspect_ratio_proxy = span / (loa + 1e-6)
        features.append(aspect_ratio_proxy.reshape(-1, 1))
        new_names.append('ar_proxy')

        # 5. Panel break × sweep interaction (complex loading at break)
        panel_sweep_interaction = panel_break * (np.abs(le_sweep_p1 - le_sweep_p2))
        features.append(panel_sweep_interaction.reshape(-1, 1))
        new_names.append('panel_sweep_interaction')

        # 6. LOA inverse (smaller aircraft = less stiff)
        loa_inverse = 1000.0 / (loa + 1e-6)
        features.append(loa_inverse.reshape(-1, 1))
        new_names.append('loa_inverse')

        # 7. Sweep angle product (interaction effect)
        sweep_product = le_sweep_p1 * le_sweep_p2
        features.append(sweep_product.reshape(-1, 1))
        new_names.append('sweep_product')

        # 8. TE sweep product (negative product = opposite sweeps)
        te_sweep_product = te_sweep_p1 * te_sweep_p2
        features.append(te_sweep_product.reshape(-1, 1))
        new_names.append('te_sweep_product')

        X_enhanced = np.hstack(features)

        return X_enhanced, new_names

    def fit(
        self,
        X: np.ndarray,
        y_deflection: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: list = None,
        **xgb_params
    ):
        """
        Train deflection specialist model.

        Args:
            X: Training features (n_samples, n_features)
            y_deflection: Deflection values (n_samples,) - single output
            X_val: Validation features
            y_val: Validation deflection
            feature_names: Feature names
            **xgb_params: XGBoost parameters
        """
        logger.info("Training Deflection Specialist Model...")

        # Add structural features
        X_enhanced, self.feature_names = self.add_structural_features(X, feature_names)
        logger.info(f"Enhanced features: {X_enhanced.shape[1]} (added {X_enhanced.shape[1] - X.shape[1]} structural)")

        # Log-transform target
        y_log = self._log_transform(y_deflection)
        logger.info(f"Log-transformed deflection: [{y_log.min():.2f}, {y_log.max():.2f}]")

        # Compute sample weights
        weights = self._compute_sample_weights(y_deflection)
        logger.info(f"Sample weights: min={weights.min():.2f}, max={weights.max():.2f}, mean={weights.mean():.2f}")

        # Default XGBoost parameters optimized for deflection
        default_params = {
            'n_estimators': 1000,
            'max_depth': 12,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'min_child_weight': 5,
            'gamma': 0.1,
            'random_state': 42,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'verbosity': 0
        }
        default_params.update(xgb_params)

        # Create and train model
        self.model = xgb.XGBRegressor(**default_params)

        # Prepare validation if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_enhanced, _ = self.add_structural_features(X_val, feature_names)
            y_val_log = self._log_transform(y_val)
            eval_set = [(X_val_enhanced, y_val_log)]

        # Fit with sample weights and early stopping
        self.model.fit(
            X_enhanced, y_log,
            sample_weight=weights,
            eval_set=eval_set,
            verbose=False
        )

        self.is_fitted = True

        # Evaluate on training set
        y_pred_train = self.predict(X, feature_names)
        train_r2 = r2_score(y_deflection, y_pred_train)
        train_mae = mean_absolute_error(y_deflection, y_pred_train)
        logger.info(f"Training R²: {train_r2:.4f}, MAE: {train_mae:.4f}")

        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            y_pred_val = self.predict(X_val, feature_names)
            val_r2 = r2_score(y_val, y_pred_val)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            logger.info(f"Validation R²: {val_r2:.4f}, MAE: {val_mae:.4f}")

            # Analyze performance by deflection range
            self._analyze_by_range(y_val, y_pred_val)

        return self

    def predict(self, X: np.ndarray, feature_names: list = None) -> np.ndarray:
        """Predict deflection values"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if feature_names is None:
            feature_names = self.feature_names[:X.shape[1]]

        X_enhanced, _ = self.add_structural_features(X, feature_names)

        y_log_pred = self.model.predict(X_enhanced)
        y_pred = self._inverse_log_transform(y_log_pred)

        # Clamp negative predictions to 0
        y_pred = np.maximum(y_pred, 0)

        return y_pred

    def _analyze_by_range(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Analyze prediction performance by deflection range"""
        logger.info("\nPerformance by deflection range:")

        ranges = [
            (0, 0.1, "Low (0-0.1 in)"),
            (0.1, 1.0, "Medium (0.1-1 in)"),
            (1.0, 10.0, "High (1-10 in)"),
            (10.0, float('inf'), "Very High (>10 in)")
        ]

        for low, high, label in ranges:
            mask = (y_true >= low) & (y_true < high)
            if mask.sum() > 0:
                r2 = r2_score(y_true[mask], y_pred[mask]) if mask.sum() > 1 else 0
                mae = mean_absolute_error(y_true[mask], y_pred[mask])
                logger.info(f"  {label}: n={mask.sum()}, R²={r2:.4f}, MAE={mae:.4f}")

    def get_feature_importance(self, top_n: int = 15) -> dict:
        """Get feature importance"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        importance = self.model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1][:top_n]

        return {
            self.feature_names[i]: float(importance[i])
            for i in sorted_idx
        }

    def save(self, path: Path):
        """Save model to disk"""
        path = Path(path)

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'log_epsilon': self.LOG_EPSILON,
            'is_fitted': self.is_fitted
        }

        joblib.dump(model_data, path)
        logger.info(f"Saved deflection model to {path}")

    def load(self, path: Path):
        """Load model from disk"""
        path = Path(path)
        model_data = joblib.load(path)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.LOG_EPSILON = model_data['log_epsilon']
        self.is_fitted = model_data['is_fitted']

        logger.info(f"Loaded deflection model from {path}")


def main(data_path: str = None):
    """Train specialized deflection model"""

    if data_path is None:
        data_path = r"D:\nTop\DOERunner\gcp_10000sample_fixed_span_12ft\doe_summary.csv"

    logger.info("="*80)
    logger.info("DEFLECTION SPECIALIST MODEL TRAINING")
    logger.info("="*80)

    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")

    # Remove extreme outliers (> 100 inches is physically impossible)
    MAX_DEFLECTION = 100.0
    n_outliers = (df['wingtip_deflection_in'] > MAX_DEFLECTION).sum()
    df = df[df['wingtip_deflection_in'] <= MAX_DEFLECTION].copy()
    logger.info(f"Removed {n_outliers} extreme outliers (>{MAX_DEFLECTION} in)")

    # Also remove zero/negative deflection (simulation failures)
    n_zero = (df['wingtip_deflection_in'] <= 0).sum()
    df = df[df['wingtip_deflection_in'] > 0].copy()
    logger.info(f"Removed {n_zero} zero/negative values")

    logger.info(f"Final dataset: {len(df)} samples")

    # Define features (6 inputs for fixed-span)
    input_features = ['loa', 'le_sweep_p1', 'le_sweep_p2', 'te_sweep_p1', 'te_sweep_p2', 'panel_break']

    X = df[input_features].values
    y = df['wingtip_deflection_in'].values

    # Train/val/test split
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train model
    model = DeflectionSpecialistModel()
    model.fit(
        X_train, y_train,
        X_val, y_val,
        feature_names=input_features
    )

    # Evaluate on test set
    logger.info("\n" + "="*60)
    logger.info("TEST SET EVALUATION")
    logger.info("="*60)

    y_pred_test = model.predict(X_test, input_features)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    logger.info(f"Test R²: {test_r2:.4f}")
    logger.info(f"Test MAE: {test_mae:.4f}")

    model._analyze_by_range(y_test, y_pred_test)

    # Feature importance
    logger.info("\nTop Feature Importances:")
    importance = model.get_feature_importance(15)
    for i, (feat, imp) in enumerate(importance.items(), 1):
        logger.info(f"  {i}. {feat}: {imp:.4f}")

    # Save model
    models_dir = Path(__file__).parent.parent / "data" / "models_fixed_span_12ft"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "deflection_specialist.pkl"
    model.save(model_path)

    # Save results
    results = {
        'training_date': datetime.now().isoformat(),
        'model_type': 'DeflectionSpecialist',
        'techniques': ['log_transform', 'sample_weighting', 'structural_features'],
        'dataset': {
            'source': str(data_path),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'outliers_removed': int(n_outliers),
            'zeros_removed': int(n_zero)
        },
        'metrics': {
            'test_r2': float(test_r2),
            'test_mae': float(test_mae)
        },
        'feature_importance': importance
    }

    results_path = models_dir / "deflection_training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSaved results to {results_path}")
    logger.info("="*80)

    return results


if __name__ == "__main__":
    main()
