"""
Training Script for Fixed-Span (12ft) DOE Drone Optimizer Models
Trains XGBoost, Neural Network, and creates Ensemble for fixed 144" span designs
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime

from app.models.data_loader import DOEDataLoader
from app.models.xgboost_model import train_xgboost_model
from app.models.neural_model import train_neural_network_model
from app.models.ensemble import create_ensemble_model, optimize_ensemble_weights

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FixedSpanFeatureEngineer:
    """
    Feature engineering for fixed-span designs (6 inputs, no span).

    Raw features (6):
    - LOA (Length Overall)
    - LE Sweep P1/P2 (Leading Edge sweeps)
    - TE Sweep P1/P2 (Trailing Edge sweeps)
    - Panel Break (span fraction)

    Fixed value:
    - Span = 144 inches (12 ft)
    """

    FIXED_SPAN = 144.0  # inches

    def __init__(self):
        self.feature_names = []
        self.raw_feature_names = []

    def fit(self, X: np.ndarray, feature_names: list = None) -> 'FixedSpanFeatureEngineer':
        """Fit feature engineer (compute feature names)"""
        if feature_names is None:
            feature_names = [
                'loa', 'le_sweep_p1', 'le_sweep_p2',
                'te_sweep_p1', 'te_sweep_p2', 'panel_break'
            ]

        self.raw_feature_names = feature_names
        self.feature_names = self._get_all_feature_names()

        logger.info(f"Fixed-span feature engineering: {len(self.raw_feature_names)} raw → "
                   f"{len(self.feature_names)} engineered features")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform raw features into engineered features

        Args:
            X: Raw input features (n_samples, 6)
               Columns: [LOA, LE_Sweep_P1, LE_Sweep_P2,
                        TE_Sweep_P1, TE_Sweep_P2, Panel_Break]

        Returns:
            Engineered features (n_samples, 15)
        """
        # Extract raw features (span is fixed, not in input)
        loa = X[:, 0]          # Length Overall (inches)
        le_sweep_p1 = X[:, 1]  # LE sweep panel 1 (degrees)
        le_sweep_p2 = X[:, 2]  # LE sweep panel 2 (degrees)
        te_sweep_p1 = X[:, 3]  # TE sweep panel 1 (degrees)
        te_sweep_p2 = X[:, 4]  # TE sweep panel 2 (degrees)
        panel_break = X[:, 5]  # Panel break location (fraction 0-1)

        # Fixed span
        span = np.full_like(loa, self.FIXED_SPAN)

        features = []

        # 1. Raw features (6) - include them
        features.extend([
            loa, le_sweep_p1, le_sweep_p2,
            te_sweep_p1, te_sweep_p2, panel_break
        ])

        # 2. Aspect ratio proxy (span^2 / wing_area_estimate)
        wing_area_proxy = loa * span
        aspect_ratio_proxy = span ** 2 / (wing_area_proxy + 1e-6)
        features.append(aspect_ratio_proxy)

        # 3. Sweep differentials (indicate wing taper)
        sweep_diff_p1 = te_sweep_p1 - le_sweep_p1
        sweep_diff_p2 = te_sweep_p2 - le_sweep_p2
        features.extend([sweep_diff_p1, sweep_diff_p2])

        # 4. Average sweeps
        avg_le_sweep = (le_sweep_p1 + le_sweep_p2) / 2.0
        avg_te_sweep = (te_sweep_p1 + te_sweep_p2) / 2.0
        features.extend([avg_le_sweep, avg_te_sweep])

        # 5. Sweep asymmetry (difference between panels)
        sweep_asymmetry = np.abs(le_sweep_p1 - le_sweep_p2)
        features.append(sweep_asymmetry)

        # 6. Wing loading proxy (inverse of wing area)
        wing_loading_proxy = 1000.0 / (wing_area_proxy + 1e-6)
        features.append(wing_loading_proxy)

        # 7. Planform complexity (sum of absolute sweep differentials)
        planform_complexity = np.abs(sweep_diff_p1) + np.abs(sweep_diff_p2)
        features.append(planform_complexity)

        # 8. Span to LOA ratio (wing slenderness) - constant since span is fixed
        span_loa_ratio = span / (loa + 1e-6)
        features.append(span_loa_ratio)

        # 9. Panel break interaction with sweep
        panel_break_sweep_interaction = panel_break * avg_le_sweep
        features.append(panel_break_sweep_interaction)

        # Stack all features
        X_engineered = np.column_stack(features)

        return X_engineered

    def fit_transform(self, X: np.ndarray, feature_names: list = None) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(X, feature_names)
        return self.transform(X)

    def get_feature_names(self) -> list:
        """Get list of all engineered feature names"""
        return self.feature_names.copy()

    def _get_all_feature_names(self) -> list:
        """Generate list of all feature names"""
        names = self.raw_feature_names.copy()

        # Add derived feature names
        names.extend([
            'Aspect_Ratio_Proxy',
            'Sweep_Diff_P1',
            'Sweep_Diff_P2',
            'Avg_LE_Sweep',
            'Avg_TE_Sweep',
            'Sweep_Asymmetry',
            'Wing_Loading_Proxy',
            'Planform_Complexity',
            'Span_LOA_Ratio',
            'Panel_Break_Sweep_Interaction'
        ])

        return names


class FixedSpanDataLoader(DOEDataLoader):
    """
    Data loader for fixed-span (12ft / 144") dataset.
    Excludes 'span' from input features since it's constant.
    """

    # 6 inputs instead of 7 (no span)
    INPUT_FEATURES = [
        'loa',              # Length Overall (inches)
        'le_sweep_p1',      # Leading Edge Sweep Panel 1 (degrees)
        'le_sweep_p2',      # Leading Edge Sweep Panel 2 (degrees)
        'te_sweep_p1',      # Trailing Edge Sweep Panel 1 (degrees)
        'te_sweep_p2',      # Trailing Edge Sweep Panel 2 (degrees)
        'panel_break'       # Panel break location (fraction)
    ]

    def __init__(self, data_path: str):
        """
        Initialize fixed-span data loader

        Args:
            data_path: Path to fixed-span doe_summary.csv
        """
        super().__init__(data_path)
        self.fixed_span = None  # Will be set after loading data

    def load_data(self) -> pd.DataFrame:
        """Load and validate fixed-span data"""
        df = super().load_data()

        # Verify span is constant
        span_values = df['span'].unique()
        if len(span_values) != 1:
            logger.warning(f"Expected fixed span, but found {len(span_values)} unique values: {span_values[:5]}...")
        else:
            self.fixed_span = span_values[0]
            logger.info(f"Fixed span confirmed: {self.fixed_span} inches ({self.fixed_span/12:.1f} ft)")

        return df


def main(data_path: str = None):
    """
    Main training pipeline for fixed-span model

    Args:
        data_path: Path to fixed-span dataset CSV
    """

    if data_path is None:
        # Default to the new 10K fixed-span dataset
        data_path = r"D:\nTop\DOERunner\gcp_10000sample_fixed_span_12ft\doe_summary.csv"

    logger.info("="*80)
    logger.info("FIXED-SPAN (12ft) DOE DRONE OPTIMIZER - MODEL TRAINING")
    logger.info("="*80)

    # ========================================
    # 1. LOAD AND PREPARE DATA
    # ========================================
    logger.info("\n[1/6] Loading fixed-span data...")

    loader = FixedSpanDataLoader(data_path)
    loader.load_data()
    loader.clean_data()

    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_train_test_split(
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )

    # Store in loader for later access
    loader.X_train = X_train
    loader.X_val = X_val
    loader.X_test = X_test
    loader.y_train = y_train
    loader.y_val = y_val
    loader.y_test = y_test

    logger.info(f"Data splits:")
    logger.info(f"  Train: {len(X_train)} samples")
    logger.info(f"  Validation: {len(X_val)} samples")
    logger.info(f"  Test: {len(X_test)} samples")
    logger.info(f"  Fixed span: {loader.fixed_span} inches")
    logger.info(f"  Input features: {loader.get_feature_names()}")

    # ========================================
    # 2. FEATURE ENGINEERING
    # ========================================
    logger.info("\n[2/6] Engineering features...")

    engineer = FixedSpanFeatureEngineer()
    X_train_eng = engineer.fit_transform(X_train, loader.get_feature_names())
    X_val_eng = engineer.transform(X_val)
    X_test_eng = engineer.transform(X_test)

    logger.info(f"Engineered features: {X_train_eng.shape[1]} (from {X_train.shape[1]} raw)")
    logger.info(f"Feature names: {engineer.get_feature_names()}")

    # ========================================
    # 3. TRAIN XGBOOST MODEL
    # ========================================
    logger.info("\n[3/6] Training XGBoost model...")

    # Use slightly deeper trees for fixed-span (fewer inputs = can afford more depth)
    xgb_model, xgb_val_metrics = train_xgboost_model(
        X_train_eng, y_train,
        X_val_eng, y_val,
        n_estimators=600,  # More trees for smaller feature space
        max_depth=10,       # Deeper trees
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )

    # Evaluate XGBoost on test set
    xgb_test_metrics = xgb_model.evaluate(X_test_eng, y_test, loader.get_output_names())

    logger.info("\nXGBoost Test Results:")
    logger.info(f"  Overall R²: {xgb_test_metrics['overall']['r2']:.4f}")
    logger.info(f"  Per-output R²:")
    for name, metrics in xgb_test_metrics['per_output'].items():
        logger.info(f"    {name}: {metrics['r2']:.4f} (MAE: {metrics['mae']:.2f})")

    # Feature importance
    logger.info("\nTop 10 XGBoost Feature Importances:")
    importance = xgb_model.get_feature_importance(engineer.get_feature_names(), top_n=10)
    for i, (feat, imp) in enumerate(importance.items(), 1):
        logger.info(f"  {i}. {feat}: {imp:.4f}")

    # ========================================
    # 4. TRAIN NEURAL NETWORK MODEL
    # ========================================
    logger.info("\n[4/6] Training Neural Network...")

    # Scale input features for neural network
    from sklearn.preprocessing import RobustScaler
    scaler_X_eng = RobustScaler()

    X_train_scaled = scaler_X_eng.fit_transform(X_train_eng)
    X_val_scaled = scaler_X_eng.transform(X_val_eng)
    X_test_scaled = scaler_X_eng.transform(X_test_eng)

    # Train NN - slightly simpler for fixed-span (fewer input dimensions)
    nn_model, nn_val_metrics = train_neural_network_model(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        input_dim=X_train_scaled.shape[1],
        hidden_dims=[48, 24, 12],  # Smaller network for 6 inputs
        output_dim=y_train.shape[1],
        dropout_rate=0.15,
        learning_rate=0.001,
        weight_decay=1e-4,
        batch_size=32,
        epochs=300,
        early_stopping_patience=30
    )

    # Evaluate Neural Network on test set
    nn_test_metrics = nn_model.evaluate(X_test_scaled, y_test, loader.get_output_names())

    logger.info("\nNeural Network Test Results:")
    logger.info(f"  Overall R²: {nn_test_metrics['overall']['r2']:.4f}")
    logger.info(f"  Per-output R²:")
    for name, metrics in nn_test_metrics['per_output'].items():
        logger.info(f"    {name}: {metrics['r2']:.4f} (MAE: {metrics['mae']:.2f})")

    # ========================================
    # 5. CREATE ENSEMBLE MODEL
    # ========================================
    logger.info("\n[5/6] Creating and optimizing ensemble...")

    # Optimize ensemble weights on validation set
    best_xgb_w, best_nn_w, _ = optimize_ensemble_weights(
        xgb_model, nn_model,
        X_val_eng, y_val,
        X_val_scaled=X_val_scaled
    )

    # Create ensemble with optimized weights
    ensemble = create_ensemble_model(xgb_model, nn_model, best_xgb_w, best_nn_w)

    # Evaluate ensemble on test set
    ensemble_test_metrics = ensemble.evaluate(
        X_test_eng, y_test, loader.get_output_names(),
        X_scaled=X_test_scaled
    )

    logger.info("\nEnsemble Test Results:")
    logger.info(f"  Overall R²: {ensemble_test_metrics['ensemble']['overall']['r2']:.4f}")
    logger.info(f"  Per-output R²:")
    for name, metrics in ensemble_test_metrics['ensemble']['per_output'].items():
        logger.info(f"    {name}: {metrics['r2']:.4f} (MAE: {metrics['mae']:.2f})")

    # ========================================
    # 6. SAVE MODELS AND RESULTS
    # ========================================
    logger.info("\n[6/6] Saving models and results...")

    # Create models directory for fixed-span variant
    models_dir = Path(__file__).parent.parent / "data" / "models_fixed_span_12ft"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save XGBoost model
    xgb_model_path = models_dir / "xgboost_v1.pkl"
    xgb_model.save(xgb_model_path)

    # Save Neural Network model
    nn_model_path = models_dir / "neural_v1.pt"
    nn_model.save(nn_model_path)

    # Save feature engineer
    import joblib
    engineer_path = models_dir / "feature_engineer.pkl"
    joblib.dump(engineer, engineer_path)
    logger.info(f"Saved feature engineer to {engineer_path}")

    # Save scaler
    scaler_X_eng_path = models_dir / "scaler_X_engineered.pkl"
    joblib.dump(scaler_X_eng, scaler_X_eng_path)
    logger.info(f"Saved feature scaler to {models_dir}")

    # Save ensemble metadata
    ensemble.save(models_dir)

    # ========================================
    # SAVE COMPREHENSIVE RESULTS
    # ========================================
    results = {
        'training_date': datetime.now().isoformat(),
        'model_variant': 'fixed_span_12ft',
        'fixed_span_inches': float(loader.fixed_span) if loader.fixed_span else 144.0,
        'dataset': {
            'source_path': str(data_path),
            'n_train': int(len(X_train)),
            'n_val': int(len(X_val)),
            'n_test': int(len(X_test)),
            'n_features_raw': int(X_train.shape[1]),
            'n_features_engineered': int(X_train_eng.shape[1]),
            'input_features': loader.get_feature_names()
        },
        'models': {
            'xgboost': {
                'test_metrics': xgb_test_metrics,
                'model_path': str(xgb_model_path.name)
            },
            'neural_network': {
                'test_metrics': nn_test_metrics,
                'model_path': str(nn_model_path.name),
                'training_epochs': len(nn_model.training_history['train_loss'])
            },
            'ensemble': {
                'test_metrics': ensemble_test_metrics,
                'xgb_weight': float(best_xgb_w),
                'nn_weight': float(best_nn_w)
            }
        },
        'feature_importance': importance
    }

    # Convert numpy types to native Python types
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_native(obj.tolist())
        else:
            return obj

    results_path = models_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(convert_to_native(results), f, indent=2)

    logger.info(f"Saved training results to {results_path}")

    # ========================================
    # SUMMARY
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("FIXED-SPAN (12ft) TRAINING COMPLETE - SUMMARY")
    logger.info("="*80)

    output_names = loader.get_output_names()

    logger.info(f"\nTest Set Performance (R² scores):")
    logger.info(f"{'Model':<20} {'Overall':<12} " + " ".join(f"{n[:10]:<12}" for n in output_names))
    logger.info("-"*100)

    # XGBoost
    xgb_r2s = " ".join(f"{xgb_test_metrics['per_output'][n]['r2']:<12.4f}" for n in output_names)
    logger.info(f"{'XGBoost':<20} {xgb_test_metrics['overall']['r2']:<12.4f} {xgb_r2s}")

    # Neural Network
    nn_r2s = " ".join(f"{nn_test_metrics['per_output'][n]['r2']:<12.4f}" for n in output_names)
    logger.info(f"{'Neural Network':<20} {nn_test_metrics['overall']['r2']:<12.4f} {nn_r2s}")

    # Ensemble
    ens_metrics = ensemble_test_metrics['ensemble']
    ens_r2s = " ".join(f"{ens_metrics['per_output'][n]['r2']:<12.4f}" for n in output_names)
    logger.info(f"{'Ensemble':<20} {ens_metrics['overall']['r2']:<12.4f} {ens_r2s}")

    logger.info(f"\nSaved models to: {models_dir}")
    logger.info(f"  - xgboost_v1.pkl")
    logger.info(f"  - neural_v1.pt")
    logger.info(f"  - feature_engineer.pkl")
    logger.info(f"  - scaler_X_engineered.pkl")
    logger.info(f"  - ensemble_metadata.json")
    logger.info(f"  - training_results.json")

    # Check deflection target
    deflection_r2 = ens_metrics['per_output']['wingtip_deflection_in']['r2']
    logger.info(f"\nWingtip Deflection Achievement:")
    if deflection_r2 >= 0.95:
        logger.info(f"  ✓ PASSED: Deflection R² ({deflection_r2:.4f}) >= 0.95 target")
    elif deflection_r2 >= 0.90:
        logger.info(f"  ~ CLOSE: Deflection R² ({deflection_r2:.4f}) approaching 0.95 target")
    else:
        logger.warning(f"  ✗ NEEDS WORK: Deflection R² ({deflection_r2:.4f}) < 0.95 target")

    logger.info("\n" + "="*80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train fixed-span (12ft) DOE models")
    parser.add_argument(
        "--data-path",
        type=str,
        default=r"D:\nTop\DOERunner\gcp_10000sample_fixed_span_12ft\doe_summary.csv",
        help="Path to fixed-span DOE dataset CSV"
    )

    args = parser.parse_args()
    results = main(args.data_path)
