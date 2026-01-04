"""
Training Script with 5-Fold Cross-Validation and Bootstrap Ensemble
For Fixed-Span (12ft) DOE Drone Optimizer Models

Features:
- 5-fold cross-validation for robust performance estimates
- Bootstrap ensemble (5 XGBoost models) for uncertainty quantification
- Structural features for improved deflection prediction
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
from typing import List, Dict, Tuple
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

from app.models.data_loader import DOEDataLoader
from app.models.xgboost_model import XGBoostDroneModel, train_xgboost_model
from app.models.feature_engineering import FixedSpanFeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FixedSpanDataLoader(DOEDataLoader):
    """Data loader for fixed-span (12ft / 144") dataset."""

    INPUT_FEATURES = [
        'loa', 'le_sweep_p1', 'le_sweep_p2',
        'te_sweep_p1', 'te_sweep_p2', 'panel_break'
    ]

    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.fixed_span = None

    def load_data(self) -> pd.DataFrame:
        df = super().load_data()
        span_values = df['span'].unique()
        if len(span_values) == 1:
            self.fixed_span = span_values[0]
            logger.info(f"Fixed span confirmed: {self.fixed_span} inches")
        return df


class BootstrapXGBoostEnsemble:
    """
    Bootstrap ensemble of XGBoost models for uncertainty quantification.

    Trains N models with different random seeds and computes predictions
    as mean ± std across ensemble members.
    """

    def __init__(self, n_models: int = 5):
        self.n_models = n_models
        self.models: List[XGBoostDroneModel] = []
        self.random_seeds = list(range(42, 42 + n_models))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            **xgb_params) -> 'BootstrapXGBoostEnsemble':
        """
        Train ensemble of XGBoost models with different random seeds.
        """
        logger.info(f"Training bootstrap ensemble with {self.n_models} models...")

        for i, seed in enumerate(self.random_seeds):
            logger.info(f"  Training model {i+1}/{self.n_models} (seed={seed})...")

            model, _ = train_xgboost_model(
                X_train, y_train,
                X_val, y_val,
                random_state=seed,
                **xgb_params
            )
            self.models.append(model)

        logger.info(f"Bootstrap ensemble training complete ({self.n_models} models)")
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty from ensemble.

        Returns:
            predictions: Mean predictions (n_samples, n_outputs)
            uncertainty: Std across ensemble (n_samples, n_outputs)
        """
        all_predictions = []

        for model in self.models:
            pred = model.predict(X)
            all_predictions.append(pred)

        # Stack: (n_models, n_samples, n_outputs)
        stacked = np.stack(all_predictions, axis=0)

        # Compute mean and std
        mean_pred = np.mean(stacked, axis=0)
        std_pred = np.std(stacked, axis=0)

        return mean_pred, std_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 output_names: List[str]) -> Dict:
        """Evaluate ensemble on test data."""
        predictions, uncertainty = self.predict(X)

        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        metrics = {
            'per_output': {},
            'overall': {},
            'uncertainty': {}
        }

        r2_scores = []
        for i, name in enumerate(output_names):
            r2 = r2_score(y[:, i], predictions[:, i])
            mae = mean_absolute_error(y[:, i], predictions[:, i])
            rmse = np.sqrt(mean_squared_error(y[:, i], predictions[:, i]))
            mean_uncertainty = np.mean(uncertainty[:, i])

            metrics['per_output'][name] = {
                'r2': r2,
                'mae': mae,
                'rmse': rmse
            }
            metrics['uncertainty'][name] = {
                'mean_std': mean_uncertainty,
                'coverage_1std': float(np.mean(np.abs(y[:, i] - predictions[:, i]) <= uncertainty[:, i]))
            }
            r2_scores.append(r2)

        metrics['overall']['r2'] = np.mean(r2_scores)
        metrics['overall']['mean_uncertainty'] = np.mean(uncertainty)

        return metrics

    def save(self, models_dir: Path):
        """Save all ensemble models."""
        for i, model in enumerate(self.models):
            model_path = models_dir / f"xgboost_bootstrap_{i}.pkl"
            model.save(model_path)

        # Save ensemble metadata
        metadata = {
            'n_models': self.n_models,
            'random_seeds': self.random_seeds,
            'model_files': [f"xgboost_bootstrap_{i}.pkl" for i in range(self.n_models)]
        }
        metadata_path = models_dir / "bootstrap_ensemble_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved bootstrap ensemble ({self.n_models} models) to {models_dir}")

    @classmethod
    def load(cls, models_dir: Path) -> 'BootstrapXGBoostEnsemble':
        """Load ensemble from saved models."""
        metadata_path = models_dir / "bootstrap_ensemble_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        ensemble = cls(n_models=metadata['n_models'])
        ensemble.random_seeds = metadata['random_seeds']

        for model_file in metadata['model_files']:
            model = XGBoostDroneModel.load(models_dir / model_file)
            ensemble.models.append(model)

        return ensemble


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    feature_engineer: FixedSpanFeatureEngineer,
    n_folds: int = 5,
    xgb_params: dict = None
) -> Dict:
    """
    Run k-fold cross-validation for XGBoost model.

    Returns:
        Dictionary with per-fold and aggregate metrics
    """
    logger.info(f"\nRunning {n_folds}-fold cross-validation...")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    if xgb_params is None:
        xgb_params = {
            'n_estimators': 600,
            'max_depth': 10,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }

    output_names = ['range_nm', 'endurance_hr', 'mtow_lbm', 'material_cost_usd', 'wingtip_deflection_in']

    cv_results = {
        'fold_metrics': [],
        'per_output': {name: {'r2': [], 'mae': [], 'rmse': []} for name in output_names},
        'overall': {'r2': []}
    }

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        logger.info(f"  Fold {fold+1}/{n_folds}...")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Feature engineering
        engineer_fold = FixedSpanFeatureEngineer()
        X_train_eng = engineer_fold.fit_transform(X_train_fold)
        X_val_eng = engineer_fold.transform(X_val_fold)

        # Train XGBoost
        model, _ = train_xgboost_model(
            X_train_eng, y_train_fold,
            X_val_eng, y_val_fold,
            random_state=42 + fold,
            **xgb_params
        )

        # Evaluate
        metrics = model.evaluate(X_val_eng, y_val_fold, output_names)
        cv_results['fold_metrics'].append(metrics)
        cv_results['overall']['r2'].append(metrics['overall']['r2'])

        for name in output_names:
            cv_results['per_output'][name]['r2'].append(metrics['per_output'][name]['r2'])
            cv_results['per_output'][name]['mae'].append(metrics['per_output'][name]['mae'])
            cv_results['per_output'][name]['rmse'].append(metrics['per_output'][name]['rmse'])

    # Compute aggregate statistics
    cv_results['summary'] = {
        'overall': {
            'r2_mean': float(np.mean(cv_results['overall']['r2'])),
            'r2_std': float(np.std(cv_results['overall']['r2']))
        },
        'per_output': {}
    }

    for name in output_names:
        cv_results['summary']['per_output'][name] = {
            'r2_mean': float(np.mean(cv_results['per_output'][name]['r2'])),
            'r2_std': float(np.std(cv_results['per_output'][name]['r2'])),
            'mae_mean': float(np.mean(cv_results['per_output'][name]['mae'])),
            'mae_std': float(np.std(cv_results['per_output'][name]['mae']))
        }

    logger.info(f"\nCross-validation results ({n_folds} folds):")
    logger.info(f"  Overall R²: {cv_results['summary']['overall']['r2_mean']:.4f} ± {cv_results['summary']['overall']['r2_std']:.4f}")
    logger.info(f"  Per-output R²:")
    for name in output_names:
        stats = cv_results['summary']['per_output'][name]
        logger.info(f"    {name}: {stats['r2_mean']:.4f} ± {stats['r2_std']:.4f}")

    return cv_results


def main(data_path: str = None):
    """
    Main training pipeline with CV and bootstrap ensemble.
    """
    if data_path is None:
        data_path = r"D:\nTop\DOERunner\gcp_10000sample_fixed_span_12ft\doe_summary.csv"

    logger.info("="*80)
    logger.info("FIXED-SPAN MODEL TRAINING WITH CV AND BOOTSTRAP ENSEMBLE")
    logger.info("="*80)

    # ========================================
    # 1. LOAD DATA
    # ========================================
    logger.info("\n[1/5] Loading data...")

    loader = FixedSpanDataLoader(data_path)
    loader.load_data()
    loader.clean_data()

    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_train_test_split(
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )

    logger.info(f"Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Combine train+val for CV
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.vstack([y_train, y_val])

    # ========================================
    # 2. CROSS-VALIDATION
    # ========================================
    logger.info("\n[2/5] Running 5-fold cross-validation...")

    engineer = FixedSpanFeatureEngineer()

    xgb_params = {
        'n_estimators': 600,
        'max_depth': 10,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    }

    cv_results = run_cross_validation(
        X_train_full, y_train_full,
        engineer,
        n_folds=5,
        xgb_params=xgb_params
    )

    # ========================================
    # 3. FEATURE ENGINEERING (full training set)
    # ========================================
    logger.info("\n[3/5] Engineering features...")

    engineer = FixedSpanFeatureEngineer()
    X_train_eng = engineer.fit_transform(X_train, loader.get_feature_names())
    X_val_eng = engineer.transform(X_val)
    X_test_eng = engineer.transform(X_test)

    logger.info(f"Engineered features: {X_train_eng.shape[1]} (from {X_train.shape[1]} raw)")
    logger.info(f"  Includes structural features: Span², Span³, Bending_Moment_Proxy, etc.")

    # ========================================
    # 4. BOOTSTRAP ENSEMBLE
    # ========================================
    logger.info("\n[4/5] Training bootstrap ensemble (5 models)...")

    bootstrap_ensemble = BootstrapXGBoostEnsemble(n_models=5)
    bootstrap_ensemble.fit(
        X_train_eng, y_train,
        X_val_eng, y_val,
        **xgb_params
    )

    # Evaluate on test set
    output_names = loader.get_output_names()
    bootstrap_metrics = bootstrap_ensemble.evaluate(X_test_eng, y_test, output_names)

    logger.info("\nBootstrap Ensemble Test Results:")
    logger.info(f"  Overall R²: {bootstrap_metrics['overall']['r2']:.4f}")
    logger.info(f"  Mean uncertainty: {bootstrap_metrics['overall']['mean_uncertainty']:.4f}")
    logger.info(f"  Per-output R² and uncertainty:")
    for name in output_names:
        r2 = bootstrap_metrics['per_output'][name]['r2']
        unc = bootstrap_metrics['uncertainty'][name]['mean_std']
        cov = bootstrap_metrics['uncertainty'][name]['coverage_1std']
        logger.info(f"    {name}: R²={r2:.4f}, σ={unc:.3f}, coverage={cov:.1%}")

    # Also train single best model for compatibility
    logger.info("\n  Training primary XGBoost model...")
    primary_model, _ = train_xgboost_model(
        X_train_eng, y_train,
        X_val_eng, y_val,
        random_state=42,
        **xgb_params
    )
    primary_metrics = primary_model.evaluate(X_test_eng, y_test, output_names)

    # ========================================
    # 5. SAVE MODELS AND RESULTS
    # ========================================
    logger.info("\n[5/5] Saving models and results...")

    models_dir = Path(__file__).parent.parent / "data" / "models_fixed_span_12ft"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save primary model
    primary_model.save(models_dir / "xgboost_v1.pkl")

    # Save bootstrap ensemble
    bootstrap_ensemble.save(models_dir)

    # Save feature engineer
    import joblib
    joblib.dump(engineer, models_dir / "feature_engineer.pkl")

    # Save scaler (for potential NN use)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_eng)
    joblib.dump(scaler, models_dir / "scaler_X_engineered.pkl")

    # Save comprehensive results
    results = {
        'training_date': datetime.now().isoformat(),
        'model_variant': 'fixed_span_12ft_cv_bootstrap',
        'fixed_span_inches': float(loader.fixed_span) if loader.fixed_span else 144.0,
        'dataset': {
            'source_path': str(data_path),
            'n_train': int(len(X_train)),
            'n_val': int(len(X_val)),
            'n_test': int(len(X_test)),
            'n_features_raw': int(X_train.shape[1]),
            'n_features_engineered': int(X_train_eng.shape[1]),
            'feature_names': engineer.get_feature_names()
        },
        'cross_validation': {
            'n_folds': 5,
            'summary': cv_results['summary']
        },
        'bootstrap_ensemble': {
            'n_models': 5,
            'test_metrics': bootstrap_metrics
        },
        'primary_model': {
            'test_metrics': primary_metrics
        },
        'xgb_params': xgb_params
    }

    # Convert numpy types
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

    # ========================================
    # SUMMARY
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("="*80)

    logger.info(f"\n5-Fold Cross-Validation Results:")
    logger.info(f"  Overall R²: {cv_results['summary']['overall']['r2_mean']:.4f} ± {cv_results['summary']['overall']['r2_std']:.4f}")

    logger.info(f"\nBootstrap Ensemble Test Results (5 models):")
    logger.info(f"  Overall R²: {bootstrap_metrics['overall']['r2']:.4f}")

    logger.info(f"\nPer-Output Performance (Test Set):")
    logger.info(f"{'Output':<25} {'CV R² (±std)':<18} {'Bootstrap R²':<12} {'Uncertainty':<12}")
    logger.info("-"*70)
    for name in output_names:
        cv_r2 = cv_results['summary']['per_output'][name]['r2_mean']
        cv_std = cv_results['summary']['per_output'][name]['r2_std']
        boot_r2 = bootstrap_metrics['per_output'][name]['r2']
        unc = bootstrap_metrics['uncertainty'][name]['mean_std']
        logger.info(f"{name:<25} {cv_r2:.4f} ± {cv_std:.4f}     {boot_r2:<12.4f} {unc:<12.3f}")

    # Deflection target check
    deflection_r2 = bootstrap_metrics['per_output']['wingtip_deflection_in']['r2']
    logger.info(f"\nWingtip Deflection R²: {deflection_r2:.4f}")
    if deflection_r2 >= 0.95:
        logger.info("  ✓ TARGET MET (>= 0.95)")
    elif deflection_r2 >= 0.90:
        logger.info("  ~ CLOSE (>= 0.90, target 0.95)")
    else:
        logger.info("  ✗ NEEDS IMPROVEMENT (< 0.90)")

    logger.info(f"\nSaved to: {models_dir}")
    logger.info("  - xgboost_v1.pkl (primary model)")
    logger.info("  - xgboost_bootstrap_*.pkl (5 ensemble models)")
    logger.info("  - bootstrap_ensemble_metadata.json")
    logger.info("  - feature_engineer.pkl")
    logger.info("  - scaler_X_engineered.pkl")
    logger.info("  - training_results.json")

    logger.info("\n" + "="*80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train models with CV and bootstrap ensemble")
    parser.add_argument(
        "--data-path",
        type=str,
        default=r"D:\nTop\DOERunner\gcp_10000sample_fixed_span_12ft\doe_summary.csv",
        help="Path to fixed-span DOE dataset CSV"
    )

    args = parser.parse_args()
    results = main(args.data_path)
