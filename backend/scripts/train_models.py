"""
Training Script for DOE Drone Optimizer Models
Trains XGBoost, Neural Network, and creates Ensemble
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import logging
import json
from datetime import datetime

from app.models.data_loader import load_doe_data
from app.models.feature_engineering import engineer_features, FeatureEngineer
from app.models.xgboost_model import train_xgboost_model
from app.models.neural_model import train_neural_network_model
from app.models.ensemble import create_ensemble_model, optimize_ensemble_weights

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline"""

    logger.info("="*80)
    logger.info("DOE DRONE OPTIMIZER - MODEL TRAINING")
    logger.info("="*80)

    # ========================================
    # 1. LOAD AND PREPARE DATA
    # ========================================
    logger.info("\n[1/6] Loading and preparing data...")

    loader, _, _, _, _, _, _ = load_doe_data(
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )

    logger.info(f"Data splits:")
    logger.info(f"  Train: {len(loader.X_train)} samples")
    logger.info(f"  Validation: {len(loader.X_val)} samples")
    logger.info(f"  Test: {len(loader.X_test)} samples")

    # ========================================
    # 2. FEATURE ENGINEERING
    # ========================================
    logger.info("\n[2/6] Engineering features...")

    engineer = FeatureEngineer()
    X_train_eng = engineer.fit_transform(loader.X_train, loader.get_feature_names())
    X_val_eng = engineer.transform(loader.X_val)
    X_test_eng = engineer.transform(loader.X_test)

    logger.info(f"Engineered features: {X_train_eng.shape[1]} (from {loader.X_train.shape[1]} raw)")
    logger.info(f"Feature names: {engineer.get_feature_names()}")

    # ========================================
    # 3. TRAIN XGBOOST MODEL
    # ========================================
    logger.info("\n[3/6] Training XGBoost model...")

    xgb_model, xgb_val_metrics = train_xgboost_model(
        X_train_eng, loader.y_train,
        X_val_eng, loader.y_val,
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )

    # Evaluate XGBoost on test set
    xgb_test_metrics = xgb_model.evaluate(X_test_eng, loader.y_test, loader.get_output_names())

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

    # Scale engineered features for neural network (need new scalers for 17 features)
    from sklearn.preprocessing import RobustScaler
    scaler_X_eng = RobustScaler()
    scaler_y = RobustScaler()

    X_train_scaled = scaler_X_eng.fit_transform(X_train_eng)
    y_train_scaled = scaler_y.fit_transform(loader.y_train)

    X_val_scaled = scaler_X_eng.transform(X_val_eng)
    y_val_scaled = scaler_y.transform(loader.y_val)

    X_test_scaled = scaler_X_eng.transform(X_test_eng)
    y_test_scaled = scaler_y.transform(loader.y_test)

    nn_model, nn_val_metrics = train_neural_network_model(
        X_train_scaled, y_train_scaled,
        X_val_scaled, y_val_scaled,
        input_dim=X_train_scaled.shape[1],
        hidden_dims=[128, 64, 32],
        output_dim=4,
        dropout_rate=0.2,
        learning_rate=0.001,
        weight_decay=1e-4,
        batch_size=64,
        epochs=200,
        early_stopping_patience=20
    )

    # Evaluate Neural Network on test set
    nn_test_metrics = nn_model.evaluate(X_test_scaled, y_test_scaled, loader.get_output_names())

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
        X_val_eng, loader.y_val
    )

    # Create ensemble with optimized weights
    ensemble = create_ensemble_model(xgb_model, nn_model, best_xgb_w, best_nn_w)

    # Evaluate ensemble on test set
    ensemble_test_metrics = ensemble.evaluate(X_test_eng, loader.y_test, loader.get_output_names())

    logger.info("\nEnsemble Test Results:")
    logger.info(f"  Overall R²: {ensemble_test_metrics['ensemble']['overall']['r2']:.4f}")
    logger.info(f"  Per-output R²:")
    for name, metrics in ensemble_test_metrics['ensemble']['per_output'].items():
        logger.info(f"    {name}: {metrics['r2']:.4f} (MAE: {metrics['mae']:.2f})")

    # ========================================
    # 6. SAVE MODELS AND RESULTS
    # ========================================
    logger.info("\n[6/6] Saving models and results...")

    # Create models directory
    models_dir = Path(__file__).parent.parent / "data" / "models"
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

    # Save scalers (for engineered features)
    scaler_X_eng_path = models_dir / "scaler_X_engineered.pkl"
    scaler_y_path = models_dir / "scaler_y.pkl"
    joblib.dump(scaler_X_eng, scaler_X_eng_path)
    joblib.dump(scaler_y, scaler_y_path)
    logger.info(f"Saved scalers to {models_dir}")

    # Save ensemble metadata
    ensemble.save(models_dir)

    # ========================================
    # SAVE COMPREHENSIVE RESULTS
    # ========================================
    results = {
        'training_date': datetime.now().isoformat(),
        'dataset': {
            'n_train': int(len(loader.X_train)),
            'n_val': int(len(loader.X_val)),
            'n_test': int(len(loader.X_test)),
            'n_features_raw': int(loader.X_train.shape[1]),
            'n_features_engineered': int(X_train_eng.shape[1])
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

    results_path = models_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved training results to {results_path}")

    # ========================================
    # SUMMARY
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("="*80)

    logger.info(f"\nTest Set Performance (R² scores):")
    logger.info(f"{'Model':<20} {'Overall':<12} {'Range':<12} {'Endurance':<12} {'MTOW':<12} {'Cost':<12}")
    logger.info("-"*80)

    # XGBoost
    logger.info(f"{'XGBoost':<20} {xgb_test_metrics['overall']['r2']:<12.4f} "
               f"{xgb_test_metrics['per_output']['Range']['r2']:<12.4f} "
               f"{xgb_test_metrics['per_output']['Endurance']['r2']:<12.4f} "
               f"{xgb_test_metrics['per_output']['MTOW']['r2']:<12.4f} "
               f"{xgb_test_metrics['per_output']['Cost']['r2']:<12.4f}")

    # Neural Network
    logger.info(f"{'Neural Network':<20} {nn_test_metrics['overall']['r2']:<12.4f} "
               f"{nn_test_metrics['per_output']['Range']['r2']:<12.4f} "
               f"{nn_test_metrics['per_output']['Endurance']['r2']:<12.4f} "
               f"{nn_test_metrics['per_output']['MTOW']['r2']:<12.4f} "
               f"{nn_test_metrics['per_output']['Cost']['r2']:<12.4f}")

    # Ensemble
    ens_metrics = ensemble_test_metrics['ensemble']
    logger.info(f"{'Ensemble':<20} {ens_metrics['overall']['r2']:<12.4f} "
               f"{ens_metrics['per_output']['Range']['r2']:<12.4f} "
               f"{ens_metrics['per_output']['Endurance']['r2']:<12.4f} "
               f"{ens_metrics['per_output']['MTOW']['r2']:<12.4f} "
               f"{ens_metrics['per_output']['Cost']['r2']:<12.4f}")

    logger.info(f"\nSaved models to: {models_dir}")
    logger.info(f"  - xgboost_v1.pkl")
    logger.info(f"  - neural_v1.pt")
    logger.info(f"  - feature_engineer.pkl")
    logger.info(f"  - scaler_X.pkl, scaler_y.pkl")
    logger.info(f"  - ensemble_metadata.json")
    logger.info(f"  - training_results.json")

    # Check if R² > 0.90 target achieved
    target_r2 = 0.90
    ensemble_r2 = ens_metrics['overall']['r2']

    logger.info(f"\nTarget Achievement:")
    if ensemble_r2 >= target_r2:
        logger.info(f"  ✓ PASSED: Ensemble R² ({ensemble_r2:.4f}) >= target ({target_r2:.2f})")
    else:
        logger.warning(f"  ✗ FAILED: Ensemble R² ({ensemble_r2:.4f}) < target ({target_r2:.2f})")

    logger.info("\n" + "="*80)

    return results


if __name__ == "__main__":
    results = main()
