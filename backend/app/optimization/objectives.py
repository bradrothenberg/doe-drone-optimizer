"""
Objective Functions for Multi-Objective Drone Design Optimization
Clean interfaces wrapping ensemble model predictions
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DroneObjectives:
    """
    Wrapper for drone design objective functions

    Objectives:
    1. Maximize Range (nm) - convert to minimization: -range
    2. Maximize Endurance (hr) - convert to minimization: -endurance
    3. Minimize MTOW (lbm)
    4. Minimize Material Cost ($)
    """

    def __init__(self, ensemble_model, feature_engineer):
        """
        Initialize objective functions

        Args:
            ensemble_model: Trained ensemble model for predictions
            feature_engineer: Feature engineering pipeline
        """
        self.ensemble_model = ensemble_model
        self.feature_engineer = feature_engineer

    def evaluate(
        self,
        X: np.ndarray,
        return_uncertainty: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate all objectives for given design parameters

        Args:
            X: Design parameters (n_designs, 7)
            return_uncertainty: If True, also return prediction uncertainty

        Returns:
            Dictionary with objectives and optionally uncertainty
        """
        # Engineer features
        X_eng = self.feature_engineer.transform(X)

        # Predict performance
        predictions, uncertainty = self.ensemble_model.predict(
            X_eng,
            return_uncertainty=True
        )

        # Extract predictions
        range_nm = predictions[:, 0]
        endurance_hr = predictions[:, 1]
        mtow_lbm = predictions[:, 2]
        cost_usd = predictions[:, 3]

        results = {
            'range_nm': range_nm,
            'endurance_hr': endurance_hr,
            'mtow_lbm': mtow_lbm,
            'cost_usd': cost_usd
        }

        if return_uncertainty:
            results['uncertainty'] = {
                'range_nm': uncertainty[:, 0],
                'endurance_hr': uncertainty[:, 1],
                'mtow_lbm': uncertainty[:, 2],
                'cost_usd': uncertainty[:, 3]
            }

        return results

    def evaluate_for_minimization(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate objectives in minimization form (for optimizers)

        Args:
            X: Design parameters (n_designs, 7)

        Returns:
            Objectives array (n_designs, 4) with all minimization form:
            [-range, -endurance, mtow, cost]
        """
        results = self.evaluate(X, return_uncertainty=False)

        objectives = np.column_stack([
            -results['range_nm'],       # Maximize → minimize negative
            -results['endurance_hr'],   # Maximize → minimize negative
            results['mtow_lbm'],        # Minimize as-is
            results['cost_usd']         # Minimize as-is
        ])

        return objectives

    def evaluate_single_objective(
        self,
        X: np.ndarray,
        objective_name: str
    ) -> np.ndarray:
        """
        Evaluate a single objective

        Args:
            X: Design parameters (n_designs, 7)
            objective_name: One of ['range', 'endurance', 'mtow', 'cost']

        Returns:
            Objective values (n_designs,)
        """
        results = self.evaluate(X, return_uncertainty=False)

        objective_map = {
            'range': -results['range_nm'],      # Minimization form
            'endurance': -results['endurance_hr'],
            'mtow': results['mtow_lbm'],
            'cost': results['cost_usd']
        }

        if objective_name not in objective_map:
            raise ValueError(f"Unknown objective: {objective_name}")

        return objective_map[objective_name]


def weighted_sum_objective(
    objectives: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Combine multiple objectives via weighted sum

    Args:
        objectives: Objective values (n_designs, n_objectives)
        weights: Objective weights (n_objectives,)

    Returns:
        Weighted sum (n_designs,)
    """
    if len(weights) != objectives.shape[1]:
        raise ValueError(f"Weight dimension ({len(weights)}) must match "
                        f"objective dimension ({objectives.shape[1]})")

    if not np.isclose(np.sum(weights), 1.0):
        logger.warning(f"Weights do not sum to 1.0: {np.sum(weights)}")

    return objectives @ weights


def normalize_objectives(
    objectives: np.ndarray,
    bounds: Dict[str, Tuple[float, float]] = None
) -> np.ndarray:
    """
    Normalize objectives to [0, 1] range

    Args:
        objectives: Raw objective values (n_designs, n_objectives)
        bounds: Optional dict with (min, max) bounds for each objective
                If None, use min/max from data

    Returns:
        Normalized objectives (n_designs, n_objectives)
    """
    normalized = np.zeros_like(objectives)

    for i in range(objectives.shape[1]):
        if bounds is not None:
            obj_name = ['range', 'endurance', 'mtow', 'cost'][i]
            min_val, max_val = bounds.get(obj_name, (objectives[:, i].min(), objectives[:, i].max()))
        else:
            min_val = objectives[:, i].min()
            max_val = objectives[:, i].max()

        # Avoid division by zero
        if np.isclose(max_val, min_val):
            normalized[:, i] = 0.5
        else:
            normalized[:, i] = (objectives[:, i] - min_val) / (max_val - min_val)

    return normalized


def compute_hypervolume_indicator(
    pareto_front: np.ndarray,
    reference_point: np.ndarray
) -> float:
    """
    Compute hypervolume indicator for Pareto front quality

    Args:
        pareto_front: Pareto front objectives (n_pareto, n_objectives)
        reference_point: Reference point (n_objectives,) - nadir point

    Returns:
        Hypervolume value (higher is better)
    """
    try:
        from pymoo.indicators.hv import HV

        # Create hypervolume indicator
        ind = HV(ref_point=reference_point)

        # Calculate hypervolume
        hv = ind(pareto_front)

        return float(hv)

    except ImportError:
        logger.warning("pymoo HV indicator not available, returning 0.0")
        return 0.0


if __name__ == "__main__":
    # Test objective functions
    from app.models.ensemble import EnsembleDroneModel
    from app.models.feature_engineering import FeatureEngineer
    from pathlib import Path
    import joblib

    print("Loading trained models...")
    models_dir = Path(__file__).parent.parent.parent / "data" / "models"

    # Load ensemble
    ensemble = EnsembleDroneModel()
    ensemble.load_models(
        xgb_model_path=models_dir / "xgboost_v1.pkl",
        nn_model_path=models_dir / "neural_v1.pt",
        input_dim=17
    )

    # Load feature engineer
    engineer = joblib.load(models_dir / "feature_engineer.pkl")

    print("\nTesting objective functions...")

    # Create test designs
    X_test = np.array([
        [150, 150, 30, 30, -30, -30, 0.4],  # Moderate design
        [192, 216, 0, 0, 0, 0, 0.5],        # Large design
        [96, 72, 60, 60, -60, -60, 0.1]     # Small design
    ])

    # Create objectives
    objectives = DroneObjectives(ensemble, engineer)

    # Evaluate
    results = objectives.evaluate(X_test, return_uncertainty=True)

    print("\nObjective Evaluations:")
    print(f"{'Design':<10} {'Range (nm)':<15} {'Endurance (hr)':<18} {'MTOW (lbm)':<15} {'Cost ($)':<12}")
    print("-" * 80)

    for i in range(len(X_test)):
        print(f"Design {i+1:<3} "
              f"{results['range_nm'][i]:>10.1f} ± {results['uncertainty']['range_nm'][i]:>4.1f}  "
              f"{results['endurance_hr'][i]:>10.2f} ± {results['uncertainty']['endurance_hr'][i]:>4.2f}  "
              f"{results['mtow_lbm'][i]:>10.1f} ± {results['uncertainty']['mtow_lbm'][i]:>4.1f}  "
              f"${results['cost_usd'][i]:>9.0f} ± ${results['uncertainty']['cost_usd'][i]:>5.0f}")

    # Test minimization form
    print("\nMinimization Form Objectives:")
    obj_min = objectives.evaluate_for_minimization(X_test)
    print(obj_min)

    # Test normalization
    print("\nNormalized Objectives:")
    obj_norm = normalize_objectives(obj_min)
    print(obj_norm)

    print("\nObjective functions test complete!")
