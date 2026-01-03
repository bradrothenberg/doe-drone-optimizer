"""
NSGA-II Multi-Objective Optimizer for Drone Design
Uses pymoo library for genetic algorithm-based optimization
"""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from typing import Dict, List, Tuple, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DroneDesignProblem(Problem):
    """
    Multi-objective optimization problem for drone design

    Design Variables (7):
    - LOA (Length Overall): 96-192 inches
    - Span: 72-216 inches
    - LE Sweep P1: 0-65 degrees
    - LE Sweep P2: -20-60 degrees
    - TE Sweep P1: -60-60 degrees
    - TE Sweep P2: -60-60 degrees
    - Panel Break: 0.10-0.65 (fraction of span)

    Objectives (4):
    - Maximize Range (nm)
    - Maximize Endurance (hr)
    - Minimize MTOW (lbm)
    - Minimize Material Cost ($)
    """

    def __init__(
        self,
        ensemble_model,
        feature_engineer,
        user_constraints: Optional[Dict] = None
    ):
        """
        Initialize drone design optimization problem

        Args:
            ensemble_model: Trained ensemble model for predictions
            feature_engineer: Feature engineering pipeline
            user_constraints: Optional user-specified constraints
        """
        self.ensemble_model = ensemble_model
        self.feature_engineer = feature_engineer
        self.user_constraints = user_constraints or {}

        # Design variable bounds (from DOE dataset)
        xl = np.array([96, 72, 0, -20, -60, -60, 0.10])   # Lower bounds
        xu = np.array([192, 216, 65, 60, 60, 60, 0.65])  # Upper bounds

        super().__init__(
            n_var=7,        # 7 design variables
            n_obj=4,        # 4 objectives
            n_constr=0,     # No hard constraints (using penalty method)
            xl=xl,
            xu=xu
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objectives for given design parameters

        Args:
            X: Design parameters (n_designs, 7)
            out: Output dictionary to store objectives
        """
        # Engineer features
        X_eng = self.feature_engineer.transform(X)

        # Predict performance using ensemble
        predictions, uncertainty = self.ensemble_model.predict(
            X_eng,
            return_uncertainty=True
        )

        # Extract predictions
        range_nm = predictions[:, 0]
        endurance_hr = predictions[:, 1]
        mtow_lbm = predictions[:, 2]
        cost_usd = predictions[:, 3]

        # Apply user constraints as penalties
        penalty = np.zeros(len(X))

        if 'min_range_nm' in self.user_constraints:
            min_range = self.user_constraints['min_range_nm']
            violation = np.maximum(0, min_range - range_nm)
            penalty += violation * 1000  # Large penalty for constraint violation

        if 'max_cost_usd' in self.user_constraints:
            max_cost = self.user_constraints['max_cost_usd']
            violation = np.maximum(0, cost_usd - max_cost)
            penalty += violation * 0.01

        if 'max_mtow_lbm' in self.user_constraints:
            max_mtow = self.user_constraints['max_mtow_lbm']
            violation = np.maximum(0, mtow_lbm - max_mtow)
            penalty += violation * 1.0

        if 'min_endurance_hr' in self.user_constraints:
            min_endurance = self.user_constraints['min_endurance_hr']
            violation = np.maximum(0, min_endurance - endurance_hr)
            penalty += violation * 100

        # Objectives (minimize negative for maximization)
        objectives = np.column_stack([
            -range_nm + penalty,      # Maximize range
            -endurance_hr + penalty,  # Maximize endurance
            mtow_lbm + penalty,       # Minimize MTOW
            cost_usd + penalty        # Minimize cost
        ])

        out["F"] = objectives


def run_nsga2_optimization(
    ensemble_model,
    feature_engineer,
    user_constraints: Optional[Dict] = None,
    population_size: int = 200,
    n_generations: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run NSGA-II optimization to find Pareto-optimal designs

    Args:
        ensemble_model: Trained ensemble model
        feature_engineer: Feature engineering pipeline
        user_constraints: User-specified constraints
        population_size: Population size for NSGA-II
        n_generations: Number of generations
        seed: Random seed for reproducibility

    Returns:
        Dictionary with optimization results
    """
    logger.info("Initializing NSGA-II optimization...")
    logger.info(f"Population size: {population_size}")
    logger.info(f"Generations: {n_generations}")

    # Create problem
    problem = DroneDesignProblem(
        ensemble_model=ensemble_model,
        feature_engineer=feature_engineer,
        user_constraints=user_constraints
    )

    # Configure NSGA-II algorithm
    algorithm = NSGA2(
        pop_size=population_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),     # Simulated Binary Crossover
        mutation=PM(eta=20),                  # Polynomial Mutation
        eliminate_duplicates=True
    )

    # Set termination criterion
    termination = get_termination("n_gen", n_generations)

    # Run optimization
    logger.info("Running NSGA-II optimization...")
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=False
    )

    logger.info(f"Optimization complete after {res.algorithm.n_gen} generations")

    # Check if optimization found valid solutions
    if res.X is None or len(res.X) == 0:
        logger.error("Optimization failed to find valid solutions")
        logger.error("This typically indicates overly restrictive constraints or numerical issues")
        raise ValueError(
            "NSGA-II optimization failed to find valid solutions. "
            "Please check constraints and try relaxing them or adjusting population/generations."
        )

    logger.info(f"Pareto front size: {len(res.X)}")

    # Extract results
    pareto_designs = res.X  # Design parameters (n_pareto, 7)
    pareto_objectives = res.F  # Objectives (n_pareto, 4) - includes penalties

    # Get clean predictions with uncertainty (without penalties)
    # This is critical: pareto_objectives contains penalty terms which corrupt the values
    X_eng = feature_engineer.transform(pareto_designs)
    predictions, uncertainty = ensemble_model.predict(X_eng, return_uncertainty=True)

    # Use clean predictions instead of penalty-corrupted objectives
    range_nm = predictions[:, 0]
    endurance_hr = predictions[:, 1]
    mtow_lbm = predictions[:, 2]
    cost_usd = predictions[:, 3]

    # Package results
    results = {
        'pareto_designs': pareto_designs,
        'pareto_objectives': {
            'range_nm': range_nm,
            'endurance_hr': endurance_hr,
            'mtow_lbm': mtow_lbm,
            'cost_usd': cost_usd
        },
        'uncertainty': {
            'range_nm': uncertainty[:, 0],
            'endurance_hr': uncertainty[:, 1],
            'mtow_lbm': uncertainty[:, 2],
            'cost_usd': uncertainty[:, 3]
        },
        'n_generations': res.algorithm.n_gen,
        'n_pareto': len(pareto_designs),
        'algorithm': 'NSGA-II',
        'population_size': population_size
    }

    return results


def optimize_with_constraints(
    ensemble_model,
    feature_engineer,
    constraints: Dict[str, float],
    population_size: int = 200,
    n_generations: int = 100
) -> Dict[str, Any]:
    """
    Convenience function for constrained optimization

    Args:
        ensemble_model: Trained ensemble model
        feature_engineer: Feature engineering pipeline
        constraints: User constraints dictionary
            - min_range_nm: Minimum range (nm)
            - max_cost_usd: Maximum cost ($)
            - max_mtow_lbm: Maximum MTOW (lbm)
            - min_endurance_hr: Minimum endurance (hr)
        population_size: Population size
        n_generations: Number of generations

    Returns:
        Optimization results dictionary
    """
    logger.info("Running constrained optimization...")
    logger.info(f"Constraints: {constraints}")

    results = run_nsga2_optimization(
        ensemble_model=ensemble_model,
        feature_engineer=feature_engineer,
        user_constraints=constraints,
        population_size=population_size,
        n_generations=n_generations
    )

    # Check constraint satisfaction
    n_feasible = 0
    for i in range(results['n_pareto']):
        feasible = True

        if 'min_range_nm' in constraints:
            if results['pareto_objectives']['range_nm'][i] < constraints['min_range_nm']:
                feasible = False

        if 'max_cost_usd' in constraints:
            if results['pareto_objectives']['cost_usd'][i] > constraints['max_cost_usd']:
                feasible = False

        if 'max_mtow_lbm' in constraints:
            if results['pareto_objectives']['mtow_lbm'][i] > constraints['max_mtow_lbm']:
                feasible = False

        if 'min_endurance_hr' in constraints:
            if results['pareto_objectives']['endurance_hr'][i] < constraints['min_endurance_hr']:
                feasible = False

        if feasible:
            n_feasible += 1

    results['n_feasible'] = n_feasible
    results['feasibility_rate'] = n_feasible / results['n_pareto'] if results['n_pareto'] > 0 else 0

    logger.info(f"Feasible designs: {n_feasible}/{results['n_pareto']} ({results['feasibility_rate']:.1%})")

    return results


if __name__ == "__main__":
    # Test NSGA-II optimization
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

    print("\nRunning unconstrained optimization...")
    results = run_nsga2_optimization(
        ensemble_model=ensemble,
        feature_engineer=engineer,
        population_size=100,
        n_generations=50
    )

    print(f"\nOptimization Results:")
    print(f"  Pareto front size: {results['n_pareto']}")
    print(f"  Range: {results['pareto_objectives']['range_nm'].min():.1f} - {results['pareto_objectives']['range_nm'].max():.1f} nm")
    print(f"  Endurance: {results['pareto_objectives']['endurance_hr'].min():.1f} - {results['pareto_objectives']['endurance_hr'].max():.1f} hr")
    print(f"  MTOW: {results['pareto_objectives']['mtow_lbm'].min():.1f} - {results['pareto_objectives']['mtow_lbm'].max():.1f} lbm")
    print(f"  Cost: ${results['pareto_objectives']['cost_usd'].min():.0f} - ${results['pareto_objectives']['cost_usd'].max():.0f}")

    print("\nRunning constrained optimization...")
    constrained_results = optimize_with_constraints(
        ensemble_model=ensemble,
        feature_engineer=engineer,
        constraints={
            'min_range_nm': 1500,
            'max_cost_usd': 35000,
            'max_mtow_lbm': 3000
        },
        population_size=100,
        n_generations=50
    )

    print(f"\nConstrained Optimization Results:")
    print(f"  Pareto front size: {constrained_results['n_pareto']}")
    print(f"  Feasible designs: {constrained_results['n_feasible']} ({constrained_results['feasibility_rate']:.1%})")
