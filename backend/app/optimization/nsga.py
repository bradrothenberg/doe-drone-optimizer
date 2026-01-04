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

    Objectives (5) - directions are configurable:
    - Range (nm) - default: maximize
    - Endurance (hr) - default: maximize
    - MTOW (lbm) - default: minimize
    - Material Cost ($) - default: minimize
    - Wingtip Deflection (in) - default: minimize
    """

    # Default optimization directions
    DEFAULT_OBJECTIVES = {
        'range_nm': 'maximize',
        'endurance_hr': 'maximize',
        'mtow_lbm': 'minimize',
        'cost_usd': 'minimize',
        'wingtip_deflection_in': 'minimize'
    }

    def __init__(
        self,
        ensemble_model,
        feature_engineer,
        user_constraints: Optional[Dict] = None,
        objectives: Optional[Dict[str, str]] = None
    ):
        """
        Initialize drone design optimization problem

        Args:
            ensemble_model: Trained ensemble model for predictions
            feature_engineer: Feature engineering pipeline
            user_constraints: Optional user-specified constraints
            objectives: Dict mapping output names to 'minimize' or 'maximize'
        """
        self.ensemble_model = ensemble_model
        self.feature_engineer = feature_engineer
        self.user_constraints = user_constraints or {}

        # Merge user objectives with defaults
        self.objectives = self.DEFAULT_OBJECTIVES.copy()
        if objectives:
            self.objectives.update(objectives)

        # Design variable bounds (from DOE dataset)
        xl = np.array([96, 72, 0, -20, -60, -60, 0.10])   # Lower bounds
        xu = np.array([192, 216, 65, 60, 60, 60, 0.65])  # Upper bounds

        super().__init__(
            n_var=7,        # 7 design variables
            n_obj=5,        # 5 objectives
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
        # Clamp wingtip deflection to non-negative (model can extrapolate to negative)
        wingtip_deflection_in = np.maximum(0, predictions[:, 4])

        # Apply user constraints as penalties
        penalty = np.zeros(len(X))

        # Geometric feasibility penalty: penalize unrealistic taper ratios and bowtie wings
        # Design variables: [LOA, Span, LE_Sweep_P1, LE_Sweep_P2, TE_Sweep_P1, TE_Sweep_P2, Panel_Break]
        loa = X[:, 0]
        span = X[:, 1]
        le_sweep_p1 = X[:, 2]
        le_sweep_p2 = X[:, 3]
        te_sweep_p1 = X[:, 4]
        te_sweep_p2 = X[:, 5]
        panel_break = X[:, 6]

        # Taper ratio penalty
        taper_diff_p2 = te_sweep_p2 - le_sweep_p2
        MIN_TAPER_DIFF = -5  # Allow slight expansion, but penalize severe negative taper
        taper_violation = np.maximum(0, MIN_TAPER_DIFF - taper_diff_p2)
        penalty += taper_violation * 100  # Strong penalty for unrealistic geometry

        # Bowtie penalty: penalize designs where chord becomes too small
        half_span = span / 2
        break_span = half_span * panel_break
        remaining_span = half_span - break_span

        le_sweep_p1_rad = np.radians(le_sweep_p1)
        le_sweep_p2_rad = np.radians(le_sweep_p2)
        te_sweep_p1_rad = np.radians(te_sweep_p1)
        te_sweep_p2_rad = np.radians(te_sweep_p2)

        # Chord at panel break and tip
        chord_at_break = loa - break_span * (np.tan(le_sweep_p1_rad) + np.tan(te_sweep_p1_rad))
        le_offset_at_break = np.tan(le_sweep_p1_rad) * break_span
        te_offset_at_break = np.tan(te_sweep_p1_rad) * break_span
        chord_at_tip = loa - (le_offset_at_break + np.tan(le_sweep_p2_rad) * remaining_span) - \
                       (te_offset_at_break + np.tan(te_sweep_p2_rad) * remaining_span)

        MIN_CHORD = 2.0  # Minimum chord in inches
        chord_violation_break = np.maximum(0, MIN_CHORD - chord_at_break)
        chord_violation_tip = np.maximum(0, MIN_CHORD - chord_at_tip)
        penalty += (chord_violation_break + chord_violation_tip) * 500  # Very strong penalty for bowtie

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

        if 'max_wingtip_deflection_in' in self.user_constraints:
            max_deflection = self.user_constraints['max_wingtip_deflection_in']
            violation = np.maximum(0, wingtip_deflection_in - max_deflection)
            penalty += violation * 10.0  # Penalty for excessive deflection

        # Build objectives based on configured directions
        # pymoo always minimizes, so we negate values we want to maximize
        def apply_direction(values, metric_name):
            if self.objectives.get(metric_name, 'minimize') == 'maximize':
                return -values + penalty  # Negate for maximization
            else:
                return values + penalty   # Keep as-is for minimization

        objectives = np.column_stack([
            apply_direction(range_nm, 'range_nm'),
            apply_direction(endurance_hr, 'endurance_hr'),
            apply_direction(mtow_lbm, 'mtow_lbm'),
            apply_direction(cost_usd, 'cost_usd'),
            apply_direction(wingtip_deflection_in, 'wingtip_deflection_in')
        ])

        out["F"] = objectives


def run_nsga2_optimization(
    ensemble_model,
    feature_engineer,
    user_constraints: Optional[Dict] = None,
    objectives: Optional[Dict[str, str]] = None,
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
        objectives: Dict mapping output names to 'minimize' or 'maximize'
        population_size: Population size for NSGA-II
        n_generations: Number of generations
        seed: Random seed for reproducibility

    Returns:
        Dictionary with optimization results
    """
    logger.info("Initializing NSGA-II optimization...")
    logger.info(f"Population size: {population_size}")
    logger.info(f"Generations: {n_generations}")
    if objectives:
        logger.info(f"Custom objectives: {objectives}")

    # Create problem
    problem = DroneDesignProblem(
        ensemble_model=ensemble_model,
        feature_engineer=feature_engineer,
        user_constraints=user_constraints,
        objectives=objectives
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
    pareto_objectives = res.F  # Objectives (n_pareto, 5) - includes penalties

    # Get clean predictions with uncertainty (without penalties)
    # This is critical: pareto_objectives contains penalty terms which corrupt the values
    X_eng = feature_engineer.transform(pareto_designs)
    predictions, uncertainty = ensemble_model.predict(X_eng, return_uncertainty=True)

    # Use clean predictions instead of penalty-corrupted objectives
    range_nm = predictions[:, 0]
    endurance_hr = predictions[:, 1]
    mtow_lbm = predictions[:, 2]
    cost_usd = predictions[:, 3]
    # Clamp wingtip deflection to non-negative values
    # The ML model can extrapolate to negative values for certain design combinations,
    # but physically deflection cannot be negative
    wingtip_deflection_in = np.maximum(0, predictions[:, 4])

    # Filter out designs that violate hard constraints
    # This ensures results strictly satisfy user-specified limits
    feasible_mask = np.ones(len(pareto_designs), dtype=bool)

    # Geometric feasibility: filter out unrealistic taper ratios and bowtie wings
    # Design variables: [LOA, Span, LE_Sweep_P1, LE_Sweep_P2, TE_Sweep_P1, TE_Sweep_P2, Panel_Break]
    loa = pareto_designs[:, 0]
    span = pareto_designs[:, 1]
    le_sweep_p1 = pareto_designs[:, 2]
    le_sweep_p2 = pareto_designs[:, 3]
    te_sweep_p1 = pareto_designs[:, 4]
    te_sweep_p2 = pareto_designs[:, 5]
    panel_break = pareto_designs[:, 6]

    # Taper ratio constraint: TE sweep should be >= LE sweep for each panel
    # This prevents "negative taper" where chord expands toward the tip
    # Allow small tolerance (5 degrees) for slightly expanding outer panels
    MIN_TAPER_DIFF = -5  # Minimum (TE_sweep - LE_sweep), negative allows slight expansion
    taper_diff_p1 = te_sweep_p1 - le_sweep_p1
    taper_diff_p2 = te_sweep_p2 - le_sweep_p2

    # Filter out designs with unrealistic taper (especially on outer panel P2)
    feasible_mask &= (taper_diff_p2 >= MIN_TAPER_DIFF)

    n_taper_filtered = np.sum(~(taper_diff_p2 >= MIN_TAPER_DIFF))
    if n_taper_filtered > 0:
        logger.info(f"Filtered {n_taper_filtered} designs with unrealistic outer panel taper ratio")

    # Bowtie detection: chord must be positive at panel break and wingtip
    # Chord at a spanwise location = LOA - (LE_offset + TE_offset)
    # where offsets are computed from sweep angles
    half_span = span / 2
    break_span = half_span * panel_break
    remaining_span = half_span - break_span

    # Convert sweep angles to radians for tangent calculation
    le_sweep_p1_rad = np.radians(le_sweep_p1)
    le_sweep_p2_rad = np.radians(le_sweep_p2)
    te_sweep_p1_rad = np.radians(te_sweep_p1)
    te_sweep_p2_rad = np.radians(te_sweep_p2)

    # LE and TE offsets at panel break (measured from root chord endpoints)
    le_offset_at_break = np.tan(le_sweep_p1_rad) * break_span
    te_offset_at_break = np.tan(te_sweep_p1_rad) * break_span

    # LE and TE offsets at wingtip
    le_offset_at_tip = le_offset_at_break + np.tan(le_sweep_p2_rad) * remaining_span
    te_offset_at_tip = te_offset_at_break + np.tan(te_sweep_p2_rad) * remaining_span

    # Chord at panel break = LOA - (le_offset - te_offset) when both sweep forward
    # More precisely: chord = (TE_y - LE_y) where TE_y = LOA - te_offset, LE_y = le_offset
    # So chord = LOA - te_offset - le_offset (when both sweep back, offsets are positive)
    # Actually: chord_at_break = LOA - le_offset_at_break - (-te_offset_at_break) if TE sweeps forward
    # Simpler: chord = (root_TE_y - te_offset) - (root_LE_y + le_offset) = LOA - le_offset - te_offset
    # Wait, need to be careful with signs. Let me compute properly:
    # At root: LE_y = 0, TE_y = LOA, chord = LOA
    # At panel break: LE_y = tan(le_sweep_p1) * break_span (positive = aft)
    #                 TE_y = LOA - tan(te_sweep_p1) * break_span (positive sweep = TE moves aft = smaller offset from LOA)
    # So chord_at_break = TE_y - LE_y = LOA - tan(te_sweep_p1)*break_span - tan(le_sweep_p1)*break_span
    #                   = LOA - break_span * (tan(le_sweep_p1) + tan(te_sweep_p1))
    # Hmm, that's not quite right either. Let me think again...
    #
    # Using the Planform.tsx logic:
    # LE offset (how far aft the LE moves) = tan(le_sweep) * span_distance
    # TE offset (how far aft the TE moves from the root TE) = tan(te_sweep) * span_distance
    # At panel break:
    #   LE_y = leOffset1 = tan(le_sweep_p1) * break_span
    #   TE_y = LOA - teOffset1 = LOA - tan(te_sweep_p1) * break_span
    #   chord = TE_y - LE_y = LOA - tan(te_sweep_p1)*break_span - tan(le_sweep_p1)*break_span
    # Bowtie if chord <= 0, i.e., LOA <= break_span * (tan(le_sweep_p1) + tan(te_sweep_p1))

    # Chord at panel break
    chord_at_break = loa - break_span * (np.tan(le_sweep_p1_rad) + np.tan(te_sweep_p1_rad))

    # Chord at wingtip
    # LE_y at tip = le_offset_at_break + tan(le_sweep_p2) * remaining_span
    # TE_y at tip = LOA - (te_offset_at_break + tan(te_sweep_p2) * remaining_span)
    # chord at tip = TE_y - LE_y
    chord_at_tip = loa - (le_offset_at_break + np.tan(le_sweep_p2_rad) * remaining_span) - \
                   (te_offset_at_break + np.tan(te_sweep_p2_rad) * remaining_span)
    # Simplify: chord_at_tip = LOA - le_offset_at_tip - te_offset_at_tip
    #                        = LOA - half_span * (tan(le_sweep_p1)*pb + tan(le_sweep_p2)*(1-pb) +
    #                                             tan(te_sweep_p1)*pb + tan(te_sweep_p2)*(1-pb))

    # Minimum chord requirement (in inches) - must have at least some positive chord
    MIN_CHORD = 2.0  # 2 inches minimum chord

    bowtie_at_break = chord_at_break < MIN_CHORD
    bowtie_at_tip = chord_at_tip < MIN_CHORD
    bowtie_mask = bowtie_at_break | bowtie_at_tip

    feasible_mask &= ~bowtie_mask

    n_bowtie_filtered = np.sum(bowtie_mask)
    if n_bowtie_filtered > 0:
        logger.info(f"Filtered {n_bowtie_filtered} designs with bowtie geometry (chord < {MIN_CHORD}in)")

    if user_constraints:
        if 'min_range_nm' in user_constraints and user_constraints['min_range_nm'] is not None:
            feasible_mask &= (range_nm >= user_constraints['min_range_nm'])

        if 'max_cost_usd' in user_constraints and user_constraints['max_cost_usd'] is not None:
            feasible_mask &= (cost_usd <= user_constraints['max_cost_usd'])

        if 'max_mtow_lbm' in user_constraints and user_constraints['max_mtow_lbm'] is not None:
            feasible_mask &= (mtow_lbm <= user_constraints['max_mtow_lbm'])

        if 'min_endurance_hr' in user_constraints and user_constraints['min_endurance_hr'] is not None:
            feasible_mask &= (endurance_hr >= user_constraints['min_endurance_hr'])

        if 'max_wingtip_deflection_in' in user_constraints and user_constraints['max_wingtip_deflection_in'] is not None:
            feasible_mask &= (wingtip_deflection_in <= user_constraints['max_wingtip_deflection_in'])

    # Apply filter
    n_total = len(pareto_designs)
    pareto_designs = pareto_designs[feasible_mask]
    range_nm = range_nm[feasible_mask]
    endurance_hr = endurance_hr[feasible_mask]
    mtow_lbm = mtow_lbm[feasible_mask]
    cost_usd = cost_usd[feasible_mask]
    wingtip_deflection_in = wingtip_deflection_in[feasible_mask]
    uncertainty = uncertainty[feasible_mask]

    n_feasible = len(pareto_designs)
    logger.info(f"Filtered to {n_feasible}/{n_total} feasible designs that satisfy all constraints")

    if n_feasible == 0:
        logger.warning("No designs satisfy all constraints! Consider relaxing constraints.")
        # Return empty results rather than raising an error
        # This allows the frontend to show a "no results" message

    # Package results
    results = {
        'pareto_designs': pareto_designs,
        'pareto_objectives': {
            'range_nm': range_nm,
            'endurance_hr': endurance_hr,
            'mtow_lbm': mtow_lbm,
            'cost_usd': cost_usd,
            'wingtip_deflection_in': wingtip_deflection_in
        },
        'uncertainty': {
            'range_nm': uncertainty[:, 0] if n_feasible > 0 else np.array([]),
            'endurance_hr': uncertainty[:, 1] if n_feasible > 0 else np.array([]),
            'mtow_lbm': uncertainty[:, 2] if n_feasible > 0 else np.array([]),
            'cost_usd': uncertainty[:, 3] if n_feasible > 0 else np.array([]),
            'wingtip_deflection_in': uncertainty[:, 4] if n_feasible > 0 else np.array([])
        },
        'n_generations': res.algorithm.n_gen,
        'n_pareto': n_feasible,
        'n_total_before_filter': n_total,
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
            - max_wingtip_deflection_in: Maximum wingtip deflection (in)
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

        if 'max_wingtip_deflection_in' in constraints:
            if results['pareto_objectives']['wingtip_deflection_in'][i] > constraints['max_wingtip_deflection_in']:
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
