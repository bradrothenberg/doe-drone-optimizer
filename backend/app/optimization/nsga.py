"""
NSGA-II Multi-Objective Optimizer for Drone Design
Uses pymoo library for genetic algorithm-based optimization
Supports both variable-span (7 inputs) and fixed-span (6 inputs) modes
"""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from typing import Dict, List, Tuple, Any, Optional
import logging

from app.core.config import settings


class TaperAwareSampling(Sampling):
    """
    Custom sampling that ensures initial population includes designs
    satisfying taper ratio constraints. This helps NSGA-II explore the
    feasible region when geometric constraints are active.
    """

    def __init__(self, geometric_constraints=None, fixed_span=None):
        super().__init__()
        self.geometric_constraints = geometric_constraints or {}
        self.fixed_span = fixed_span

    def _do(self, problem, n_samples, **kwargs):
        # Start with random sampling
        X = np.random.random((n_samples, problem.n_var))

        # Scale to bounds
        xl, xu = problem.bounds()
        X = xl + X * (xu - xl)

        gc = self.geometric_constraints
        if not gc:
            return X

        # Check if we have taper constraints that need special handling
        max_taper_p2 = gc.get('max_taper_ratio_p2')
        le_sweep_p2_fixed = gc.get('le_sweep_p2_fixed')

        if max_taper_p2 is not None and le_sweep_p2_fixed is not None:
            # We need to bias TE P2 values to achieve lower taper ratios
            # Taper P2 depends on LE P2 and TE P2 sweep angles
            # Lower taper = need TE P2 significantly lower than LE P2

            # Determine which column is TE_Sweep_P2
            if self.fixed_span is not None:
                # Fixed-span: [LOA, LE_P1, LE_P2, TE_P1, TE_P2, Panel_Break]
                te_p2_idx = 4
            else:
                # Variable-span: [LOA, Span, LE_P1, LE_P2, TE_P1, TE_P2, Panel_Break]
                te_p2_idx = 5

            # For half the population, bias TE P2 toward values that give lower taper
            # With LE P2 = 40°, we need TE P2 < 40° (ideally negative or low positive)
            # to achieve taper ratios below 0.8
            n_biased = n_samples // 2

            # Sample TE P2 from a range biased toward lower values
            # Use the lower half of the TE P2 range to explore low taper designs
            te_p2_min = xl[te_p2_idx]
            te_p2_max = xu[te_p2_idx]

            # Bias toward lower TE P2 values (lower 60% of range)
            biased_max = te_p2_min + 0.6 * (te_p2_max - te_p2_min)
            X[:n_biased, te_p2_idx] = np.random.uniform(te_p2_min, biased_max, n_biased)

            logger.info(f"TaperAwareSampling: Biased {n_biased}/{n_samples} samples toward lower TE P2 values [{te_p2_min:.1f}, {biased_max:.1f}]")

        return X

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_chords(loa, span, le_sweep_p1, le_sweep_p2, te_sweep_p1, te_sweep_p2, panel_break):
    """
    Calculate chord at root, panel break, and wingtip.

    Wing geometry:
    - LE_x(y) = y * tan(LE_sweep)  (LE moves aft with positive sweep)
    - TE_x(y) = LOA + y * tan(TE_sweep)  (TE moves aft with positive sweep)
    - Chord(y) = TE_x(y) - LE_x(y) = LOA + y * tan(TE) - y * tan(LE)
               = LOA - y * (tan(LE) - tan(TE))

    For taper (chord decreasing outboard): tan(LE) > tan(TE)
    Example: LE=40°, TE=-40° gives tan(40)-tan(-40) = 0.84-(-0.84) = 1.68 > 0 → tapers

    Args:
        loa: Length Overall (root chord) in inches
        span: Full wingspan in inches
        le_sweep_p1: Leading edge sweep panel 1 (degrees)
        le_sweep_p2: Leading edge sweep panel 2 (degrees)
        te_sweep_p1: Trailing edge sweep panel 1 (degrees)
        te_sweep_p2: Trailing edge sweep panel 2 (degrees)
        panel_break: Spanwise location of panel break (fraction 0-1)

    Returns:
        tuple: (chord_root, chord_at_break, chord_tip)
    """
    half_span = span / 2
    break_span = half_span * panel_break
    remaining_span = half_span - break_span

    le_sweep_p1_rad = np.radians(le_sweep_p1)
    le_sweep_p2_rad = np.radians(le_sweep_p2)
    te_sweep_p1_rad = np.radians(te_sweep_p1)
    te_sweep_p2_rad = np.radians(te_sweep_p2)

    # Root chord is simply LOA
    chord_root = loa

    # Chord at panel break: LOA - break_span * (tan(LE_P1) - tan(TE_P1))
    chord_at_break = loa - break_span * (np.tan(le_sweep_p1_rad) - np.tan(te_sweep_p1_rad))

    # Chord at tip: continue from panel break with P2 sweeps
    # LE position at break: break_span * tan(LE_P1)
    # TE position at break: LOA + break_span * tan(TE_P1)
    # LE position at tip: break_span * tan(LE_P1) + remaining_span * tan(LE_P2)
    # TE position at tip: LOA + break_span * tan(TE_P1) + remaining_span * tan(TE_P2)
    # Chord at tip = TE_tip - LE_tip
    le_at_tip = np.tan(le_sweep_p1_rad) * break_span + np.tan(le_sweep_p2_rad) * remaining_span
    te_at_tip = loa + np.tan(te_sweep_p1_rad) * break_span + np.tan(te_sweep_p2_rad) * remaining_span
    chord_tip = te_at_tip - le_at_tip

    return chord_root, chord_at_break, chord_tip


def calculate_taper_ratios(loa, span, le_sweep_p1, le_sweep_p2, te_sweep_p1, te_sweep_p2, panel_break):
    """
    Calculate taper ratios for each panel.

    Taper ratio = outboard_chord / inboard_chord
    - Panel 1: chord_at_break / chord_root
    - Panel 2: chord_tip / chord_at_break

    Args:
        Same as calculate_chords

    Returns:
        tuple: (taper_p1, taper_p2)
    """
    chord_root, chord_at_break, chord_tip = calculate_chords(
        loa, span, le_sweep_p1, le_sweep_p2, te_sweep_p1, te_sweep_p2, panel_break
    )

    # Avoid division by zero
    chord_root = np.maximum(chord_root, 0.1)
    chord_at_break = np.maximum(chord_at_break, 0.1)

    taper_p1 = chord_at_break / chord_root
    taper_p2 = chord_tip / chord_at_break

    return taper_p1, taper_p2


class DroneDesignProblem(Problem):
    """
    Multi-objective optimization problem for drone design

    Variable-Span Mode - Design Variables (7):
    - LOA (Length Overall): 96-192 inches
    - Span: 72-216 inches
    - LE Sweep P1: 0-65 degrees
    - LE Sweep P2: -20-60 degrees
    - TE Sweep P1: -60-60 degrees
    - TE Sweep P2: -60-60 degrees
    - Panel Break: 0.10-0.65 (fraction of span)

    Fixed-Span Mode - Design Variables (6):
    - LOA (Length Overall): 96-192 inches
    - LE Sweep P1: 0-65 degrees
    - LE Sweep P2: -20-60 degrees
    - TE Sweep P1: -60-60 degrees
    - TE Sweep P2: -60-60 degrees
    - Panel Break: 0.10-0.65 (fraction of span)
    (Span is fixed at 144 inches / 12 feet)

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
        objectives: Optional[Dict[str, str]] = None,
        fixed_span: Optional[float] = None,
        allow_unrealistic_taper: bool = False,
        geometric_constraints: Optional[Dict] = None
    ):
        """
        Initialize drone design optimization problem

        Args:
            ensemble_model: Trained ensemble model for predictions
            feature_engineer: Feature engineering pipeline
            user_constraints: Optional user-specified constraints
            objectives: Dict mapping output names to 'minimize' or 'maximize'
            fixed_span: If provided, use fixed-span mode with this span value (inches)
            allow_unrealistic_taper: If True, skip taper ratio and bowtie filtering
            geometric_constraints: Optional dict with angle bounds and taper constraints
                - le_sweep_p1_fixed, le_sweep_p1_min, le_sweep_p1_max
                - le_sweep_p2_fixed, le_sweep_p2_min, le_sweep_p2_max
                - te_sweep_p1_fixed, te_sweep_p1_min, te_sweep_p1_max
                - te_sweep_p2_fixed, te_sweep_p2_min, te_sweep_p2_max
                - min_taper_ratio_p1, max_taper_ratio_p1
                - min_taper_ratio_p2, max_taper_ratio_p2
                - min_root_chord_ratio, max_root_chord_ratio
        """
        self.ensemble_model = ensemble_model
        self.feature_engineer = feature_engineer
        self.user_constraints = user_constraints or {}
        self.fixed_span = fixed_span
        self.allow_unrealistic_taper = allow_unrealistic_taper
        self.geometric_constraints = geometric_constraints or {}

        # Merge user objectives with defaults
        self.objectives = self.DEFAULT_OBJECTIVES.copy()
        if objectives:
            self.objectives.update(objectives)

        # Default design variable bounds
        # Fixed-span: [LOA, LE_Sweep_P1, LE_Sweep_P2, TE_Sweep_P1, TE_Sweep_P2, Panel_Break]
        # Variable-span: [LOA, Span, LE_Sweep_P1, LE_Sweep_P2, TE_Sweep_P1, TE_Sweep_P2, Panel_Break]
        default_bounds = {
            'loa': (96, 192),
            'span': (72, 216),
            'le_sweep_p1': (0, 65),
            'le_sweep_p2': (-20, 60),
            'te_sweep_p1': (-60, 60),
            'te_sweep_p2': (-60, 60),
            'panel_break': (0.10, 0.65)
        }

        # Apply geometric constraints to override bounds
        gc = self.geometric_constraints

        # Helper to get bound with fixed/min/max priority
        def get_bound(param, default_min, default_max):
            fixed_key = f'{param}_fixed'
            min_key = f'{param}_min'
            max_key = f'{param}_max'

            if fixed_key in gc and gc[fixed_key] is not None:
                # Fixed value: set both bounds to that value
                return (gc[fixed_key], gc[fixed_key])
            else:
                # Range: use min/max if provided, else defaults
                lb = gc.get(min_key, default_min) if gc.get(min_key) is not None else default_min
                ub = gc.get(max_key, default_max) if gc.get(max_key) is not None else default_max
                return (lb, ub)

        # Get potentially modified bounds for angles
        le_sweep_p1_bounds = get_bound('le_sweep_p1', *default_bounds['le_sweep_p1'])
        le_sweep_p2_bounds = get_bound('le_sweep_p2', *default_bounds['le_sweep_p2'])
        te_sweep_p1_bounds = get_bound('te_sweep_p1', *default_bounds['te_sweep_p1'])
        te_sweep_p2_bounds = get_bound('te_sweep_p2', *default_bounds['te_sweep_p2'])

        # Get potentially modified bounds for panel break
        panel_break_bounds = get_bound('panel_break', *default_bounds['panel_break'])

        # Log any angle constraints applied
        for param, bounds in [('le_sweep_p1', le_sweep_p1_bounds), ('le_sweep_p2', le_sweep_p2_bounds),
                               ('te_sweep_p1', te_sweep_p1_bounds), ('te_sweep_p2', te_sweep_p2_bounds)]:
            default = default_bounds[param]
            if bounds != default:
                if bounds[0] == bounds[1]:
                    logger.info(f"Angle constraint: {param} FIXED at {bounds[0]}°")
                else:
                    logger.info(f"Angle constraint: {param} range [{bounds[0]}, {bounds[1]}]°")

        # Log panel break constraint if modified
        if panel_break_bounds != default_bounds['panel_break']:
            if panel_break_bounds[0] == panel_break_bounds[1]:
                logger.info(f"Panel break constraint: FIXED at {panel_break_bounds[0]}")
            else:
                logger.info(f"Panel break constraint: range [{panel_break_bounds[0]}, {panel_break_bounds[1]}]")

        # Design variable bounds depend on mode
        if fixed_span is not None:
            # Fixed-span mode: 6 design variables (no span)
            # Order: [LOA, LE_Sweep_P1, LE_Sweep_P2, TE_Sweep_P1, TE_Sweep_P2, Panel_Break]
            xl = np.array([
                default_bounds['loa'][0],
                le_sweep_p1_bounds[0],
                le_sweep_p2_bounds[0],
                te_sweep_p1_bounds[0],
                te_sweep_p2_bounds[0],
                panel_break_bounds[0]
            ])
            xu = np.array([
                default_bounds['loa'][1],
                le_sweep_p1_bounds[1],
                le_sweep_p2_bounds[1],
                te_sweep_p1_bounds[1],
                te_sweep_p2_bounds[1],
                panel_break_bounds[1]
            ])
            n_var = 6
            logger.info(f"Using FIXED-SPAN mode (span={fixed_span} inches)")
        else:
            # Variable-span mode: 7 design variables (includes span)
            # Order: [LOA, Span, LE_Sweep_P1, LE_Sweep_P2, TE_Sweep_P1, TE_Sweep_P2, Panel_Break]
            xl = np.array([
                default_bounds['loa'][0],
                default_bounds['span'][0],
                le_sweep_p1_bounds[0],
                le_sweep_p2_bounds[0],
                te_sweep_p1_bounds[0],
                te_sweep_p2_bounds[0],
                panel_break_bounds[0]
            ])
            xu = np.array([
                default_bounds['loa'][1],
                default_bounds['span'][1],
                le_sweep_p1_bounds[1],
                le_sweep_p2_bounds[1],
                te_sweep_p1_bounds[1],
                te_sweep_p2_bounds[1],
                panel_break_bounds[1]
            ])
            n_var = 7
            logger.info("Using VARIABLE-SPAN mode")

        super().__init__(
            n_var=n_var,
            n_obj=5,        # 5 objectives
            n_constr=0,     # No hard constraints (using penalty method)
            xl=xl,
            xu=xu
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objectives for given design parameters

        Args:
            X: Design parameters (n_designs, 6 or 7 depending on mode)
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
        # Only apply if allow_unrealistic_taper is False
        if not self.allow_unrealistic_taper:
            # Extract design variables based on mode
            if self.fixed_span is not None:
                # Fixed-span mode: [LOA, LE_Sweep_P1, LE_Sweep_P2, TE_Sweep_P1, TE_Sweep_P2, Panel_Break]
                loa = X[:, 0]
                span = np.full(len(X), self.fixed_span)  # Fixed span
                le_sweep_p1 = X[:, 1]
                le_sweep_p2 = X[:, 2]
                te_sweep_p1 = X[:, 3]
                te_sweep_p2 = X[:, 4]
                panel_break = X[:, 5]
            else:
                # Variable-span mode: [LOA, Span, LE_Sweep_P1, LE_Sweep_P2, TE_Sweep_P1, TE_Sweep_P2, Panel_Break]
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

            # Chord at panel break: LOA - break_span * (tan(LE) - tan(TE))
            chord_at_break = loa - break_span * (np.tan(le_sweep_p1_rad) - np.tan(te_sweep_p1_rad))

            # Chord at tip: TE_tip - LE_tip
            le_at_tip = np.tan(le_sweep_p1_rad) * break_span + np.tan(le_sweep_p2_rad) * remaining_span
            te_at_tip = loa + np.tan(te_sweep_p1_rad) * break_span + np.tan(te_sweep_p2_rad) * remaining_span
            chord_at_tip = te_at_tip - le_at_tip

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

        # Apply geometric constraints (taper ratios, root chord ratio)
        gc = self.geometric_constraints
        if gc:
            # Extract design variables if not already done
            if self.fixed_span is not None:
                loa = X[:, 0]
                span = np.full(len(X), self.fixed_span)
                le_sweep_p1 = X[:, 1]
                le_sweep_p2 = X[:, 2]
                te_sweep_p1 = X[:, 3]
                te_sweep_p2 = X[:, 4]
                panel_break = X[:, 5]
            else:
                loa = X[:, 0]
                span = X[:, 1]
                le_sweep_p1 = X[:, 2]
                le_sweep_p2 = X[:, 3]
                te_sweep_p1 = X[:, 4]
                te_sweep_p2 = X[:, 5]
                panel_break = X[:, 6]

            # Calculate taper ratios using geometry
            taper_p1, taper_p2 = calculate_taper_ratios(
                loa, span, le_sweep_p1, le_sweep_p2, te_sweep_p1, te_sweep_p2, panel_break
            )

            # Taper ratio constraints for Panel 1
            # Fixed value takes precedence over range
            if gc.get('taper_ratio_p1_fixed') is not None:
                target = gc['taper_ratio_p1_fixed']
                violation = np.abs(taper_p1 - target)
                penalty += violation * 10000  # Very strong penalty for deviation from fixed value
            else:
                if gc.get('min_taper_ratio_p1') is not None:
                    violation = np.maximum(0, gc['min_taper_ratio_p1'] - taper_p1)
                    penalty += violation * 5000  # Strong penalty

                if gc.get('max_taper_ratio_p1') is not None:
                    violation = np.maximum(0, taper_p1 - gc['max_taper_ratio_p1'])
                    penalty += violation * 5000  # Strong penalty

            # Taper ratio constraints for Panel 2
            # Fixed value takes precedence over range
            if gc.get('taper_ratio_p2_fixed') is not None:
                target = gc['taper_ratio_p2_fixed']
                violation = np.abs(taper_p2 - target)
                penalty += violation * 10000  # Very strong penalty for deviation from fixed value
            else:
                if gc.get('min_taper_ratio_p2') is not None:
                    violation = np.maximum(0, gc['min_taper_ratio_p2'] - taper_p2)
                    penalty += violation * 5000  # Strong penalty

                if gc.get('max_taper_ratio_p2') is not None:
                    violation = np.maximum(0, taper_p2 - gc['max_taper_ratio_p2'])
                    penalty += violation * 5000  # Strong penalty

            # Root chord ratio constraints (chord / span)
            root_chord_ratio = loa / span

            # Fixed value takes precedence over range
            if gc.get('root_chord_ratio_fixed') is not None:
                target = gc['root_chord_ratio_fixed']
                violation = np.abs(root_chord_ratio - target)
                penalty += violation * 1000  # Strong penalty for deviation from fixed value
            else:
                if gc.get('min_root_chord_ratio') is not None:
                    violation = np.maximum(0, gc['min_root_chord_ratio'] - root_chord_ratio)
                    penalty += violation * 500

                if gc.get('max_root_chord_ratio') is not None:
                    violation = np.maximum(0, root_chord_ratio - gc['max_root_chord_ratio'])
                    penalty += violation * 500

            # Panel break constraints (fixed values handled via bounds, range handled here)
            if gc.get('panel_break_fixed') is None:  # Only apply range constraints if not fixed
                if gc.get('min_panel_break') is not None:
                    violation = np.maximum(0, gc['min_panel_break'] - panel_break)
                    penalty += violation * 500

                if gc.get('max_panel_break') is not None:
                    violation = np.maximum(0, panel_break - gc['max_panel_break'])
                    penalty += violation * 500

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
    seed: int = 42,
    fixed_span: Optional[float] = None,
    allow_unrealistic_taper: bool = False,
    geometric_constraints: Optional[Dict] = None
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
        fixed_span: If provided, use fixed-span mode with this span value (inches)
        allow_unrealistic_taper: If True, skip taper ratio and bowtie filtering
        geometric_constraints: Optional dict with angle and taper constraints

    Returns:
        Dictionary with optimization results
    """
    logger.info("Initializing NSGA-II optimization...")
    logger.info(f"Population size: {population_size}")
    logger.info(f"Generations: {n_generations}")
    if fixed_span:
        logger.info(f"Fixed-span mode: span={fixed_span} inches")
    if objectives:
        logger.info(f"Custom objectives: {objectives}")
    if allow_unrealistic_taper:
        logger.info("Allowing unrealistic taper ratios and bowtie geometries")
    if geometric_constraints:
        logger.info(f"Geometric constraints: {geometric_constraints}")

    # Create problem
    problem = DroneDesignProblem(
        ensemble_model=ensemble_model,
        feature_engineer=feature_engineer,
        user_constraints=user_constraints,
        objectives=objectives,
        fixed_span=fixed_span,
        allow_unrealistic_taper=allow_unrealistic_taper,
        geometric_constraints=geometric_constraints
    )

    # Configure NSGA-II algorithm
    # Use taper-aware sampling if geometric constraints are present
    if geometric_constraints:
        sampling = TaperAwareSampling(
            geometric_constraints=geometric_constraints,
            fixed_span=fixed_span
        )
    else:
        sampling = FloatRandomSampling()

    algorithm = NSGA2(
        pop_size=population_size,
        sampling=sampling,
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
    pareto_designs = res.X  # Design parameters (n_pareto, 6 or 7)
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
    # Only apply if allow_unrealistic_taper is False
    if not allow_unrealistic_taper:
        # Extract design variables based on mode
        if fixed_span is not None:
            # Fixed-span mode: [LOA, LE_Sweep_P1, LE_Sweep_P2, TE_Sweep_P1, TE_Sweep_P2, Panel_Break]
            loa = pareto_designs[:, 0]
            span = np.full(len(pareto_designs), fixed_span)  # Fixed span
            le_sweep_p1 = pareto_designs[:, 1]
            le_sweep_p2 = pareto_designs[:, 2]
            te_sweep_p1 = pareto_designs[:, 3]
            te_sweep_p2 = pareto_designs[:, 4]
            panel_break = pareto_designs[:, 5]
        else:
            # Variable-span mode: [LOA, Span, LE_Sweep_P1, LE_Sweep_P2, TE_Sweep_P1, TE_Sweep_P2, Panel_Break]
            loa = pareto_designs[:, 0]
            span = pareto_designs[:, 1]
            le_sweep_p1 = pareto_designs[:, 2]
            le_sweep_p2 = pareto_designs[:, 3]
            te_sweep_p1 = pareto_designs[:, 4]
            te_sweep_p2 = pareto_designs[:, 5]
            panel_break = pareto_designs[:, 6]

        # Bowtie detection: chord must be positive at panel break and wingtip
        half_span = span / 2
        break_span = half_span * panel_break
        remaining_span = half_span - break_span

        # Convert sweep angles to radians for tangent calculation
        le_sweep_p1_rad = np.radians(le_sweep_p1)
        le_sweep_p2_rad = np.radians(le_sweep_p2)
        te_sweep_p1_rad = np.radians(te_sweep_p1)
        te_sweep_p2_rad = np.radians(te_sweep_p2)

        # Chord at panel break: LOA - break_span * (tan(LE) - tan(TE))
        chord_at_break = loa - break_span * (np.tan(le_sweep_p1_rad) - np.tan(te_sweep_p1_rad))

        # Chord at wingtip: TE_tip - LE_tip
        le_at_tip = np.tan(le_sweep_p1_rad) * break_span + np.tan(le_sweep_p2_rad) * remaining_span
        te_at_tip = loa + np.tan(te_sweep_p1_rad) * break_span + np.tan(te_sweep_p2_rad) * remaining_span
        chord_at_tip = te_at_tip - le_at_tip

        # Minimum chord requirement (in inches) - must have at least some positive chord
        MIN_CHORD = 2.0  # 2 inches minimum chord

        bowtie_at_break = chord_at_break < MIN_CHORD
        bowtie_at_tip = chord_at_tip < MIN_CHORD
        bowtie_mask = bowtie_at_break | bowtie_at_tip

        feasible_mask &= ~bowtie_mask

        n_bowtie_filtered = np.sum(bowtie_mask)
        if n_bowtie_filtered > 0:
            logger.info(f"Filtered {n_bowtie_filtered} designs with bowtie geometry (chord < {MIN_CHORD}in)")

        # Unrealistic taper ratio filter: use actual computed taper ratios
        # taper_p1 = chord_at_break / chord_root (root chord = LOA)
        # taper_p2 = chord_at_tip / chord_at_break
        # Filter out designs where taper ratio is too extreme (e.g., < 0.1 or > 1.5)
        MIN_TAPER_RATIO = 0.1  # No panel should taper to less than 10% of its root
        MAX_TAPER_RATIO = 1.5  # No panel should expand to more than 150% of its root

        # Only compute taper where chords are positive to avoid division issues
        valid_chords = (loa > MIN_CHORD) & (chord_at_break > MIN_CHORD) & (chord_at_tip > MIN_CHORD)

        taper_p1 = np.where(valid_chords, chord_at_break / loa, 1.0)
        taper_p2 = np.where(valid_chords, chord_at_tip / chord_at_break, 1.0)

        unrealistic_taper_mask = (
            (taper_p1 < MIN_TAPER_RATIO) | (taper_p1 > MAX_TAPER_RATIO) |
            (taper_p2 < MIN_TAPER_RATIO) | (taper_p2 > MAX_TAPER_RATIO)
        )

        feasible_mask &= ~unrealistic_taper_mask

        n_taper_filtered = np.sum(unrealistic_taper_mask)
        if n_taper_filtered > 0:
            logger.info(f"Filtered {n_taper_filtered} designs with unrealistic taper ratios (outside {MIN_TAPER_RATIO}-{MAX_TAPER_RATIO} range)")

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

    # Apply geometric constraints as hard filters
    if geometric_constraints:
        logger.info(f"Applying geometric constraint hard filters: {geometric_constraints}")
        # Extract design variables for geometric calculations
        if fixed_span is not None:
            loa = pareto_designs[:, 0]
            span_arr = np.full(len(pareto_designs), fixed_span)
            le_sweep_p1 = pareto_designs[:, 1]
            le_sweep_p2 = pareto_designs[:, 2]
            te_sweep_p1 = pareto_designs[:, 3]
            te_sweep_p2 = pareto_designs[:, 4]
            panel_break = pareto_designs[:, 5]
        else:
            loa = pareto_designs[:, 0]
            span_arr = pareto_designs[:, 1]
            le_sweep_p1 = pareto_designs[:, 2]
            le_sweep_p2 = pareto_designs[:, 3]
            te_sweep_p1 = pareto_designs[:, 4]
            te_sweep_p2 = pareto_designs[:, 5]
            panel_break = pareto_designs[:, 6]

        # Calculate taper ratios
        taper_p1, taper_p2 = calculate_taper_ratios(
            loa, span_arr, le_sweep_p1, le_sweep_p2, te_sweep_p1, te_sweep_p2, panel_break
        )

        # Taper ratio filters
        gc = geometric_constraints
        logger.info(f"Calculated taper ratios - P1 range: [{np.min(taper_p1):.3f}, {np.max(taper_p1):.3f}], P2 range: [{np.min(taper_p2):.3f}, {np.max(taper_p2):.3f}]")

        n_before = np.sum(feasible_mask)

        # Panel 1 taper ratio - fixed value takes precedence over range
        if gc.get('taper_ratio_p1_fixed') is not None:
            # Allow tolerance of 0.05 for fixed value
            target = gc['taper_ratio_p1_fixed']
            tolerance = 0.05
            feasible_mask &= (np.abs(taper_p1 - target) <= tolerance)
            logger.info(f"After taper_ratio_p1_fixed ({target} ± {tolerance}): {np.sum(feasible_mask)}/{n_before} designs remain")
        else:
            if gc.get('min_taper_ratio_p1') is not None:
                feasible_mask &= (taper_p1 >= gc['min_taper_ratio_p1'])
                logger.info(f"After min_taper_ratio_p1 ({gc['min_taper_ratio_p1']}): {np.sum(feasible_mask)}/{n_before} designs remain")
            if gc.get('max_taper_ratio_p1') is not None:
                feasible_mask &= (taper_p1 <= gc['max_taper_ratio_p1'])
                logger.info(f"After max_taper_ratio_p1 ({gc['max_taper_ratio_p1']}): {np.sum(feasible_mask)}/{n_before} designs remain")

        # Panel 2 taper ratio - fixed value takes precedence over range
        if gc.get('taper_ratio_p2_fixed') is not None:
            # Allow tolerance of 0.05 for fixed value
            target = gc['taper_ratio_p2_fixed']
            tolerance = 0.05
            feasible_mask &= (np.abs(taper_p2 - target) <= tolerance)
            logger.info(f"After taper_ratio_p2_fixed ({target} ± {tolerance}): {np.sum(feasible_mask)}/{n_before} designs remain")
        else:
            if gc.get('min_taper_ratio_p2') is not None:
                feasible_mask &= (taper_p2 >= gc['min_taper_ratio_p2'])
                logger.info(f"After min_taper_ratio_p2 ({gc['min_taper_ratio_p2']}): {np.sum(feasible_mask)}/{n_before} designs remain")
            if gc.get('max_taper_ratio_p2') is not None:
                feasible_mask &= (taper_p2 <= gc['max_taper_ratio_p2'])
                logger.info(f"After max_taper_ratio_p2 ({gc['max_taper_ratio_p2']}): {np.sum(feasible_mask)}/{n_before} designs remain")

        # Root chord ratio filter - fixed value takes precedence over range
        root_chord_ratio = loa / span_arr
        if gc.get('root_chord_ratio_fixed') is not None:
            # Allow tolerance of 0.05 for fixed value
            target = gc['root_chord_ratio_fixed']
            tolerance = 0.05
            feasible_mask &= (np.abs(root_chord_ratio - target) <= tolerance)
            logger.info(f"After root_chord_ratio_fixed ({target} ± {tolerance}): {np.sum(feasible_mask)}/{n_before} designs remain")
        else:
            if gc.get('min_root_chord_ratio') is not None:
                feasible_mask &= (root_chord_ratio >= gc['min_root_chord_ratio'])
            if gc.get('max_root_chord_ratio') is not None:
                feasible_mask &= (root_chord_ratio <= gc['max_root_chord_ratio'])

        # Panel break filter - fixed values are handled via design variable bounds
        # but we also apply range filter here if range constraints exist
        if gc.get('panel_break_fixed') is not None:
            # For fixed panel break, allow tolerance (handled via bounds, but verify here)
            target = gc['panel_break_fixed']
            tolerance = 0.02
            feasible_mask &= (np.abs(panel_break - target) <= tolerance)
            logger.info(f"After panel_break_fixed ({target} ± {tolerance}): {np.sum(feasible_mask)}/{n_before} designs remain")
        else:
            if gc.get('min_panel_break') is not None:
                feasible_mask &= (panel_break >= gc['min_panel_break'])
                logger.info(f"After min_panel_break ({gc['min_panel_break']}): {np.sum(feasible_mask)}/{n_before} designs remain")
            if gc.get('max_panel_break') is not None:
                feasible_mask &= (panel_break <= gc['max_panel_break'])
                logger.info(f"After max_panel_break ({gc['max_panel_break']}): {np.sum(feasible_mask)}/{n_before} designs remain")

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
        'population_size': population_size,
        'fixed_span': fixed_span  # None for variable-span, value for fixed-span mode
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
