"""
Multi-Objective Optimization Engine for Drone Design
Uses NSGA-II algorithm to find Pareto-optimal designs
"""

from app.optimization.nsga import (
    DroneDesignProblem,
    run_nsga2_optimization,
    optimize_with_constraints
)
from app.optimization.objectives import (
    DroneObjectives,
    weighted_sum_objective,
    normalize_objectives,
    compute_hypervolume_indicator
)
from app.optimization.pareto import (
    is_pareto_optimal,
    is_pareto_optimal_2d,
    is_pareto_optimal_3d,
    extract_pareto_front,
    find_pareto_frontiers,
    compute_crowding_distance,
    select_diverse_subset
)
from app.optimization.constraints import (
    ConstraintHandler,
    validate_constraints
)

__all__ = [
    # NSGA-II optimization
    "DroneDesignProblem",
    "run_nsga2_optimization",
    "optimize_with_constraints",

    # Objectives
    "DroneObjectives",
    "weighted_sum_objective",
    "normalize_objectives",
    "compute_hypervolume_indicator",

    # Pareto extraction
    "is_pareto_optimal",
    "is_pareto_optimal_2d",
    "is_pareto_optimal_3d",
    "extract_pareto_front",
    "find_pareto_frontiers",
    "compute_crowding_distance",
    "select_diverse_subset",

    # Constraints
    "ConstraintHandler",
    "validate_constraints"
]
