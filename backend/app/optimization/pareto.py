"""
Pareto Front Extraction and Analysis
Adapted from generate_report.py for multi-objective optimization results
"""

import numpy as np
from typing import List, Dict, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_pareto_optimal(
    objectives: np.ndarray,
    maximize_mask: np.ndarray = None
) -> np.ndarray:
    """
    Find Pareto-optimal points for N-dimensional objectives

    Args:
        objectives: Objective values (n_points, n_objectives)
        maximize_mask: Boolean mask for maximization (True) vs minimization (False)
                      If None, assumes all minimization

    Returns:
        Boolean array indicating Pareto-optimal points (n_points,)
    """
    n_points = objectives.shape[0]
    n_objectives = objectives.shape[1]

    if maximize_mask is None:
        maximize_mask = np.zeros(n_objectives, dtype=bool)

    # Convert to minimization form (flip sign for maximization objectives)
    obj_min = objectives.copy()
    for i, maximize in enumerate(maximize_mask):
        if maximize:
            obj_min[:, i] = -obj_min[:, i]

    is_optimal = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if not is_optimal[i]:
            continue

        # Check if any other point dominates this one
        for j in range(n_points):
            if i == j:
                continue

            # Point j dominates i if:
            # - j is better or equal in all objectives
            # - j is strictly better in at least one objective
            dominates = True
            strictly_better = False

            for k in range(n_objectives):
                if obj_min[j, k] > obj_min[i, k]:
                    # j is worse in objective k
                    dominates = False
                    break
                if obj_min[j, k] < obj_min[i, k]:
                    # j is strictly better in objective k
                    strictly_better = True

            if dominates and strictly_better:
                is_optimal[i] = False
                break

    return is_optimal


def is_pareto_optimal_2d(
    points: np.ndarray,
    maximize_both: bool = True
) -> np.ndarray:
    """
    Find Pareto-optimal points for 2D optimization (optimized version)

    Args:
        points: 2D points (n_points, 2)
        maximize_both: If True, maximize both objectives. If False, minimize first, maximize second

    Returns:
        Boolean array indicating Pareto-optimal points (n_points,)
    """
    n = len(points)
    is_optimal = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_optimal[i]:
            continue

        for j in range(n):
            if i == j:
                continue

            if maximize_both:
                # Both objectives maximize
                if points[j, 0] >= points[i, 0] and points[j, 1] >= points[i, 1]:
                    if points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1]:
                        is_optimal[i] = False
                        break
            else:
                # First minimize, second maximize
                if points[j, 0] <= points[i, 0] and points[j, 1] >= points[i, 1]:
                    if points[j, 0] < points[i, 0] or points[j, 1] > points[i, 1]:
                        is_optimal[i] = False
                        break

    return is_optimal


def is_pareto_optimal_3d(
    points: np.ndarray,
    directions: Tuple[int, int, int] = (1, 1, -1)
) -> np.ndarray:
    """
    Find Pareto-optimal points for 3D optimization

    Args:
        points: 3D points (n_points, 3)
        directions: Tuple of 1 (maximize) or -1 (minimize) for each objective

    Returns:
        Boolean array indicating Pareto-optimal points (n_points,)
    """
    n = len(points)
    is_optimal = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_optimal[i]:
            continue

        for j in range(n):
            if i == j:
                continue

            dominated = True
            strictly_better = False

            for k in range(3):
                val_i = points[i, k] * directions[k]
                val_j = points[j, k] * directions[k]

                if val_j < val_i:
                    dominated = False
                    break
                if val_j > val_i:
                    strictly_better = True

            if dominated and strictly_better:
                is_optimal[i] = False
                break

    return is_optimal


def extract_pareto_front(
    designs: np.ndarray,
    objectives: np.ndarray,
    maximize_mask: np.ndarray = None,
    return_indices: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract Pareto front from optimization results

    Args:
        designs: Design parameters (n_points, n_vars)
        objectives: Objective values (n_points, n_objectives)
        maximize_mask: Boolean mask for maximization objectives
        return_indices: If True, also return indices of Pareto-optimal points

    Returns:
        pareto_designs: Pareto-optimal design parameters
        pareto_objectives: Pareto-optimal objective values
        (optional) pareto_indices: Indices of Pareto-optimal points
    """
    is_optimal = is_pareto_optimal(objectives, maximize_mask)

    pareto_designs = designs[is_optimal]
    pareto_objectives = objectives[is_optimal]

    if return_indices:
        pareto_indices = np.where(is_optimal)[0]
        return pareto_designs, pareto_objectives, pareto_indices
    else:
        return pareto_designs, pareto_objectives


def find_pareto_frontiers(
    designs: np.ndarray,
    predictions: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, Any]]:
    """
    Find Pareto-optimal designs for different objective combinations

    Args:
        designs: Design parameters (n_points, 7)
        predictions: Dictionary with prediction arrays:
            - range_nm: Range in nautical miles
            - endurance_hr: Endurance in hours
            - mtow_lbm: Max Takeoff Weight
            - cost_usd: Material cost

    Returns:
        Dictionary with Pareto fronts for different combinations:
        - 'range_mtow': Minimize MTOW, maximize Range (2D)
        - 'range_cost': Minimize Cost, maximize Range (2D)
        - 'range_endurance_mtow': Maximize Range/Endurance, minimize MTOW (3D)
        - 'all_objectives': All 4 objectives (4D)
    """
    range_nm = predictions['range_nm']
    endurance_hr = predictions['endurance_hr']
    mtow_lbm = predictions['mtow_lbm']
    cost_usd = predictions['cost_usd']

    n_points = len(designs)
    results = {}

    # 2D: Range vs MTOW (minimize MTOW, maximize Range)
    points_2d_mtow = np.column_stack([mtow_lbm, range_nm])
    optimal_2d_mtow = is_pareto_optimal_2d(points_2d_mtow, maximize_both=False)
    results['range_mtow'] = {
        'indices': np.where(optimal_2d_mtow)[0],
        'designs': designs[optimal_2d_mtow],
        'objectives': {
            'range_nm': range_nm[optimal_2d_mtow],
            'mtow_lbm': mtow_lbm[optimal_2d_mtow]
        },
        'n_pareto': int(np.sum(optimal_2d_mtow))
    }

    # 2D: Range vs Cost (minimize Cost, maximize Range)
    points_2d_cost = np.column_stack([cost_usd, range_nm])
    optimal_2d_cost = is_pareto_optimal_2d(points_2d_cost, maximize_both=False)
    results['range_cost'] = {
        'indices': np.where(optimal_2d_cost)[0],
        'designs': designs[optimal_2d_cost],
        'objectives': {
            'range_nm': range_nm[optimal_2d_cost],
            'cost_usd': cost_usd[optimal_2d_cost]
        },
        'n_pareto': int(np.sum(optimal_2d_cost))
    }

    # 3D: Range, Endurance, MTOW (maximize Range/Endurance, minimize MTOW)
    points_3d = np.column_stack([range_nm, endurance_hr, mtow_lbm])
    optimal_3d = is_pareto_optimal_3d(points_3d, directions=(1, 1, -1))
    results['range_endurance_mtow'] = {
        'indices': np.where(optimal_3d)[0],
        'designs': designs[optimal_3d],
        'objectives': {
            'range_nm': range_nm[optimal_3d],
            'endurance_hr': endurance_hr[optimal_3d],
            'mtow_lbm': mtow_lbm[optimal_3d]
        },
        'n_pareto': int(np.sum(optimal_3d))
    }

    # 4D: All objectives (maximize Range/Endurance, minimize MTOW/Cost)
    objectives_4d = np.column_stack([range_nm, endurance_hr, mtow_lbm, cost_usd])
    maximize_mask = np.array([True, True, False, False])  # max range/endurance, min mtow/cost
    optimal_4d = is_pareto_optimal(objectives_4d, maximize_mask)
    results['all_objectives'] = {
        'indices': np.where(optimal_4d)[0],
        'designs': designs[optimal_4d],
        'objectives': {
            'range_nm': range_nm[optimal_4d],
            'endurance_hr': endurance_hr[optimal_4d],
            'mtow_lbm': mtow_lbm[optimal_4d],
            'cost_usd': cost_usd[optimal_4d]
        },
        'n_pareto': int(np.sum(optimal_4d))
    }

    logger.info(f"Pareto front extraction complete:")
    logger.info(f"  Range-MTOW (2D): {results['range_mtow']['n_pareto']} points")
    logger.info(f"  Range-Cost (2D): {results['range_cost']['n_pareto']} points")
    logger.info(f"  Range-Endurance-MTOW (3D): {results['range_endurance_mtow']['n_pareto']} points")
    logger.info(f"  All objectives (4D): {results['all_objectives']['n_pareto']} points")

    return results


def compute_crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """
    Compute crowding distance for Pareto front points (diversity metric)

    Args:
        objectives: Objective values (n_points, n_objectives)

    Returns:
        Crowding distances (n_points,)
    """
    n_points = objectives.shape[0]
    n_objectives = objectives.shape[1]

    if n_points <= 2:
        # Boundary points get infinite distance
        return np.full(n_points, np.inf)

    crowding = np.zeros(n_points)

    for m in range(n_objectives):
        # Sort by objective m
        sorted_indices = np.argsort(objectives[:, m])

        # Boundary points get infinite distance
        crowding[sorted_indices[0]] = np.inf
        crowding[sorted_indices[-1]] = np.inf

        # Normalize
        obj_range = objectives[sorted_indices[-1], m] - objectives[sorted_indices[0], m]

        if obj_range == 0:
            continue

        # Compute distances
        for i in range(1, n_points - 1):
            crowding[sorted_indices[i]] += (
                objectives[sorted_indices[i + 1], m] - objectives[sorted_indices[i - 1], m]
            ) / obj_range

    return crowding


def select_diverse_subset(
    designs: np.ndarray,
    objectives: np.ndarray,
    n_select: int,
    method: str = 'crowding'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select diverse subset of designs from Pareto front

    Args:
        designs: Design parameters (n_points, n_vars)
        objectives: Objective values (n_points, n_objectives)
        n_select: Number of designs to select
        method: Selection method ('crowding' or 'uniform')

    Returns:
        selected_designs: Selected design parameters
        selected_objectives: Selected objective values
        selected_indices: Indices of selected designs
    """
    n_points = designs.shape[0]

    if n_select >= n_points:
        return designs, objectives, np.arange(n_points)

    if method == 'crowding':
        # Select based on crowding distance
        crowding = compute_crowding_distance(objectives)
        selected_indices = np.argsort(crowding)[-n_select:]
    elif method == 'uniform':
        # Uniform sampling
        selected_indices = np.linspace(0, n_points - 1, n_select, dtype=int)
    else:
        raise ValueError(f"Unknown selection method: {method}")

    return designs[selected_indices], objectives[selected_indices], selected_indices


if __name__ == "__main__":
    # Test Pareto extraction
    print("Testing Pareto front extraction...")

    # Create test data (4 objectives: maximize range/endurance, minimize mtow/cost)
    np.random.seed(42)
    n_test = 100

    # Generate test designs
    designs_test = np.random.rand(n_test, 7) * 100 + 100

    # Generate test objectives with tradeoffs
    range_nm = np.random.rand(n_test) * 2000 + 1000
    endurance_hr = range_nm / 200 + np.random.rand(n_test) * 2  # Correlated with range
    mtow_lbm = 2000 + range_nm * 0.5 + np.random.rand(n_test) * 500  # Increases with range
    cost_usd = mtow_lbm * 10 + np.random.rand(n_test) * 5000  # Correlated with MTOW

    predictions_test = {
        'range_nm': range_nm,
        'endurance_hr': endurance_hr,
        'mtow_lbm': mtow_lbm,
        'cost_usd': cost_usd
    }

    # Find Pareto frontiers
    pareto_results = find_pareto_frontiers(designs_test, predictions_test)

    print(f"\nPareto Front Results:")
    print(f"  Total designs: {n_test}")
    for key, result in pareto_results.items():
        print(f"  {key}: {result['n_pareto']} Pareto-optimal points")

    # Test crowding distance
    obj_4d = np.column_stack([range_nm, endurance_hr, mtow_lbm, cost_usd])
    pareto_4d = pareto_results['all_objectives']
    crowding = compute_crowding_distance(pareto_4d['objectives']['range_nm'].reshape(-1, 1))

    print(f"\nCrowding distances: min={crowding[np.isfinite(crowding)].min():.4f}, "
          f"max={crowding[np.isfinite(crowding)].max():.4f}, "
          f"infinite={np.sum(np.isinf(crowding))}")

    # Test diverse subset selection
    n_select = 20
    designs_4d = pareto_4d['designs']
    objectives_4d = np.column_stack([
        pareto_4d['objectives']['range_nm'],
        pareto_4d['objectives']['endurance_hr'],
        pareto_4d['objectives']['mtow_lbm'],
        pareto_4d['objectives']['cost_usd']
    ])

    selected_designs, selected_obj, selected_idx = select_diverse_subset(
        designs_4d, objectives_4d, n_select, method='crowding'
    )

    print(f"\nDiverse subset selection: {len(selected_idx)} designs selected from {len(designs_4d)}")

    print("\nPareto extraction test complete!")
