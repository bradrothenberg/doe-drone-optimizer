"""
Constraint Handling and Relaxation for Drone Design Optimization
Manages user constraints and provides feasible alternatives when targets are infeasible
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConstraintHandler:
    """
    Handles constraint checking and relaxation for optimization
    """

    def __init__(
        self,
        constraints: Optional[Dict[str, float]] = None,
        tolerance: float = 0.01,
        validate: bool = True
    ):
        """
        Initialize constraint handler

        Args:
            constraints: Dictionary of user constraints:
                - min_range_nm: Minimum range (nm)
                - max_cost_usd: Maximum cost ($)
                - max_mtow_lbm: Maximum MTOW (lbm)
                - min_endurance_hr: Minimum endurance (hr)
            tolerance: Relative tolerance for constraint satisfaction (e.g., 0.01 = 1%)
            validate: If True (default), validate constraints on initialization.
                      Raises ValueError if constraints are invalid (e.g., non-positive values).
                      Set to False to skip validation (not recommended).
        """
        self.constraints = constraints or {}
        self.tolerance = tolerance
        self.original_constraints = constraints.copy() if constraints else {}
        
        # ROOT CAUSE FIX: Validate constraints at construction time to prevent invalid state
        if validate and self.constraints:
            is_valid, errors, warnings = validate_constraints(self.constraints)
            if not is_valid:
                raise ValueError(f"Invalid constraints: {'; '.join(errors)}")
            for warning in warnings:
                logger.warning(f"Constraint warning: {warning}")

    def check_feasibility(
        self,
        predictions: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Check which designs satisfy all constraints

        Args:
            predictions: Dictionary with prediction arrays:
                - range_nm
                - endurance_hr
                - mtow_lbm
                - cost_usd

        Returns:
            feasible_mask: Boolean array indicating feasible designs
            violations: Dictionary with violation amounts for each constraint
        """
        n_designs = len(predictions['range_nm'])
        feasible_mask = np.ones(n_designs, dtype=bool)
        violations = {}

        if 'min_range_nm' in self.constraints:
            min_range = self.constraints['min_range_nm']
            range_violation = min_range - predictions['range_nm']
            violations['min_range_nm'] = np.maximum(0, range_violation)
            feasible_mask &= (predictions['range_nm'] >= min_range * (1 - self.tolerance))

        if 'max_cost_usd' in self.constraints:
            max_cost = self.constraints['max_cost_usd']
            cost_violation = predictions['cost_usd'] - max_cost
            violations['max_cost_usd'] = np.maximum(0, cost_violation)
            feasible_mask &= (predictions['cost_usd'] <= max_cost * (1 + self.tolerance))

        if 'max_mtow_lbm' in self.constraints:
            max_mtow = self.constraints['max_mtow_lbm']
            mtow_violation = predictions['mtow_lbm'] - max_mtow
            violations['max_mtow_lbm'] = np.maximum(0, mtow_violation)
            feasible_mask &= (predictions['mtow_lbm'] <= max_mtow * (1 + self.tolerance))

        if 'min_endurance_hr' in self.constraints:
            min_endurance = self.constraints['min_endurance_hr']
            endurance_violation = min_endurance - predictions['endurance_hr']
            violations['min_endurance_hr'] = np.maximum(0, endurance_violation)
            feasible_mask &= (predictions['endurance_hr'] >= min_endurance * (1 - self.tolerance))

        return feasible_mask, violations

    def compute_penalty(
        self,
        predictions: Dict[str, np.ndarray],
        penalty_weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Compute penalty values for constraint violations (for penalty-based optimization)

        Args:
            predictions: Dictionary with prediction arrays
            penalty_weights: Dictionary with penalty weights for each constraint
                            If None, uses default weights

        Returns:
            Penalty array (n_designs,)
        """
        if penalty_weights is None:
            penalty_weights = {
                'min_range_nm': 1000.0,
                'max_cost_usd': 0.01,
                'max_mtow_lbm': 1.0,
                'min_endurance_hr': 100.0
            }

        n_designs = len(predictions['range_nm'])
        penalty = np.zeros(n_designs)

        if 'min_range_nm' in self.constraints:
            min_range = self.constraints['min_range_nm']
            violation = np.maximum(0, min_range - predictions['range_nm'])
            penalty += violation * penalty_weights['min_range_nm']

        if 'max_cost_usd' in self.constraints:
            max_cost = self.constraints['max_cost_usd']
            violation = np.maximum(0, predictions['cost_usd'] - max_cost)
            penalty += violation * penalty_weights['max_cost_usd']

        if 'max_mtow_lbm' in self.constraints:
            max_mtow = self.constraints['max_mtow_lbm']
            violation = np.maximum(0, predictions['mtow_lbm'] - max_mtow)
            penalty += violation * penalty_weights['max_mtow_lbm']

        if 'min_endurance_hr' in self.constraints:
            min_endurance = self.constraints['min_endurance_hr']
            violation = np.maximum(0, min_endurance - predictions['endurance_hr'])
            penalty += violation * penalty_weights['min_endurance_hr']

        return penalty

    def relax_constraints(
        self,
        relaxation_strategy: str = 'balanced'
    ) -> Dict[str, Any]:
        """
        Relax constraints to increase feasibility

        Args:
            relaxation_strategy: Strategy for relaxation:
                - 'balanced': Relax all constraints equally
                - 'cost_priority': Prioritize cost relaxation
                - 'performance_priority': Prioritize range/endurance relaxation

        Returns:
            Dictionary with:
                - relaxed_constraints: New relaxed constraints
                - relaxation_factors: How much each constraint was relaxed
                - relaxation_description: Human-readable description
        """
        relaxed = self.constraints.copy()
        relaxation_factors = {}
        descriptions = []

        if relaxation_strategy == 'balanced':
            # Relax all constraints by 10%
            if 'min_range_nm' in relaxed:
                original = relaxed['min_range_nm']
                relaxed['min_range_nm'] = original * 0.90
                relaxation_factors['min_range_nm'] = 0.90
                descriptions.append(f"Minimum range reduced from {original:.0f} to {relaxed['min_range_nm']:.0f} nm (-10%)")

            if 'max_cost_usd' in relaxed:
                original = relaxed['max_cost_usd']
                relaxed['max_cost_usd'] = original * 1.10
                relaxation_factors['max_cost_usd'] = 1.10
                descriptions.append(f"Maximum cost increased from ${original:.0f} to ${relaxed['max_cost_usd']:.0f} (+10%)")

            if 'max_mtow_lbm' in relaxed:
                original = relaxed['max_mtow_lbm']
                relaxed['max_mtow_lbm'] = original * 1.10
                relaxation_factors['max_mtow_lbm'] = 1.10
                descriptions.append(f"Maximum MTOW increased from {original:.0f} to {relaxed['max_mtow_lbm']:.0f} lbm (+10%)")

            if 'min_endurance_hr' in relaxed:
                original = relaxed['min_endurance_hr']
                relaxed['min_endurance_hr'] = original * 0.90
                relaxation_factors['min_endurance_hr'] = 0.90
                descriptions.append(f"Minimum endurance reduced from {original:.1f} to {relaxed['min_endurance_hr']:.1f} hr (-10%)")

        elif relaxation_strategy == 'cost_priority':
            # Relax cost more aggressively
            if 'max_cost_usd' in relaxed:
                original = relaxed['max_cost_usd']
                relaxed['max_cost_usd'] = original * 1.20
                relaxation_factors['max_cost_usd'] = 1.20
                descriptions.append(f"Maximum cost increased from ${original:.0f} to ${relaxed['max_cost_usd']:.0f} (+20%)")

            if 'max_mtow_lbm' in relaxed:
                original = relaxed['max_mtow_lbm']
                relaxed['max_mtow_lbm'] = original * 1.15
                relaxation_factors['max_mtow_lbm'] = 1.15
                descriptions.append(f"Maximum MTOW increased from {original:.0f} to {relaxed['max_mtow_lbm']:.0f} lbm (+15%)")

            # Relax performance slightly
            if 'min_range_nm' in relaxed:
                original = relaxed['min_range_nm']
                relaxed['min_range_nm'] = original * 0.95
                relaxation_factors['min_range_nm'] = 0.95
                descriptions.append(f"Minimum range reduced from {original:.0f} to {relaxed['min_range_nm']:.0f} nm (-5%)")

            if 'min_endurance_hr' in relaxed:
                original = relaxed['min_endurance_hr']
                relaxed['min_endurance_hr'] = original * 0.95
                relaxation_factors['min_endurance_hr'] = 0.95
                descriptions.append(f"Minimum endurance reduced from {original:.1f} to {relaxed['min_endurance_hr']:.1f} hr (-5%)")

        elif relaxation_strategy == 'performance_priority':
            # Relax performance more aggressively
            if 'min_range_nm' in relaxed:
                original = relaxed['min_range_nm']
                relaxed['min_range_nm'] = original * 0.85
                relaxation_factors['min_range_nm'] = 0.85
                descriptions.append(f"Minimum range reduced from {original:.0f} to {relaxed['min_range_nm']:.0f} nm (-15%)")

            if 'min_endurance_hr' in relaxed:
                original = relaxed['min_endurance_hr']
                relaxed['min_endurance_hr'] = original * 0.85
                relaxation_factors['min_endurance_hr'] = 0.85
                descriptions.append(f"Minimum endurance reduced from {original:.1f} to {relaxed['min_endurance_hr']:.1f} hr (-15%)")

            # Relax cost/weight slightly
            if 'max_cost_usd' in relaxed:
                original = relaxed['max_cost_usd']
                relaxed['max_cost_usd'] = original * 1.10
                relaxation_factors['max_cost_usd'] = 1.10
                descriptions.append(f"Maximum cost increased from ${original:.0f} to ${relaxed['max_cost_usd']:.0f} (+10%)")

            if 'max_mtow_lbm' in relaxed:
                original = relaxed['max_mtow_lbm']
                relaxed['max_mtow_lbm'] = original * 1.10
                relaxation_factors['max_mtow_lbm'] = 1.10
                descriptions.append(f"Maximum MTOW increased from {original:.0f} to {relaxed['max_mtow_lbm']:.0f} lbm (+10%)")

        else:
            raise ValueError(f"Unknown relaxation strategy: {relaxation_strategy}")

        # Update internal constraints
        self.constraints = relaxed

        return {
            'relaxed_constraints': relaxed,
            'relaxation_factors': relaxation_factors,
            'relaxation_description': descriptions,
            'strategy': relaxation_strategy
        }

    def find_nearest_feasible(
        self,
        designs: np.ndarray,
        predictions: Dict[str, np.ndarray],
        n_nearest: int = 5
    ) -> Dict[str, Any]:
        """
        Find designs nearest to satisfying constraints

        Args:
            designs: Design parameters (n_designs, 7)
            predictions: Prediction dictionary
            n_nearest: Number of nearest designs to return

        Returns:
            Dictionary with nearest designs and their constraint violations
        """
        feasible_mask, violations = self.check_feasibility(predictions)

        if np.sum(feasible_mask) > 0:
            # Some designs are feasible, return those
            return {
                'feasible': True,
                'designs': designs[feasible_mask],
                'predictions': {k: v[feasible_mask] for k, v in predictions.items()},
                'n_feasible': int(np.sum(feasible_mask))
            }

        # No feasible designs, find nearest
        # Compute total violation score
        violation_scores = np.zeros(len(designs))

        for constraint_name, violation_array in violations.items():
            # Normalize violation by constraint value
            target = self.constraints[constraint_name]
            
            # DEFENSIVE FIX: Guard against division by zero (defense in depth)
            # This protects against edge cases even if validation was skipped or
            # constraints were modified after construction
            if target <= 0:
                logger.warning(
                    f"Constraint {constraint_name} has non-positive value ({target}). "
                    "Skipping normalization to avoid division by zero."
                )
                # Use raw violation value without normalization
                violation_scores += violation_array
            else:
                violation_scores += violation_array / target

        # Select designs with smallest violations
        nearest_indices = np.argsort(violation_scores)[:n_nearest]

        nearest_violations = {}
        for constraint_name, violation_array in violations.items():
            nearest_violations[constraint_name] = violation_array[nearest_indices]

        return {
            'feasible': False,
            'designs': designs[nearest_indices],
            'predictions': {k: v[nearest_indices] for k, v in predictions.items()},
            'violations': nearest_violations,
            'violation_scores': violation_scores[nearest_indices],
            'n_nearest': len(nearest_indices)
        }

    def reset_constraints(self):
        """Reset constraints to original values"""
        self.constraints = self.original_constraints.copy()


def validate_constraints(constraints: Dict[str, float]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate user-provided constraints

    Args:
        constraints: Constraint dictionary

    Returns:
        is_valid: True if all constraints are valid
        errors: List of error messages
        warnings: List of warning messages (don't affect validity)
    """
    errors = []
    warnings = []

    # Check constraint names
    valid_names = {'min_range_nm', 'max_cost_usd', 'max_mtow_lbm', 'min_endurance_hr'}
    for name in constraints.keys():
        if name not in valid_names:
            errors.append(f"Unknown constraint: {name}")

    # Check constraint values
    if 'min_range_nm' in constraints:
        if constraints['min_range_nm'] <= 0:
            errors.append("min_range_nm must be positive")
        if constraints['min_range_nm'] > 6000:
            errors.append("min_range_nm exceeds typical dataset range (0-6000 nm)")

    if 'max_cost_usd' in constraints:
        if constraints['max_cost_usd'] <= 0:
            errors.append("max_cost_usd must be positive")
        if constraints['max_cost_usd'] > 100000:
            errors.append("max_cost_usd exceeds typical dataset range ($0-$100k)")

    if 'max_mtow_lbm' in constraints:
        if constraints['max_mtow_lbm'] <= 0:
            errors.append("max_mtow_lbm must be positive")
        if constraints['max_mtow_lbm'] > 10000:
            errors.append("max_mtow_lbm exceeds typical dataset range (0-10k lbm)")

    if 'min_endurance_hr' in constraints:
        if constraints['min_endurance_hr'] <= 0:
            errors.append("min_endurance_hr must be positive")
        if constraints['min_endurance_hr'] > 40:
            errors.append("min_endurance_hr exceeds typical dataset range (0-40 hr)")

    # Check for conflicting constraints (warnings only, don't fail validation)
    if 'min_range_nm' in constraints and 'max_cost_usd' in constraints:
        if constraints['min_range_nm'] > 3000 and constraints['max_cost_usd'] < 20000:
            warnings.append("High range requirement with low cost limit may be infeasible")

    return len(errors) == 0, errors, warnings


if __name__ == "__main__":
    # Test constraint handling
    print("Testing constraint handling...")

    # Create test predictions
    np.random.seed(42)
    n_test = 50

    predictions_test = {
        'range_nm': np.random.rand(n_test) * 2000 + 1000,
        'endurance_hr': np.random.rand(n_test) * 10 + 5,
        'mtow_lbm': np.random.rand(n_test) * 2000 + 2000,
        'cost_usd': np.random.rand(n_test) * 30000 + 20000
    }

    designs_test = np.random.rand(n_test, 7) * 100 + 100

    # Define constraints
    constraints_test = {
        'min_range_nm': 2000,
        'max_cost_usd': 35000,
        'max_mtow_lbm': 3500,
        'min_endurance_hr': 10
    }

    # Validate constraints
    is_valid, errors, warnings = validate_constraints(constraints_test)
    print(f"\nConstraint validation: {'PASSED' if is_valid else 'FAILED'}")
    if errors:
        print("  Errors:")
        for error in errors:
            print(f"    - {error}")
    if warnings:
        print("  Warnings:")
        for warning in warnings:
            print(f"    - {warning}")

    # Create handler
    handler = ConstraintHandler(constraints_test, tolerance=0.01)

    # Check feasibility
    feasible_mask, violations = handler.check_feasibility(predictions_test)
    n_feasible = np.sum(feasible_mask)

    print(f"\nFeasibility check:")
    print(f"  Feasible designs: {n_feasible}/{n_test} ({n_feasible/n_test*100:.1f}%)")

    for constraint_name, violation_array in violations.items():
        max_violation = np.max(violation_array)
        n_violated = np.sum(violation_array > 0)
        print(f"  {constraint_name}: {n_violated} violations (max: {max_violation:.1f})")

    # Compute penalties
    penalties = handler.compute_penalty(predictions_test)
    print(f"\nPenalty statistics:")
    print(f"  Mean: {np.mean(penalties):.1f}")
    print(f"  Max: {np.max(penalties):.1f}")
    print(f"  Designs with penalty > 0: {np.sum(penalties > 0)}")

    # Test relaxation
    print(f"\nTesting constraint relaxation (balanced strategy)...")
    relaxation_result = handler.relax_constraints('balanced')

    print(f"Relaxation applied:")
    for desc in relaxation_result['relaxation_description']:
        print(f"  - {desc}")

    # Check feasibility after relaxation
    feasible_mask_relaxed, _ = handler.check_feasibility(predictions_test)
    n_feasible_relaxed = np.sum(feasible_mask_relaxed)
    print(f"\nFeasibility after relaxation: {n_feasible_relaxed}/{n_test} ({n_feasible_relaxed/n_test*100:.1f}%)")

    # Reset and test nearest feasible
    handler.reset_constraints()
    nearest_result = handler.find_nearest_feasible(designs_test, predictions_test, n_nearest=5)

    print(f"\nNearest feasible designs:")
    print(f"  Found {nearest_result.get('n_feasible', nearest_result.get('n_nearest', 0))} designs")
    print(f"  Feasible: {nearest_result['feasible']}")

    print("\nConstraint handling test complete!")
