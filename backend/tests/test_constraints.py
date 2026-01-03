"""
Tests for constraint handling and validation
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(backend_dir))

from app.optimization.constraints import ConstraintHandler, validate_constraints


class TestConstraintValidation:
    """Tests for constraint validation"""

    def test_validate_positive_constraints(self):
        """Test that valid constraints pass validation"""
        constraints = {
            'min_range_nm': 1500,
            'max_cost_usd': 35000,
            'max_mtow_lbm': 3000,
            'min_endurance_hr': 10
        }
        is_valid, errors, warnings = validate_constraints(constraints)
        assert is_valid
        assert len(errors) == 0

    def test_validate_rejects_zero_constraints(self):
        """Test that zero constraint values are rejected"""
        constraints = {
            'min_range_nm': 0,
            'max_cost_usd': 35000
        }
        is_valid, errors, warnings = validate_constraints(constraints)
        assert not is_valid
        assert any('min_range_nm must be positive' in error for error in errors)

    def test_validate_rejects_negative_constraints(self):
        """Test that negative constraint values are rejected"""
        constraints = {
            'min_range_nm': -100,
            'max_cost_usd': 35000
        }
        is_valid, errors, warnings = validate_constraints(constraints)
        assert not is_valid
        assert any('min_range_nm must be positive' in error for error in errors)


class TestConstraintHandler:
    """Tests for ConstraintHandler class"""

    def test_find_nearest_feasible_zero_constraint(self):
        """Test that zero constraint values don't cause division by zero"""
        # Create handler with zero max constraint (bypassing validation)
        # Using max_cost_usd=0 ensures all designs violate it
        handler = ConstraintHandler({'max_cost_usd': 0}, validate=False)
        
        # Create test data where all designs violate the zero constraint
        n_designs = 10
        designs = np.random.rand(n_designs, 7) * 100 + 100
        predictions = {
            'range_nm': np.array([2000.0] * n_designs),
            'endurance_hr': np.array([10.0] * n_designs),
            'mtow_lbm': np.array([2000.0] * n_designs),
            'cost_usd': np.array([25000.0] * n_designs)  # All violate max_cost_usd=0
        }
        
        # Should not raise exception or produce NaN/inf
        result = handler.find_nearest_feasible(designs, predictions, n_nearest=5)
        
        # Verify no NaN or inf values in violation scores
        assert 'violation_scores' in result
        violation_scores = result['violation_scores']
        assert not np.any(np.isnan(violation_scores)), "Violation scores contain NaN"
        assert not np.any(np.isinf(violation_scores)), "Violation scores contain inf"
        
        # Verify result structure
        assert result['feasible'] == False
        assert len(result['designs']) == 5
        assert 'violations' in result

    def test_find_nearest_feasible_negative_constraint(self):
        """Test that negative constraint values don't cause division by zero"""
        handler = ConstraintHandler({'max_cost_usd': -1000}, validate=False)
        
        n_designs = 10
        designs = np.random.rand(n_designs, 7) * 100 + 100
        predictions = {
            'range_nm': np.array([2000.0] * n_designs),
            'endurance_hr': np.array([10.0] * n_designs),
            'mtow_lbm': np.array([2000.0] * n_designs),
            'cost_usd': np.array([50000.0] * n_designs)  # All violate negative constraint
        }
        
        result = handler.find_nearest_feasible(designs, predictions, n_nearest=5)
        
        # Verify no NaN or inf values
        violation_scores = result['violation_scores']
        assert not np.any(np.isnan(violation_scores))
        assert not np.any(np.isinf(violation_scores))

    def test_find_nearest_feasible_mixed_constraints(self):
        """Test handling when some constraints are zero and others are valid"""
        handler = ConstraintHandler({
            'min_range_nm': 0,  # Zero constraint
            'max_cost_usd': 35000  # Valid constraint
        }, validate=False)
        
        n_designs = 10
        designs = np.random.rand(n_designs, 7) * 100 + 100
        predictions = {
            'range_nm': np.array([100.0] * n_designs),
            'endurance_hr': np.array([10.0] * n_designs),
            'mtow_lbm': np.array([2000.0] * n_designs),
            'cost_usd': np.array([50000.0] * n_designs)  # Violates max_cost
        }
        
        result = handler.find_nearest_feasible(designs, predictions, n_nearest=5)
        
        # Should handle mixed constraints without errors
        violation_scores = result['violation_scores']
        assert not np.any(np.isnan(violation_scores))
        assert not np.any(np.isinf(violation_scores))

    def test_validation_on_init(self):
        """Test that validation can be enabled on initialization"""
        # Should raise ValueError when validate=True and constraints are invalid
        with pytest.raises(ValueError, match="Invalid constraints"):
            ConstraintHandler({'min_range_nm': 0}, validate=True)
        
        # Should not raise when constraints are valid
        handler = ConstraintHandler({
            'min_range_nm': 1500,
            'max_cost_usd': 35000
        }, validate=True)
        assert handler.constraints['min_range_nm'] == 1500

    def test_find_nearest_feasible_with_valid_constraints(self):
        """Test that normal operation works with valid constraints"""
        handler = ConstraintHandler({
            'min_range_nm': 2000,
            'max_cost_usd': 30000
        })
        
        n_designs = 10
        designs = np.random.rand(n_designs, 7) * 100 + 100
        predictions = {
            'range_nm': np.array([1000.0] * n_designs),  # All violate min_range
            'endurance_hr': np.array([10.0] * n_designs),
            'mtow_lbm': np.array([2000.0] * n_designs),
            'cost_usd': np.array([25000.0] * n_designs)
        }
        
        result = handler.find_nearest_feasible(designs, predictions, n_nearest=5)
        
        # Should work normally with valid constraints
        assert 'violation_scores' in result
        violation_scores = result['violation_scores']
        assert not np.any(np.isnan(violation_scores))
        assert not np.any(np.isinf(violation_scores))
        assert len(result['designs']) == 5

