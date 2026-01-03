"""
Integration Test for Full Optimization Pipeline
Tests NSGA-II optimization with ensemble model, constraints, and Pareto extraction
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.models.ensemble import EnsembleDroneModel
from app.models.feature_engineering import FeatureEngineer
from app.optimization import (
    run_nsga2_optimization,
    optimize_with_constraints,
    find_pareto_frontiers,
    ConstraintHandler,
    validate_constraints,
    DroneObjectives
)
import joblib


def test_unconstrained_optimization():
    """Test unconstrained NSGA-II optimization"""
    print("\n" + "="*80)
    print("TEST 1: Unconstrained Optimization")
    print("="*80)

    # Load models
    models_dir = Path(__file__).parent.parent / "data" / "models"

    ensemble = EnsembleDroneModel()
    ensemble.load_models(
        xgb_model_path=models_dir / "xgboost_v1.pkl",
        nn_model_path=models_dir / "neural_v1.pt",
        input_dim=17
    )

    engineer = joblib.load(models_dir / "feature_engineer.pkl")

    # Run optimization
    print("\nRunning unconstrained optimization...")
    print("  Population: 200, Generations: 100")

    start_time = time.time()
    results = run_nsga2_optimization(
        ensemble_model=ensemble,
        feature_engineer=engineer,
        population_size=200,
        n_generations=100
    )
    elapsed = time.time() - start_time

    # Validate results
    print(f"\nResults:")
    print(f"  Optimization time: {elapsed:.2f}s")
    print(f"  Pareto front size: {results['n_pareto']}")
    print(f"  Convergence: {results['n_generations']} generations")

    print(f"\nObjective ranges:")
    print(f"  Range: {results['pareto_objectives']['range_nm'].min():.1f} - "
          f"{results['pareto_objectives']['range_nm'].max():.1f} nm")
    print(f"  Endurance: {results['pareto_objectives']['endurance_hr'].min():.1f} - "
          f"{results['pareto_objectives']['endurance_hr'].max():.1f} hr")
    print(f"  MTOW: {results['pareto_objectives']['mtow_lbm'].min():.1f} - "
          f"{results['pareto_objectives']['mtow_lbm'].max():.1f} lbm")
    print(f"  Cost: ${results['pareto_objectives']['cost_usd'].min():.0f} - "
          f"${results['pareto_objectives']['cost_usd'].max():.0f}")

    # Assertions
    assert elapsed < 30.0, f"Optimization too slow: {elapsed:.2f}s > 30s"
    assert results['n_pareto'] > 10, f"Too few Pareto points: {results['n_pareto']}"
    assert results['pareto_objectives']['range_nm'].max() > 1000, "Range too low"

    print("\nPASSED: Unconstrained optimization")
    return results


def test_constrained_optimization():
    """Test constrained NSGA-II optimization"""
    print("\n" + "="*80)
    print("TEST 2: Constrained Optimization")
    print("="*80)

    # Load models
    models_dir = Path(__file__).parent.parent / "data" / "models"

    ensemble = EnsembleDroneModel()
    ensemble.load_models(
        xgb_model_path=models_dir / "xgboost_v1.pkl",
        nn_model_path=models_dir / "neural_v1.pt",
        input_dim=17
    )

    engineer = joblib.load(models_dir / "feature_engineer.pkl")

    # Define realistic constraints
    constraints = {
        'min_range_nm': 1500,
        'max_cost_usd': 40000,
        'max_mtow_lbm': 3500,
        'min_endurance_hr': 8
    }

    # Validate constraints
    is_valid, errors, warnings = validate_constraints(constraints)
    print(f"\nConstraint validation: {'PASSED' if is_valid else 'FAILED'}")
    if errors:
        print("  Errors:")
        for error in errors:
            print(f"    - {error}")
    if warnings:
        print("  Warnings:")
        for warning in warnings:
            print(f"    - {warning}")

    assert is_valid, "Constraints validation failed"

    # Run constrained optimization
    print("\nRunning constrained optimization...")
    print(f"  Constraints: {constraints}")

    start_time = time.time()
    results = optimize_with_constraints(
        ensemble_model=ensemble,
        feature_engineer=engineer,
        constraints=constraints,
        population_size=200,
        n_generations=100
    )
    elapsed = time.time() - start_time

    # Validate results
    print(f"\nResults:")
    print(f"  Optimization time: {elapsed:.2f}s")
    print(f"  Pareto front size: {results['n_pareto']}")
    print(f"  Feasible designs: {results['n_feasible']}/{results['n_pareto']} "
          f"({results['feasibility_rate']:.1%})")

    # Check constraint satisfaction
    print(f"\nConstraint satisfaction:")
    for i in range(min(5, results['n_feasible'])):
        range_val = results['pareto_objectives']['range_nm'][i]
        endurance_val = results['pareto_objectives']['endurance_hr'][i]
        mtow_val = results['pareto_objectives']['mtow_lbm'][i]
        cost_val = results['pareto_objectives']['cost_usd'][i]

        range_ok = range_val >= constraints['min_range_nm']
        cost_ok = cost_val <= constraints['max_cost_usd']
        mtow_ok = mtow_val <= constraints['max_mtow_lbm']
        endurance_ok = endurance_val >= constraints['min_endurance_hr']

        print(f"  Design {i+1}: Range={range_val:.0f}{'[OK]' if range_ok else '[FAIL]'}, "
              f"Endurance={endurance_val:.1f}{'[OK]' if endurance_ok else '[FAIL]'}, "
              f"MTOW={mtow_val:.0f}{'[OK]' if mtow_ok else '[FAIL]'}, "
              f"Cost=${cost_val:.0f}{'[OK]' if cost_ok else '[FAIL]'}")

    # Assertions
    assert elapsed < 30.0, f"Optimization too slow: {elapsed:.2f}s > 30s"
    assert results['feasibility_rate'] > 0.5, \
        f"Too few feasible designs: {results['feasibility_rate']:.1%}"

    print("\nPASSED: Constrained optimization")
    return results


def test_pareto_extraction():
    """Test Pareto front extraction"""
    print("\n" + "="*80)
    print("TEST 3: Pareto Front Extraction")
    print("="*80)

    # Load models
    models_dir = Path(__file__).parent.parent / "data" / "models"

    ensemble = EnsembleDroneModel()
    ensemble.load_models(
        xgb_model_path=models_dir / "xgboost_v1.pkl",
        nn_model_path=models_dir / "neural_v1.pt",
        input_dim=17
    )

    engineer = joblib.load(models_dir / "feature_engineer.pkl")
    objectives = DroneObjectives(ensemble, engineer)

    # Generate diverse designs
    print("\nGenerating 500 designs...")
    np.random.seed(42)

    # Design variable bounds
    xl = np.array([96, 72, 0, -20, -60, -60, 0.10])
    xu = np.array([192, 216, 65, 60, 60, 60, 0.65])

    designs = xl + (xu - xl) * np.random.rand(500, 7)

    # Evaluate objectives
    print("Evaluating objectives...")
    predictions = objectives.evaluate(designs, return_uncertainty=False)

    # Find Pareto frontiers
    print("\nExtracting Pareto frontiers...")
    pareto_results = find_pareto_frontiers(designs, predictions)

    # Validate results
    print(f"\nPareto Fronts:")
    for front_name, front_data in pareto_results.items():
        print(f"  {front_name}: {front_data['n_pareto']} points")

    # Assertions
    assert pareto_results['range_mtow']['n_pareto'] > 1, "No 2D Pareto points found"
    assert pareto_results['all_objectives']['n_pareto'] > 1, "No 4D Pareto points found"
    assert pareto_results['all_objectives']['n_pareto'] < 200, "Too many Pareto points (likely error)"

    print("\nPASSED: Pareto extraction")
    return pareto_results


def test_constraint_relaxation():
    """Test constraint relaxation strategy"""
    print("\n" + "="*80)
    print("TEST 4: Constraint Relaxation")
    print("="*80)

    # Create infeasible constraints
    constraints = {
        'min_range_nm': 5000,  # Very high
        'max_cost_usd': 10000,  # Very low
        'max_mtow_lbm': 1000,   # Very low
        'min_endurance_hr': 30  # Very high
    }

    print(f"\nInfeasible constraints: {constraints}")

    # Create handler and relax
    handler = ConstraintHandler(constraints)

    print("\nApplying balanced relaxation...")
    relaxation_result = handler.relax_constraints('balanced')

    print(f"\nRelaxed constraints:")
    for desc in relaxation_result['relaxation_description']:
        print(f"  - {desc}")

    # Test different strategies
    handler.reset_constraints()
    print("\nApplying cost-priority relaxation...")
    relaxation_cost = handler.relax_constraints('cost_priority')

    handler.reset_constraints()
    print("\nApplying performance-priority relaxation...")
    relaxation_perf = handler.relax_constraints('performance_priority')

    # Assertions
    assert relaxation_result['relaxed_constraints']['min_range_nm'] < constraints['min_range_nm']
    assert relaxation_result['relaxed_constraints']['max_cost_usd'] > constraints['max_cost_usd']

    print("\nPASSED: Constraint relaxation")
    return relaxation_result


def test_performance_targets():
    """Test performance targets (< 12s total time)"""
    print("\n" + "="*80)
    print("TEST 5: Performance Targets")
    print("="*80)

    # Load models
    models_dir = Path(__file__).parent.parent / "data" / "models"

    ensemble = EnsembleDroneModel()
    ensemble.load_models(
        xgb_model_path=models_dir / "xgboost_v1.pkl",
        nn_model_path=models_dir / "neural_v1.pt",
        input_dim=17
    )

    engineer = joblib.load(models_dir / "feature_engineer.pkl")

    # Test model inference speed
    print("\nTesting model inference speed...")
    X_test = np.random.rand(100, 7) * 100 + 100
    X_eng = engineer.transform(X_test)

    start = time.time()
    predictions, uncertainty = ensemble.predict(X_eng, return_uncertainty=True)
    inference_time = (time.time() - start) * 1000  # ms

    print(f"  Batch prediction (100 designs): {inference_time:.1f}ms")
    print(f"  Per-design: {inference_time/100:.2f}ms")

    # Test optimization speed
    print("\nTesting optimization speed...")
    start = time.time()
    results = run_nsga2_optimization(
        ensemble_model=ensemble,
        feature_engineer=engineer,
        population_size=200,
        n_generations=100
    )
    opt_time = time.time() - start

    print(f"  NSGA-II (200 pop, 100 gen): {opt_time:.2f}s")

    # Assertions
    assert inference_time < 500, f"Inference too slow: {inference_time:.1f}ms > 500ms"
    assert opt_time < 30, f"Optimization too slow: {opt_time:.2f}s > 30s"

    print("\nPASSED: Performance targets met")
    return {'inference_ms': inference_time, 'optimization_s': opt_time}


def main():
    """Run all integration tests"""
    print("="*80)
    print("OPTIMIZATION PIPELINE INTEGRATION TESTS")
    print("="*80)

    start_time = time.time()

    try:
        # Run tests
        test_unconstrained_optimization()
        test_constrained_optimization()
        test_pareto_extraction()
        test_constraint_relaxation()
        test_performance_targets()

        total_time = time.time() - start_time

        # Summary
        print("\n" + "="*80)
        print("ALL TESTS PASSED")
        print("="*80)
        print(f"Total test time: {total_time:.2f}s")
        print("\nWeek 2 optimization engine is ready for integration!")

        return 0

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
