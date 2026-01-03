"""
Integration tests for FastAPI endpoints
Tests health, predict, and optimize endpoints
"""

import pytest
import sys
from pathlib import Path
import numpy as np
import time

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(backend_dir))

# Import FastAPI test client and app
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    """Create test client with model loading"""
    from app.core.model_manager import ModelManager

    # Load models
    model_manager = ModelManager()
    models_dir = backend_dir / "data" / "models"
    model_manager.load_models(models_dir)

    # Set in app state
    app.state.model_manager = model_manager

    return TestClient(app)


class TestHealthEndpoint:
    """Test suite for /health endpoint"""

    def test_health_check(self, client):
        """Test basic health check"""
        print("\n" + "="*80)
        print("TEST 1: Health Check")
        print("="*80)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        print(f"Status: {data['status']}")
        print(f"Models loaded: {data['models_loaded']}")
        print(f"Version: {data['version']}")

        assert data['status'] == 'healthy'
        assert data['models_loaded'] is True
        assert 'timestamp' in data
        assert 'model_info' in data

        # Check model info
        model_info = data['model_info']
        assert 'ensemble' in model_info
        assert 'feature_engineer' in model_info

        print(f"Ensemble weights: XGB={model_info['ensemble']['xgb_weight']}, NN={model_info['ensemble']['nn_weight']}")
        print(f"Features: {model_info['feature_engineer']['n_features']}")

        print("PASSED\n")

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert 'message' in data
        assert 'version' in data
        assert 'docs' in data


class TestPredictEndpoint:
    """Test suite for /api/predict endpoint"""

    def test_single_design_prediction(self, client):
        """Test prediction for a single design"""
        print("\n" + "="*80)
        print("TEST 2: Single Design Prediction")
        print("="*80)

        request_data = {
            "designs": [
                {
                    "loa": 150,
                    "span": 180,
                    "le_sweep_p1": 20,
                    "le_sweep_p2": 10,
                    "te_sweep_p1": -15,
                    "te_sweep_p2": -10,
                    "panel_break": 0.4
                }
            ],
            "return_uncertainty": True
        }

        start_time = time.time()
        response = client.post("/api/predict", json=request_data)
        elapsed_ms = (time.time() - start_time) * 1000

        assert response.status_code == 200
        data = response.json()

        assert data['n_designs'] == 1
        assert len(data['predictions']) == 1
        assert data['inference_time_ms'] > 0

        pred = data['predictions'][0]
        print(f"Design: LOA={request_data['designs'][0]['loa']}, Span={request_data['designs'][0]['span']}")
        print(f"Predictions:")
        print(f"  Range: {pred['range_nm']:.1f} ± {pred['range_nm_uncertainty']:.1f} nm")
        print(f"  Endurance: {pred['endurance_hr']:.2f} ± {pred['endurance_hr_uncertainty']:.2f} hr")
        print(f"  MTOW: {pred['mtow_lbm']:.1f} ± {pred['mtow_lbm_uncertainty']:.1f} lbm")
        print(f"  Cost: ${pred['cost_usd']:.0f} ± ${pred['cost_usd_uncertainty']:.0f}")
        print(f"Inference time: {data['inference_time_ms']:.2f}ms (API: {elapsed_ms:.2f}ms)")

        # Validate predictions are reasonable
        assert 0 < pred['range_nm'] < 10000
        assert 0 < pred['endurance_hr'] < 400  # Max from dataset is ~340 hr
        assert 0 < pred['mtow_lbm'] < 20000
        assert 0 < pred['cost_usd'] < 200000

        print("PASSED\n")

    def test_batch_prediction(self, client):
        """Test batch prediction for multiple designs"""
        print("\n" + "="*80)
        print("TEST 3: Batch Prediction (100 designs)")
        print("="*80)

        # Generate 100 random designs
        np.random.seed(42)
        designs = []
        for _ in range(100):
            designs.append({
                "loa": float(np.random.uniform(96, 192)),
                "span": float(np.random.uniform(72, 216)),
                "le_sweep_p1": float(np.random.uniform(0, 65)),
                "le_sweep_p2": float(np.random.uniform(-20, 60)),
                "te_sweep_p1": float(np.random.uniform(-60, 60)),
                "te_sweep_p2": float(np.random.uniform(-60, 60)),
                "panel_break": float(np.random.uniform(0.1, 0.65))
            })

        request_data = {
            "designs": designs,
            "return_uncertainty": True
        }

        start_time = time.time()
        response = client.post("/api/predict", json=request_data)
        elapsed_ms = (time.time() - start_time) * 1000

        assert response.status_code == 200
        data = response.json()

        assert data['n_designs'] == 100
        assert len(data['predictions']) == 100

        print(f"Batch size: {data['n_designs']}")
        print(f"Inference time: {data['inference_time_ms']:.2f}ms (API: {elapsed_ms:.2f}ms)")
        print(f"Per-design time: {data['inference_time_ms']/100:.2f}ms")

        # Check performance target
        assert data['inference_time_ms'] < 500, "Batch inference should be < 500ms"

        print("PASSED\n")

    def test_invalid_design_parameters(self, client):
        """Test validation of invalid design parameters"""
        print("\n" + "="*80)
        print("TEST 4: Invalid Design Parameters")
        print("="*80)

        # Test out-of-bounds LOA
        request_data = {
            "designs": [
                {
                    "loa": 300,  # Invalid: max is 192
                    "span": 180,
                    "le_sweep_p1": 20,
                    "le_sweep_p2": 10,
                    "te_sweep_p1": -15,
                    "te_sweep_p2": -10,
                    "panel_break": 0.4
                }
            ]
        }

        response = client.post("/api/predict", json=request_data)

        print(f"Response status: {response.status_code}")
        assert response.status_code == 422  # Validation error

        print("PASSED - Invalid parameters rejected\n")


class TestOptimizeEndpoint:
    """Test suite for /api/optimize endpoint"""

    def test_unconstrained_optimization(self, client):
        """Test unconstrained optimization"""
        print("\n" + "="*80)
        print("TEST 5: Unconstrained Optimization")
        print("="*80)

        request_data = {
            "population_size": 100,
            "n_generations": 50,
            "n_designs": 20
        }

        start_time = time.time()
        response = client.post("/api/optimize", json=request_data)
        elapsed_s = time.time() - start_time

        assert response.status_code == 200
        data = response.json()

        assert data['n_pareto'] >= 20
        assert len(data['pareto_designs']) == 20
        assert data['feasible'] is True

        print(f"Optimization time: {data['optimization_time_s']:.2f}s (API: {elapsed_s:.2f}s)")
        print(f"Pareto front size: {data['n_pareto']}")
        print(f"Returned designs: {len(data['pareto_designs'])}")

        # Check design structure
        design = data['pareto_designs'][0]
        assert 'loa' in design
        assert 'range_nm' in design
        assert 'uncertainty_range_nm' in design

        # Print objective ranges
        ranges = [d['range_nm'] for d in data['pareto_designs']]
        endurances = [d['endurance_hr'] for d in data['pareto_designs']]
        mtows = [d['mtow_lbm'] for d in data['pareto_designs']]
        costs = [d['cost_usd'] for d in data['pareto_designs']]

        print(f"\nObjective ranges:")
        print(f"  Range: {min(ranges):.1f} - {max(ranges):.1f} nm")
        print(f"  Endurance: {min(endurances):.1f} - {max(endurances):.1f} hr")
        print(f"  MTOW: {min(mtows):.1f} - {max(mtows):.1f} lbm")
        print(f"  Cost: ${min(costs):.0f} - ${max(costs):.0f}")

        # Check performance target
        assert data['optimization_time_s'] < 30, "Optimization should be < 30s"

        print("PASSED\n")

    def test_constrained_optimization(self, client):
        """Test optimization with user constraints"""
        print("\n" + "="*80)
        print("TEST 6: Constrained Optimization")
        print("="*80)

        request_data = {
            "constraints": {
                "min_range_nm": 1500,
                "max_cost_usd": 40000,
                "max_mtow_lbm": 3500,
                "min_endurance_hr": 8
            },
            "population_size": 100,
            "n_generations": 50,
            "n_designs": 20
        }

        print(f"Constraints: {request_data['constraints']}")

        start_time = time.time()
        response = client.post("/api/optimize", json=request_data)
        elapsed_s = time.time() - start_time

        assert response.status_code == 200
        data = response.json()

        print(f"Optimization time: {data['optimization_time_s']:.2f}s")
        print(f"Pareto front size: {data['n_pareto']}")
        print(f"Feasible: {data['feasible']}")

        if data['constraint_relaxation']:
            print(f"WARNING:  Constraint relaxation applied:")
            print(f"  Strategy: {data['constraint_relaxation']['strategy']}")
            print(f"  Original: {data['constraint_relaxation']['original']}")
            print(f"  Relaxed: {data['constraint_relaxation']['relaxed']}")

        if data['warnings']:
            print(f"WARNING:  Warnings: {data['warnings']}")

        # Verify constraints are satisfied (or relaxed)
        for design in data['pareto_designs']:
            if data['feasible']:
                assert design['range_nm'] >= request_data['constraints']['min_range_nm']
                assert design['cost_usd'] <= request_data['constraints']['max_cost_usd']
                assert design['mtow_lbm'] <= request_data['constraints']['max_mtow_lbm']
                assert design['endurance_hr'] >= request_data['constraints']['min_endurance_hr']

        print("PASSED\n")

    def test_invalid_constraints(self, client):
        """Test validation of invalid constraints"""
        print("\n" + "="*80)
        print("TEST 7: Invalid Constraints")
        print("="*80)

        # Test negative constraint
        request_data = {
            "constraints": {
                "min_range_nm": -1000  # Invalid: negative
            }
        }

        response = client.post("/api/optimize", json=request_data)

        print(f"Response status: {response.status_code}")
        assert response.status_code == 422  # Validation error

        print("PASSED - Invalid constraints rejected\n")

    def test_tight_constraints_relaxation(self, client):
        """Test constraint relaxation with very tight constraints"""
        print("\n" + "="*80)
        print("TEST 8: Tight Constraints with Relaxation")
        print("="*80)

        request_data = {
            "constraints": {
                "min_range_nm": 4000,  # Very high
                "max_cost_usd": 15000,  # Very low
                "max_mtow_lbm": 1500,   # Very low
                "min_endurance_hr": 20   # Very high
            },
            "population_size": 100,
            "n_generations": 50,
            "n_designs": 10
        }

        print(f"Tight constraints: {request_data['constraints']}")

        response = client.post("/api/optimize", json=request_data)

        # Should succeed with relaxation or fail gracefully
        if response.status_code == 200:
            data = response.json()
            print(f"Optimization succeeded")
            print(f"Feasible: {data['feasible']}")

            if data['constraint_relaxation']:
                print(f"Constraint relaxation applied: {data['constraint_relaxation']['strategy']}")
                print(f"Relaxed constraints: {data['constraint_relaxation']['relaxed']}")

            print("PASSED - Relaxation handled gracefully\n")
        elif response.status_code == 400:
            data = response.json()
            print(f"WARNING:  Optimization failed: {data['detail']}")
            print("PASSED - Failed gracefully with clear error\n")
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")


class TestPerformance:
    """Test suite for performance benchmarks"""

    def test_end_to_end_performance(self, client):
        """Test full optimization pipeline performance"""
        print("\n" + "="*80)
        print("TEST 9: End-to-End Performance")
        print("="*80)

        request_data = {
            "constraints": {
                "min_range_nm": 1500,
                "max_cost_usd": 35000
            },
            "population_size": 200,
            "n_generations": 100,
            "n_designs": 50
        }

        start_time = time.time()
        response = client.post("/api/optimize", json=request_data)
        elapsed_s = time.time() - start_time

        assert response.status_code == 200
        data = response.json()

        print(f"Total API time: {elapsed_s:.2f}s")
        print(f"Optimization time: {data['optimization_time_s']:.2f}s")
        print(f"Overhead: {(elapsed_s - data['optimization_time_s']):.2f}s")
        print(f"Pareto designs returned: {len(data['pareto_designs'])}")

        # Performance targets
        assert data['optimization_time_s'] < 30, "Optimization should be < 30s"
        assert elapsed_s < 35, "Total API time should be < 35s"

        print("PASSED - Performance targets met\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("FastAPI Integration Tests")
    print("DOE Drone Design Optimizer API")
    print("="*80)

    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
