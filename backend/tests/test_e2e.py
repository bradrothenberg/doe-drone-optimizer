"""
End-to-End tests for DOE Drone Optimizer
Tests complete user workflows and edge cases

Updated to use fixed-span (12ft) model
"""

import pytest
import sys
from pathlib import Path
import time

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app


# Use fixed-span models directory (the models we have in the repo)
MODELS_DIR = backend_dir / "data" / "models_fixed_span_12ft"


@pytest.fixture(scope="module")
def client():
    """Create test client with model loading"""
    from app.core.model_manager import ModelManager

    # Check if models exist
    if not MODELS_DIR.exists():
        pytest.skip(f"Models directory not found: {MODELS_DIR}")

    model_manager = ModelManager()
    model_manager.load_models(MODELS_DIR)
    app.state.model_manager = model_manager

    return TestClient(app)


class TestWingtipDeflection:
    """Tests for wingtip deflection constraint (added in Week 4)"""

    def test_prediction_includes_wingtip_deflection(self, client):
        """Verify predictions include wingtip deflection (fixed-span model)"""
        # Fixed-span model: 6 inputs, NO span parameter
        request_data = {
            "designs": [{
                "loa": 150,
                "le_sweep_p1": 20,
                "le_sweep_p2": 10,
                "te_sweep_p1": -15,
                "te_sweep_p2": -10,
                "panel_break": 0.4
            }],
            "return_uncertainty": True
        }

        response = client.post("/api/predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        pred = data['predictions'][0]

        # Check wingtip deflection is present
        assert 'wingtip_deflection_in' in pred
        assert 'wingtip_deflection_in_uncertainty' in pred
        assert pred['wingtip_deflection_in'] >= 0
        assert pred['wingtip_deflection_in'] <= 100  # Clamped max

    def test_optimization_with_wingtip_constraint(self, client):
        """Test optimization with wingtip deflection constraint"""
        request_data = {
            "constraints": {
                "min_range_nm": 1000,
                "max_wingtip_deflection_in": 20  # Max 20 inches deflection
            },
            "population_size": 100,
            "n_generations": 50,
            "n_designs": 10
        }

        response = client.post("/api/optimize", json=request_data)
        assert response.status_code == 200

        data = response.json()

        # Check wingtip deflection in results
        for design in data['pareto_designs']:
            assert 'wingtip_deflection_in' in design
            # If feasible, should respect constraint
            if data['feasible']:
                assert design['wingtip_deflection_in'] <= 20


class TestEdgeCases:
    """Tests for edge cases and boundary conditions (fixed-span model)"""

    def test_minimum_design_parameters(self, client):
        """Test with minimum valid design parameters (fixed-span model)"""
        # Fixed-span model: 6 inputs, NO span parameter
        request_data = {
            "designs": [{
                "loa": 96,      # Min LOA
                "le_sweep_p1": 0,
                "le_sweep_p2": -20,
                "te_sweep_p1": -60,
                "te_sweep_p2": -60,
                "panel_break": 0.1
            }]
        }

        response = client.post("/api/predict", json=request_data)
        assert response.status_code == 200

        pred = response.json()['predictions'][0]
        assert pred['range_nm'] > 0
        assert pred['mtow_lbm'] > 0

    def test_maximum_design_parameters(self, client):
        """Test with maximum valid design parameters (fixed-span model)"""
        # Fixed-span model: 6 inputs, NO span parameter
        request_data = {
            "designs": [{
                "loa": 192,     # Max LOA
                "le_sweep_p1": 65,
                "le_sweep_p2": 60,
                "te_sweep_p1": 60,
                "te_sweep_p2": 60,
                "panel_break": 0.65
            }]
        }

        response = client.post("/api/predict", json=request_data)
        assert response.status_code == 200

        pred = response.json()['predictions'][0]
        assert pred['range_nm'] > 0
        assert pred['mtow_lbm'] > 0

    def test_empty_constraints(self, client):
        """Test optimization with empty constraints object"""
        request_data = {
            "constraints": {},
            "population_size": 50,
            "n_generations": 20,
            "n_designs": 5
        }

        response = client.post("/api/optimize", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data['pareto_designs']) == 5

    def test_single_constraint(self, client):
        """Test optimization with only one constraint"""
        request_data = {
            "constraints": {
                "min_range_nm": 2000
            },
            "population_size": 50,
            "n_generations": 20,
            "n_designs": 5
        }

        response = client.post("/api/optimize", json=request_data)
        assert response.status_code == 200

        data = response.json()
        if data['feasible']:
            for design in data['pareto_designs']:
                assert design['range_nm'] >= 2000


class TestPresetScenarios:
    """Tests matching frontend preset configurations"""

    def test_long_range_preset(self, client):
        """Test 'Long Range' preset configuration"""
        request_data = {
            "constraints": {
                "min_range_nm": 2500,
                "max_cost_usd": 50000,
                "max_mtow_lbm": 5000,
                "min_endurance_hr": 15,
                "max_wingtip_deflection_in": 40
            },
            "population_size": 100,
            "n_generations": 50,
            "n_designs": 20
        }

        response = client.post("/api/optimize", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data['pareto_designs']) > 0

    def test_low_cost_preset(self, client):
        """Test 'Low Cost' preset configuration"""
        request_data = {
            "constraints": {
                "min_range_nm": 1000,
                "max_cost_usd": 25000,
                "max_mtow_lbm": 2500,
                "min_endurance_hr": 5,
                "max_wingtip_deflection_in": 25
            },
            "population_size": 100,
            "n_generations": 50,
            "n_designs": 20
        }

        response = client.post("/api/optimize", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data['pareto_designs']) > 0

    def test_balanced_preset(self, client):
        """Test 'Balanced' preset configuration"""
        request_data = {
            "constraints": {
                "min_range_nm": 1500,
                "max_cost_usd": 35000,
                "max_mtow_lbm": 3000,
                "min_endurance_hr": 8,
                "max_wingtip_deflection_in": 30
            },
            "population_size": 100,
            "n_generations": 50,
            "n_designs": 20
        }

        response = client.post("/api/optimize", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data['pareto_designs']) > 0


class TestCSVExportFormat:
    """Tests to verify CSV export data format matches nTop requirements (fixed-span model)"""

    def test_design_parameters_for_ntop(self, client):
        """Verify all 6 nTop input parameters are in response (fixed-span model)"""
        request_data = {
            "population_size": 50,
            "n_generations": 20,
            "n_designs": 5
        }

        response = client.post("/api/optimize", json=request_data)
        assert response.status_code == 200

        data = response.json()
        design = data['pareto_designs'][0]

        # Required nTop input parameters for fixed-span model (no span)
        ntop_params = ['loa', 'le_sweep_p1', 'le_sweep_p2',
                       'te_sweep_p1', 'te_sweep_p2', 'panel_break']

        for param in ntop_params:
            assert param in design, f"Missing nTop parameter: {param}"
            assert isinstance(design[param], (int, float)), f"{param} should be numeric"

    def test_panel_break_is_ratio(self, client):
        """Verify panel_break is 0-1 ratio (not percentage)"""
        request_data = {
            "population_size": 50,
            "n_generations": 20,
            "n_designs": 10
        }

        response = client.post("/api/optimize", json=request_data)
        data = response.json()

        for design in data['pareto_designs']:
            assert 0 < design['panel_break'] < 1, \
                f"panel_break should be 0-1 ratio, got {design['panel_break']}"


class TestConcurrency:
    """Tests for concurrent request handling"""

    def test_concurrent_predictions(self, client):
        """Test multiple prediction requests (fixed-span model)"""
        import concurrent.futures

        # Fixed-span model: 6 inputs, NO span parameter
        design = {
            "loa": 150, "le_sweep_p1": 20, "le_sweep_p2": 10,
            "te_sweep_p1": -15, "te_sweep_p2": -10, "panel_break": 0.4
        }

        def make_request():
            return client.post("/api/predict", json={"designs": [design]})

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in futures]

        # All requests should succeed
        for r in results:
            assert r.status_code == 200


class TestAPIDocumentation:
    """Tests for API documentation endpoints"""

    def test_openapi_schema(self, client):
        """Test OpenAPI schema is available"""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert 'openapi' in schema
        assert 'paths' in schema
        assert '/api/predict' in schema['paths']
        assert '/api/optimize' in schema['paths']

    def test_docs_endpoint(self, client):
        """Test Swagger UI docs endpoint"""
        response = client.get("/docs")
        assert response.status_code == 200


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("End-to-End Tests - DOE Drone Design Optimizer (Fixed-Span Model)")
    print("=" * 80)

    pytest.main([__file__, "-v", "-s", "--tb=short"])
