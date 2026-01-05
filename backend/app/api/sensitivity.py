"""
Sensitivity Analysis API
Computes local sensitivity of outputs to input changes
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class DesignPoint(BaseModel):
    """Single design point for sensitivity analysis"""
    loa: float = Field(..., ge=96, le=192)
    span: Optional[float] = Field(default=None, ge=72, le=216)  # Optional for fixed-span mode
    le_sweep_p1: float = Field(..., ge=0, le=65)
    le_sweep_p2: float = Field(..., ge=-20, le=60)
    te_sweep_p1: float = Field(..., ge=-60, le=60)
    te_sweep_p2: float = Field(..., ge=-60, le=60)
    panel_break: float = Field(..., ge=0.1, le=0.65)


class SensitivityRequest(BaseModel):
    """Request for sensitivity analysis"""
    design: DesignPoint
    perturbation_pct: float = Field(default=10.0, ge=1.0, le=50.0)


class InputSensitivity(BaseModel):
    """Sensitivity of outputs to a single input"""
    input_name: str
    base_value: float
    perturbed_value: float
    range_nm_delta: float
    endurance_hr_delta: float
    mtow_lbm_delta: float
    cost_usd_delta: float
    wingtip_deflection_in_delta: float


class SensitivityResponse(BaseModel):
    """Response with sensitivity results"""
    design: DesignPoint
    perturbation_pct: float
    sensitivities: List[InputSensitivity]
    computation_time_ms: float


# Input bounds for proper perturbation
INPUT_BOUNDS = {
    'loa': (96, 192),
    'span': (72, 216),
    'le_sweep_p1': (0, 65),
    'le_sweep_p2': (-20, 60),
    'te_sweep_p1': (-60, 60),
    'te_sweep_p2': (-60, 60),
    'panel_break': (0.1, 0.65)
}

INPUT_LABELS = {
    'loa': 'LOA (in)',
    'span': 'Span (in)',
    'le_sweep_p1': 'LE Sweep P1 (deg)',
    'le_sweep_p2': 'LE Sweep P2 (deg)',
    'te_sweep_p1': 'TE Sweep P1 (deg)',
    'te_sweep_p2': 'TE Sweep P2 (deg)',
    'panel_break': 'Panel Break (%)'
}


def predict_single(ensemble_model, feature_engineer, inputs: np.ndarray) -> dict:
    """Helper to predict outputs for a single design point"""
    X_eng = feature_engineer.transform(inputs)
    predictions, _ = ensemble_model.predict(X_eng, return_uncertainty=False)
    return {
        'range_nm': float(predictions[0, 0]),
        'endurance_hr': float(predictions[0, 1]),
        'mtow_lbm': float(predictions[0, 2]),
        'cost_usd': float(predictions[0, 3]),
        # Clamp wingtip deflection to non-negative (model can extrapolate to negative)
        'wingtip_deflection_in': float(max(0, predictions[0, 4]))
    }


@router.post("/sensitivity", response_model=SensitivityResponse)
async def compute_sensitivity(request: SensitivityRequest, req: Request):
    """
    Compute local sensitivity analysis around a design point.

    Perturbs each input by the specified percentage and measures
    the change in all outputs.

    In fixed-span mode (12ft), span is not varied.
    """
    start_time = time.time()

    model_manager = req.app.state.model_manager
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Get models
    ensemble_model = model_manager.get_ensemble_model()
    feature_engineer = model_manager.get_feature_engineer()

    # Check if using fixed-span mode
    fixed_span = model_manager.get_fixed_span()
    is_fixed_span = fixed_span is not None

    design = request.design
    pct = request.perturbation_pct / 100.0

    # Build base design as numpy array based on mode
    if is_fixed_span:
        # Fixed-span mode: 6 inputs (no span)
        base_inputs = np.array([[
            design.loa, design.le_sweep_p1, design.le_sweep_p2,
            design.te_sweep_p1, design.te_sweep_p2, design.panel_break
        ]])
        # Use fixed-span input bounds and labels (exclude span)
        input_bounds = {k: v for k, v in INPUT_BOUNDS.items() if k != 'span'}
        input_labels = {k: v for k, v in INPUT_LABELS.items() if k != 'span'}
        logger.info(f"Sensitivity analysis using fixed-span mode (span={fixed_span} inches)")
    else:
        # Variable-span mode: 7 inputs (includes span)
        span_val = design.span if design.span is not None else 144.0
        base_inputs = np.array([[
            design.loa, span_val, design.le_sweep_p1, design.le_sweep_p2,
            design.te_sweep_p1, design.te_sweep_p2, design.panel_break
        ]])
        input_bounds = INPUT_BOUNDS
        input_labels = INPUT_LABELS

    # Get base prediction
    base_outputs = predict_single(ensemble_model, feature_engineer, base_inputs)

    sensitivities = []
    input_names = list(input_bounds.keys())

    for i, name in enumerate(input_names):
        bounds = input_bounds[name]
        base_val = base_inputs[0, i]

        # Compute perturbation (use range-based for bounded inputs)
        input_range = bounds[1] - bounds[0]
        delta = input_range * pct

        # Perturb upward (clamped to bounds)
        perturbed_val = min(base_val + delta, bounds[1])

        # Create perturbed input
        perturbed_inputs = base_inputs.copy()
        perturbed_inputs[0, i] = perturbed_val

        # Get perturbed prediction
        perturbed_outputs = predict_single(ensemble_model, feature_engineer, perturbed_inputs)

        # Compute deltas
        sensitivity = InputSensitivity(
            input_name=input_labels[name],
            base_value=float(base_val),
            perturbed_value=float(perturbed_val),
            range_nm_delta=float(perturbed_outputs['range_nm'] - base_outputs['range_nm']),
            endurance_hr_delta=float(perturbed_outputs['endurance_hr'] - base_outputs['endurance_hr']),
            mtow_lbm_delta=float(perturbed_outputs['mtow_lbm'] - base_outputs['mtow_lbm']),
            cost_usd_delta=float(perturbed_outputs['cost_usd'] - base_outputs['cost_usd']),
            wingtip_deflection_in_delta=float(perturbed_outputs['wingtip_deflection_in'] - base_outputs['wingtip_deflection_in'])
        )
        sensitivities.append(sensitivity)

    computation_time_ms = (time.time() - start_time) * 1000

    # Update design span for response (use fixed span if in fixed-span mode)
    response_design = DesignPoint(
        loa=design.loa,
        span=fixed_span if is_fixed_span else design.span,
        le_sweep_p1=design.le_sweep_p1,
        le_sweep_p2=design.le_sweep_p2,
        te_sweep_p1=design.te_sweep_p1,
        te_sweep_p2=design.te_sweep_p2,
        panel_break=design.panel_break
    )

    return SensitivityResponse(
        design=response_design,
        perturbation_pct=request.perturbation_pct,
        sensitivities=sensitivities,
        computation_time_ms=computation_time_ms
    )
