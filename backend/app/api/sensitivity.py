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
    span: float = Field(..., ge=72, le=216)
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


@router.post("/sensitivity", response_model=SensitivityResponse)
async def compute_sensitivity(request: SensitivityRequest, req: Request):
    """
    Compute local sensitivity analysis around a design point.

    Perturbs each input by the specified percentage and measures
    the change in all outputs.
    """
    start_time = time.time()

    model_manager = req.app.state.model_manager
    if not model_manager or not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Models not loaded")

    design = request.design
    pct = request.perturbation_pct / 100.0

    # Base design as numpy array
    base_inputs = np.array([[
        design.loa, design.span, design.le_sweep_p1, design.le_sweep_p2,
        design.te_sweep_p1, design.te_sweep_p2, design.panel_break
    ]])

    # Get base prediction
    base_pred = model_manager.predict(base_inputs)
    base_outputs = base_pred['predictions'][0]

    sensitivities = []
    input_names = list(INPUT_BOUNDS.keys())

    for i, name in enumerate(input_names):
        bounds = INPUT_BOUNDS[name]
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
        perturbed_pred = model_manager.predict(perturbed_inputs)
        perturbed_outputs = perturbed_pred['predictions'][0]

        # Compute deltas
        sensitivity = InputSensitivity(
            input_name=INPUT_LABELS[name],
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

    return SensitivityResponse(
        design=design,
        perturbation_pct=request.perturbation_pct,
        sensitivities=sensitivities,
        computation_time_ms=computation_time_ms
    )
