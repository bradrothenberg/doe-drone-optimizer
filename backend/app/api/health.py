"""
Health check and model info endpoints
"""

from fastapi import APIRouter, Request, HTTPException
from datetime import datetime, UTC
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from app.schemas.health import HealthResponse

router = APIRouter()


class ModelInfoResponse(BaseModel):
    """Detailed model information response"""
    model_type: str = Field(..., description="Model variant: 'fixed_span_12ft' or 'variable_span'")
    fixed_span_inches: Optional[float] = Field(None, description="Fixed span value (if fixed-span model)")
    n_inputs: int = Field(..., description="Number of raw input features")
    n_features: int = Field(..., description="Number of engineered features")
    feature_names: List[str] = Field(..., description="List of all feature names")
    outputs: List[str] = Field(..., description="Model output names")
    ensemble: Dict[str, Any] = Field(..., description="Ensemble model configuration")
    bootstrap_ensemble: Dict[str, Any] = Field(..., description="Bootstrap ensemble info for uncertainty")
    performance_notes: List[str] = Field(default_factory=list, description="Important notes about model performance")


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(request: Request):
    """
    Health check endpoint

    Returns:
        HealthResponse with status, timestamp, and model info
    """
    model_manager = request.app.state.model_manager

    # Get model info
    model_info = None
    if model_manager.is_loaded:
        model_info = model_manager.get_model_info()

    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "unhealthy",
        timestamp=datetime.now(UTC).isoformat(),
        models_loaded=model_manager.is_loaded,
        model_info=model_info,
        version="1.0.0"
    )


@router.get("/api/model-info", response_model=ModelInfoResponse, tags=["health"])
async def get_model_info(request: Request):
    """
    Get detailed model information for runtime introspection

    Returns comprehensive information about the loaded model including:
    - Model type (fixed-span vs variable-span)
    - Feature engineering details
    - Ensemble configuration
    - Bootstrap uncertainty availability
    - Performance notes and known limitations

    This endpoint helps users understand which model configuration is active
    and any important caveats about predictions.
    """
    model_manager = request.app.state.model_manager

    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Determine model type and input count
    is_fixed_span = model_manager.is_fixed_span_model()
    n_inputs = 6 if is_fixed_span else 7

    # Get feature names
    feature_names = model_manager.feature_engineer.get_feature_names()

    # Output names
    outputs = [
        "range_nm",
        "endurance_hr",
        "mtow_lbm",
        "cost_usd",
        "wingtip_deflection_in"
    ]

    # Ensemble info
    ensemble_info = {
        "xgb_weight": model_manager.ensemble_model.xgb_weight,
        "nn_weight": model_manager.ensemble_model.nn_weight,
        "xgb_loaded": model_manager.ensemble_model.xgb_model is not None,
        "nn_loaded": model_manager.ensemble_model.nn_model is not None
    }

    # Bootstrap info
    bootstrap_info = {
        "available": model_manager.has_bootstrap_ensemble(),
        "n_models": len(model_manager.bootstrap_models) if model_manager.bootstrap_models else 0,
        "uncertainty_method": "bootstrap_std" if model_manager.has_bootstrap_ensemble() else "ensemble_disagreement"
    }

    # Performance notes - important caveats for users
    performance_notes = []

    # Note about deflection prediction accuracy
    performance_notes.append(
        "Wingtip deflection predictions have lower accuracy (R²≈0.71) compared to other outputs (R²>0.97). "
        "Use deflection predictions with caution and consider adding safety margins."
    )

    if is_fixed_span:
        performance_notes.append(
            f"Fixed-span model: All predictions assume span = {model_manager.fixed_span} inches (12 ft). "
            "Span is not a design variable in this configuration."
        )

    if not model_manager.has_bootstrap_ensemble():
        performance_notes.append(
            "Bootstrap ensemble not available. Uncertainty estimates may be less reliable."
        )

    return ModelInfoResponse(
        model_type="fixed_span_12ft" if is_fixed_span else "variable_span",
        fixed_span_inches=model_manager.fixed_span if is_fixed_span else None,
        n_inputs=n_inputs,
        n_features=len(feature_names),
        feature_names=feature_names,
        outputs=outputs,
        ensemble=ensemble_info,
        bootstrap_ensemble=bootstrap_info,
        performance_notes=performance_notes
    )
