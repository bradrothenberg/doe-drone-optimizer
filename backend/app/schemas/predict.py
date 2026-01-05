"""
Pydantic schemas for prediction endpoint
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any


class DesignParameters(BaseModel):
    """Design parameters for a single drone design

    For fixed-span models (12ft), span is optional and defaults to 144 inches.
    For variable-span models, span must be provided.
    """
    loa: float = Field(..., ge=96, le=192, description="Length Overall (inches)")
    span: Optional[float] = Field(default=144.0, ge=72, le=216, description="Wing Span (inches) - optional for fixed-span model")
    le_sweep_p1: float = Field(..., ge=0, le=65, description="Leading Edge Sweep Panel 1 (degrees)")
    le_sweep_p2: float = Field(..., ge=-20, le=60, description="Leading Edge Sweep Panel 2 (degrees)")
    te_sweep_p1: float = Field(..., ge=-60, le=60, description="Trailing Edge Sweep Panel 1 (degrees)")
    te_sweep_p2: float = Field(..., ge=-60, le=60, description="Trailing Edge Sweep Panel 2 (degrees)")
    panel_break: float = Field(..., ge=0.10, le=0.65, description="Panel Break Location (fraction of span)")

    def to_array(self, include_span: bool = True) -> List[float]:
        """Convert to array for model input

        Args:
            include_span: If True (variable-span model), include span in output.
                         If False (fixed-span model), exclude span.
        """
        if include_span:
            return [
                self.loa,
                self.span if self.span is not None else 144.0,
                self.le_sweep_p1,
                self.le_sweep_p2,
                self.te_sweep_p1,
                self.te_sweep_p2,
                self.panel_break
            ]
        else:
            return [
                self.loa,
                self.le_sweep_p1,
                self.le_sweep_p2,
                self.te_sweep_p1,
                self.te_sweep_p2,
                self.panel_break
            ]


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint"""
    designs: List[DesignParameters] = Field(..., min_length=1, max_length=1000, description="Design parameters (max 1000 designs)")
    return_uncertainty: bool = Field(default=True, description="Include uncertainty estimates")

    @field_validator('designs')
    @classmethod
    def validate_designs(cls, v):
        if len(v) == 0:
            raise ValueError("At least one design required")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 designs per request")
        return v


class PredictionResult(BaseModel):
    """Prediction result for a single design"""
    range_nm: float = Field(..., description="Predicted range (nautical miles)")
    endurance_hr: float = Field(..., description="Predicted endurance (hours)")
    mtow_lbm: float = Field(..., description="Predicted MTOW (lbm)")
    cost_usd: float = Field(..., description="Predicted material cost (USD)")
    wingtip_deflection_in: float = Field(..., description="Predicted wingtip deflection (inches)")

    # Optional uncertainty estimates
    range_nm_uncertainty: Optional[float] = Field(None, description="Range uncertainty (std dev)")
    endurance_hr_uncertainty: Optional[float] = Field(None, description="Endurance uncertainty (std dev)")
    mtow_lbm_uncertainty: Optional[float] = Field(None, description="MTOW uncertainty (std dev)")
    cost_usd_uncertainty: Optional[float] = Field(None, description="Cost uncertainty (std dev)")
    wingtip_deflection_in_uncertainty: Optional[float] = Field(None, description="Wingtip deflection uncertainty (std dev)")


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint"""
    predictions: List[PredictionResult] = Field(..., description="Predictions for each design")
    n_designs: int = Field(..., description="Number of designs predicted")
    inference_time_ms: float = Field(..., description="Total inference time (milliseconds)")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model ensemble weights (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "range_nm": 2500.5,
                        "endurance_hr": 15.3,
                        "mtow_lbm": 2000.0,
                        "cost_usd": 25000.0,
                        "range_nm_uncertainty": 50.2,
                        "endurance_hr_uncertainty": 0.8,
                        "mtow_lbm_uncertainty": 120.5,
                        "cost_usd_uncertainty": 1500.0
                    }
                ],
                "n_designs": 1,
                "inference_time_ms": 12.5,
                "model_info": {
                    "xgb_weight": 0.6,
                    "nn_weight": 0.4
                }
            }
        }
