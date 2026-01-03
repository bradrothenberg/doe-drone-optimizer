"""
Pydantic schemas for optimization endpoint
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any
from app.schemas.predict import DesignParameters, PredictionResult


class Constraints(BaseModel):
    """User constraints for optimization"""
    min_range_nm: Optional[float] = Field(None, ge=0, le=10000, description="Minimum range (nm)")
    max_cost_usd: Optional[float] = Field(None, ge=0, le=200000, description="Maximum cost ($)")
    max_mtow_lbm: Optional[float] = Field(None, ge=0, le=20000, description="Maximum MTOW (lbm)")
    min_endurance_hr: Optional[float] = Field(None, ge=0, le=100, description="Minimum endurance (hr)")

    @field_validator('*')
    @classmethod
    def check_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError("All constraints must be non-negative")
        return v

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class OptimizeRequest(BaseModel):
    """Request schema for optimization endpoint"""
    constraints: Optional[Constraints] = Field(None, description="User constraints (optional)")
    population_size: int = Field(default=200, ge=50, le=500, description="NSGA-II population size")
    n_generations: int = Field(default=100, ge=20, le=200, description="Number of generations")
    n_designs: int = Field(default=50, ge=1, le=200, description="Number of Pareto designs to return")

    class Config:
        json_schema_extra = {
            "example": {
                "constraints": {
                    "min_range_nm": 1500,
                    "max_cost_usd": 35000,
                    "max_mtow_lbm": 3000,
                    "min_endurance_hr": 8
                },
                "population_size": 200,
                "n_generations": 100,
                "n_designs": 50
            }
        }


class DesignResult(BaseModel):
    """Complete design result with parameters and predictions"""
    # Design parameters
    loa: float = Field(..., description="Length Overall (inches)")
    span: float = Field(..., description="Wing Span (inches)")
    le_sweep_p1: float = Field(..., description="Leading Edge Sweep Panel 1 (degrees)")
    le_sweep_p2: float = Field(..., description="Leading Edge Sweep Panel 2 (degrees)")
    te_sweep_p1: float = Field(..., description="Trailing Edge Sweep Panel 1 (degrees)")
    te_sweep_p2: float = Field(..., description="Trailing Edge Sweep Panel 2 (degrees)")
    panel_break: float = Field(..., description="Panel Break (fraction of span)")

    # Performance predictions
    range_nm: float = Field(..., description="Range (nautical miles)")
    endurance_hr: float = Field(..., description="Endurance (hours)")
    mtow_lbm: float = Field(..., description="Max Takeoff Weight (lbm)")
    cost_usd: float = Field(..., description="Material Cost (USD)")

    # Uncertainty estimates
    uncertainty_range_nm: float = Field(..., description="Range uncertainty (nm)")
    uncertainty_endurance_hr: float = Field(..., description="Endurance uncertainty (hr)")
    uncertainty_mtow_lbm: float = Field(..., description="MTOW uncertainty (lbm)")
    uncertainty_cost_usd: float = Field(..., description="Cost uncertainty (USD)")


class OptimizeResponse(BaseModel):
    """Response schema for optimization endpoint"""
    pareto_designs: List[DesignResult] = Field(..., description="Pareto-optimal designs")
    n_pareto: int = Field(..., description="Total Pareto front size")
    feasible: bool = Field(..., description="Whether all designs satisfy constraints")
    optimization_time_s: float = Field(..., description="Optimization time (seconds)")

    # Optional constraint relaxation info
    constraint_relaxation: Optional[Dict[str, Any]] = Field(None, description="Applied constraint relaxation")
    warnings: Optional[List[str]] = Field(None, description="Validation warnings")

    class Config:
        json_schema_extra = {
            "example": {
                "designs": [
                    {
                        "design_parameters": {
                            "loa": 150,
                            "span": 180,
                            "le_sweep_p1": 20,
                            "le_sweep_p2": 15,
                            "te_sweep_p1": -10,
                            "te_sweep_p2": -5,
                            "panel_break": 0.4
                        },
                        "predictions": {
                            "range_nm": 2500,
                            "endurance_hr": 15,
                            "mtow_lbm": 2000,
                            "cost_usd": 25000
                        },
                        "rank": 1
                    }
                ],
                "n_pareto": 200,
                "n_returned": 50,
                "feasibility_rate": 1.0,
                "optimization_time_s": 10.5,
                "algorithm": "NSGA-II"
            }
        }
