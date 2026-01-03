"""
Pydantic schemas for health check endpoint
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime


class HealthResponse(BaseModel):
    """Response schema for health check endpoint"""
    status: str = Field(..., description="API status (healthy/unhealthy)")
    timestamp: str = Field(..., description="Current server timestamp (ISO format)")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")
    model_info: Optional[Dict] = Field(None, description="Information about loaded models")
    version: str = Field(default="1.0.0", description="API version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-03T12:00:00Z",
                "models_loaded": True,
                "model_info": {
                    "ensemble": {
                        "xgb_weight": 0.6,
                        "nn_weight": 0.4
                    },
                    "feature_engineer": {
                        "n_features": 17
                    }
                },
                "version": "1.0.0"
            }
        }
