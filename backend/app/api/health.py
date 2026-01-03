"""
Health check endpoint
"""

from fastapi import APIRouter, Request
from datetime import datetime, UTC

from app.schemas.health import HealthResponse

router = APIRouter()


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
