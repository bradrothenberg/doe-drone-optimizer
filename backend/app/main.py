"""
FastAPI Application for DOE Drone Design Optimizer
Provides REST API for drone design prediction and multi-objective optimization
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from app.api import predict, optimize, health, sensitivity
from app.core.model_manager import ModelManager
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model manager instance
model_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI application
    Handles startup and shutdown events
    """
    # Startup: Load models
    global model_manager
    logger.info("Starting DOE Drone Optimizer API...")

    try:
        model_manager = ModelManager()
        models_dir = Path(__file__).parent.parent / "data" / "models"
        model_manager.load_models(models_dir)

        # Store in app state
        app.state.model_manager = model_manager

        logger.info("Models loaded successfully")
        logger.info("API ready to accept requests")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Multi-objective optimization API for drone design using NSGA-II",
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan
)

# Configure CORS with environment-based origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(predict.router, prefix="/api", tags=["prediction"])
app.include_router(optimize.router, prefix="/api", tags=["optimization"])
app.include_router(sensitivity.router, prefix="/api", tags=["sensitivity"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    # Only show details in debug mode to prevent information leakage
    content = {"error": "Internal server error"}
    if settings.show_error_details:
        content["detail"] = str(exc)

    return JSONResponse(
        status_code=500,
        content=content
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
