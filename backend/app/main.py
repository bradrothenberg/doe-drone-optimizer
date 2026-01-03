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

from app.api import predict, optimize, health
from app.core.model_manager import ModelManager

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
    title="DOE Drone Design Optimizer API",
    description="Multi-objective optimization API for drone design using NSGA-II",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(predict.router, prefix="/api", tags=["prediction"])
app.include_router(optimize.router, prefix="/api", tags=["optimization"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "DOE Drone Design Optimizer API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
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
