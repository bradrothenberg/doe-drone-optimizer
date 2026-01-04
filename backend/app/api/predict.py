"""
Prediction endpoint for drone design performance
"""

from fastapi import APIRouter, Request, HTTPException
import numpy as np
import logging
import time

from app.schemas.predict import PredictRequest, PredictResponse, PredictionResult

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict_performance(request_data: PredictRequest, request: Request):
    """
    Predict performance metrics for given design parameters

    Args:
        request_data: Design parameters and options

    Returns:
        PredictResponse with predictions and optionally uncertainty
    """
    try:
        start_time = time.time()

        # Get models from app state
        model_manager = request.app.state.model_manager

        if not model_manager.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")

        ensemble_model = model_manager.get_ensemble_model()
        feature_engineer = model_manager.get_feature_engineer()

        # Convert designs to numpy array
        designs_array = np.array([
            [
                d.loa,
                d.span,
                d.le_sweep_p1,
                d.le_sweep_p2,
                d.te_sweep_p1,
                d.te_sweep_p2,
                d.panel_break
            ]
            for d in request_data.designs
        ])

        logger.info(f"Predicting performance for {len(designs_array)} designs")

        # Engineer features
        X_eng = feature_engineer.transform(designs_array)

        # Predict
        predictions, uncertainty = ensemble_model.predict(
            X_eng,
            return_uncertainty=True
        )

        # Build results
        results = []
        for i in range(len(predictions)):
            result_dict = {
                'range_nm': float(predictions[i, 0]),
                'endurance_hr': float(predictions[i, 1]),
                'mtow_lbm': float(predictions[i, 2]),
                'cost_usd': float(predictions[i, 3]),
                'wingtip_deflection_in': float(predictions[i, 4])
            }

            # Add uncertainty if requested
            if request_data.return_uncertainty:
                result_dict['range_nm_uncertainty'] = float(uncertainty[i, 0])
                result_dict['endurance_hr_uncertainty'] = float(uncertainty[i, 1])
                result_dict['mtow_lbm_uncertainty'] = float(uncertainty[i, 2])
                result_dict['cost_usd_uncertainty'] = float(uncertainty[i, 3])
                result_dict['wingtip_deflection_in_uncertainty'] = float(uncertainty[i, 4])

            results.append(PredictionResult(**result_dict))

        inference_time_ms = (time.time() - start_time) * 1000

        logger.info(f"Prediction complete in {inference_time_ms:.2f}ms")

        return PredictResponse(
            predictions=results,
            n_designs=len(results),
            inference_time_ms=inference_time_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
