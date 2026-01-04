"""
Optimization endpoint for multi-objective drone design
"""

from fastapi import APIRouter, Request, HTTPException
import numpy as np
import logging
import time
from typing import Optional

from app.schemas.optimize import OptimizeRequest, OptimizeResponse, DesignResult
from app.optimization.nsga import run_nsga2_optimization
from app.optimization.constraints import validate_constraints, ConstraintHandler
from app.optimization.pareto import find_pareto_frontiers, select_diverse_subset

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/optimize", response_model=OptimizeResponse, tags=["optimization"])
async def optimize_designs(request_data: OptimizeRequest, request: Request):
    """
    Find Pareto-optimal drone designs using NSGA-II

    Args:
        request_data: Optimization constraints and parameters

    Returns:
        OptimizeResponse with Pareto-optimal designs
    """
    try:
        start_time = time.time()

        # Get models from app state
        model_manager = request.app.state.model_manager

        if not model_manager.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")

        ensemble_model = model_manager.get_ensemble_model()
        feature_engineer = model_manager.get_feature_engineer()

        # Extract constraints
        constraints_dict = {}
        if request_data.constraints:
            if request_data.constraints.min_range_nm is not None:
                constraints_dict['min_range_nm'] = request_data.constraints.min_range_nm
            if request_data.constraints.max_cost_usd is not None:
                constraints_dict['max_cost_usd'] = request_data.constraints.max_cost_usd
            if request_data.constraints.max_mtow_lbm is not None:
                constraints_dict['max_mtow_lbm'] = request_data.constraints.max_mtow_lbm
            if request_data.constraints.min_endurance_hr is not None:
                constraints_dict['min_endurance_hr'] = request_data.constraints.min_endurance_hr
            if request_data.constraints.max_wingtip_deflection_in is not None:
                constraints_dict['max_wingtip_deflection_in'] = request_data.constraints.max_wingtip_deflection_in

        # Extract objectives (optimization directions)
        objectives_dict = None
        if request_data.objectives:
            objectives_dict = request_data.objectives.to_dict()

        logger.info(f"Running optimization with constraints: {constraints_dict}")
        if objectives_dict:
            logger.info(f"Custom objectives: {objectives_dict}")

        # Validate constraints
        is_valid, errors, warnings = validate_constraints(constraints_dict)

        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid constraints: {'; '.join(errors)}"
            )

        # Log warnings but continue
        if warnings:
            logger.warning(f"Constraint warnings: {'; '.join(warnings)}")

        # Attempt optimization
        constraint_relaxation_applied = None
        optimization_error = None

        try:
            # Run NSGA-II optimization
            results = run_nsga2_optimization(
                ensemble_model=ensemble_model,
                feature_engineer=feature_engineer,
                user_constraints=constraints_dict if constraints_dict else None,
                objectives=objectives_dict,
                population_size=request_data.population_size,
                n_generations=request_data.n_generations,
                seed=42
            )

            feasible = True

        except ValueError as e:
            # Optimization failed - try constraint relaxation
            logger.warning(f"Optimization failed: {e}. Attempting constraint relaxation...")

            if constraints_dict:
                # Try balanced relaxation
                handler = ConstraintHandler(constraints_dict)
                relaxation_result = handler.relax_constraints(
                    relaxation_strategy='balanced'
                )

                relaxed_constraints = relaxation_result['relaxed_constraints']
                logger.info(f"Relaxed constraints: {relaxed_constraints}")
                logger.info(f"Relaxation: {relaxation_result['relaxation_description']}")

                try:
                    results = run_nsga2_optimization(
                        ensemble_model=ensemble_model,
                        feature_engineer=feature_engineer,
                        user_constraints=relaxed_constraints,
                        objectives=objectives_dict,
                        population_size=request_data.population_size,
                        n_generations=request_data.n_generations,
                        seed=42
                    )

                    constraint_relaxation_applied = {
                        'original': constraints_dict,
                        'relaxed': relaxed_constraints,
                        'strategy': relaxation_result['strategy'],
                        'description': relaxation_result['relaxation_description']
                    }
                    feasible = False

                except ValueError as e2:
                    # Even relaxation failed
                    optimization_error = str(e2)
                    logger.error(f"Optimization failed even after relaxation: {e2}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Optimization failed: {optimization_error}. Constraints may be too restrictive."
                    )
            else:
                # No constraints to relax
                optimization_error = str(e)
                raise HTTPException(
                    status_code=500,
                    detail=f"Optimization failed: {optimization_error}"
                )

        # Extract Pareto front
        pareto_designs = results['pareto_designs']
        pareto_objectives = results['pareto_objectives']
        uncertainty = results['uncertainty']

        logger.info(f"Found {len(pareto_designs)} Pareto-optimal designs")

        # Select diverse subset if requested
        if request_data.n_designs < len(pareto_designs):
            # Create objectives array for diversity selection
            objectives_array = np.column_stack([
                pareto_objectives['range_nm'],
                pareto_objectives['endurance_hr'],
                pareto_objectives['mtow_lbm'],
                pareto_objectives['cost_usd'],
                pareto_objectives['wingtip_deflection_in']
            ])

            selected_designs, selected_obj, selected_idx = select_diverse_subset(
                designs=pareto_designs,
                objectives=objectives_array,
                n_select=request_data.n_designs,
                method='crowding'
            )

            logger.info(f"Selected {len(selected_idx)} diverse designs from Pareto front")

            # Update results with selected subset
            pareto_designs = selected_designs
            pareto_objectives = {
                'range_nm': selected_obj[:, 0],
                'endurance_hr': selected_obj[:, 1],
                'mtow_lbm': selected_obj[:, 2],
                'cost_usd': selected_obj[:, 3],
                'wingtip_deflection_in': selected_obj[:, 4]
            }
            uncertainty = {
                'range_nm': uncertainty['range_nm'][selected_idx],
                'endurance_hr': uncertainty['endurance_hr'][selected_idx],
                'mtow_lbm': uncertainty['mtow_lbm'][selected_idx],
                'cost_usd': uncertainty['cost_usd'][selected_idx],
                'wingtip_deflection_in': uncertainty['wingtip_deflection_in'][selected_idx]
            }

        # Build response
        design_results = []
        for i in range(len(pareto_designs)):
            design = pareto_designs[i]
            design_result = DesignResult(
                loa=float(design[0]),
                span=float(design[1]),
                le_sweep_p1=float(design[2]),
                le_sweep_p2=float(design[3]),
                te_sweep_p1=float(design[4]),
                te_sweep_p2=float(design[5]),
                panel_break=float(design[6]),
                range_nm=float(pareto_objectives['range_nm'][i]),
                endurance_hr=float(pareto_objectives['endurance_hr'][i]),
                mtow_lbm=float(pareto_objectives['mtow_lbm'][i]),
                cost_usd=float(pareto_objectives['cost_usd'][i]),
                wingtip_deflection_in=float(pareto_objectives['wingtip_deflection_in'][i]),
                uncertainty_range_nm=float(uncertainty['range_nm'][i]),
                uncertainty_endurance_hr=float(uncertainty['endurance_hr'][i]),
                uncertainty_mtow_lbm=float(uncertainty['mtow_lbm'][i]),
                uncertainty_cost_usd=float(uncertainty['cost_usd'][i]),
                uncertainty_wingtip_deflection_in=float(uncertainty['wingtip_deflection_in'][i])
            )
            design_results.append(design_result)

        optimization_time_s = time.time() - start_time

        logger.info(f"Optimization complete in {optimization_time_s:.2f}s")

        return OptimizeResponse(
            pareto_designs=design_results,
            n_pareto=len(design_results),
            feasible=feasible,
            optimization_time_s=optimization_time_s,
            constraint_relaxation=constraint_relaxation_applied,
            warnings=warnings if warnings else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
