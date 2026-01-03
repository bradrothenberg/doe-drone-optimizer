/**
 * TypeScript type definitions for DOE Drone Optimizer
 */

export interface Constraints {
  min_range_nm?: number
  max_cost_usd?: number
  max_mtow_lbm?: number
  min_endurance_hr?: number
}

export interface DesignParameters {
  loa: number
  span: number
  le_sweep_p1: number
  le_sweep_p2: number
  te_sweep_p1: number
  te_sweep_p2: number
  panel_break: number
}

export interface PredictionResult {
  range_nm: number
  endurance_hr: number
  mtow_lbm: number
  cost_usd: number
  range_nm_uncertainty?: number
  endurance_hr_uncertainty?: number
  mtow_lbm_uncertainty?: number
  cost_usd_uncertainty?: number
}

export interface DesignResult extends DesignParameters, PredictionResult {
  uncertainty_range_nm: number
  uncertainty_endurance_hr: number
  uncertainty_mtow_lbm: number
  uncertainty_cost_usd: number
}

export interface OptimizeResponse {
  pareto_designs: DesignResult[]
  n_pareto: number
  feasible: boolean
  optimization_time_s: number
  constraint_relaxation?: {
    original: Constraints
    relaxed: Constraints
    strategy: string
    description: string
  }
  warnings?: string[]
}

export interface PredictRequest {
  designs: DesignParameters[]
  return_uncertainty?: boolean
}

export interface PredictResponse {
  predictions: PredictionResult[]
  n_designs: number
  inference_time_ms: number
  model_info?: Record<string, unknown>
}

export interface OptimizeRequest {
  constraints?: Constraints
  population_size?: number
  n_generations?: number
  n_designs?: number
}
