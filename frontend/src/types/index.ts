/**
 * TypeScript type definitions for DOE Drone Optimizer
 */

export interface Constraints {
  // Performance constraints
  min_range_nm?: number
  max_cost_usd?: number
  max_mtow_lbm?: number
  min_endurance_hr?: number
  max_wingtip_deflection_in?: number

  // Taper ratio constraints (tip_chord / root_chord)
  taper_ratio_p1_fixed?: number
  min_taper_ratio_p1?: number
  max_taper_ratio_p1?: number
  taper_ratio_p2_fixed?: number
  min_taper_ratio_p2?: number
  max_taper_ratio_p2?: number

  // Angle constraints - LE Sweep P1
  le_sweep_p1_fixed?: number
  le_sweep_p1_min?: number
  le_sweep_p1_max?: number

  // Angle constraints - LE Sweep P2
  le_sweep_p2_fixed?: number
  le_sweep_p2_min?: number
  le_sweep_p2_max?: number

  // Angle constraints - TE Sweep P1
  te_sweep_p1_fixed?: number
  te_sweep_p1_min?: number
  te_sweep_p1_max?: number

  // Angle constraints - TE Sweep P2
  te_sweep_p2_fixed?: number
  te_sweep_p2_min?: number
  te_sweep_p2_max?: number

  // Root chord ratio constraints (chord / span)
  root_chord_ratio_fixed?: number
  min_root_chord_ratio?: number
  max_root_chord_ratio?: number

  // Panel break constraints (fraction of half-span, 0-1)
  panel_break_fixed?: number
  min_panel_break?: number
  max_panel_break?: number

  // Advanced options
  allow_unrealistic_taper?: boolean
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
  wingtip_deflection_in: number
  range_nm_uncertainty?: number
  endurance_hr_uncertainty?: number
  mtow_lbm_uncertainty?: number
  cost_usd_uncertainty?: number
  wingtip_deflection_in_uncertainty?: number
}

export interface DesignResult extends DesignParameters, PredictionResult {
  uncertainty_range_nm: number
  uncertainty_endurance_hr: number
  uncertainty_mtow_lbm: number
  uncertainty_cost_usd: number
  uncertainty_wingtip_deflection_in: number
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

export type OptimizationDirection = 'minimize' | 'maximize'

export interface OptimizationObjectives {
  range_nm?: OptimizationDirection
  endurance_hr?: OptimizationDirection
  mtow_lbm?: OptimizationDirection
  cost_usd?: OptimizationDirection
  wingtip_deflection_in?: OptimizationDirection
}

export interface OptimizeRequest {
  constraints?: Constraints
  objectives?: OptimizationObjectives
  population_size?: number
  n_generations?: number
  n_designs?: number
}

export interface SensitivityRequest {
  design: DesignParameters
  perturbation_pct?: number
}

export interface InputSensitivity {
  input_name: string
  base_value: number
  perturbed_value: number
  range_nm_delta: number
  endurance_hr_delta: number
  mtow_lbm_delta: number
  cost_usd_delta: number
  wingtip_deflection_in_delta: number
}

export interface SensitivityResponse {
  design: DesignParameters
  perturbation_pct: number
  sensitivities: InputSensitivity[]
  computation_time_ms: number
}
