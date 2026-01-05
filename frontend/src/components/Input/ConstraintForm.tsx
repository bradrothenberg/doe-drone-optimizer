import { useState, useEffect } from 'react'
import { Box, Button, Slider, Typography, ToggleButton, ToggleButtonGroup, Checkbox, FormControlLabel, Tooltip, Collapse, TextField, Select, MenuItem, InputAdornment } from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import ExpandLessIcon from '@mui/icons-material/ExpandLess'
import type { Constraints, OptimizationObjectives, OptimizationDirection } from '../../types'

interface ConstraintFormProps {
  constraints: Constraints
  objectives: OptimizationObjectives
  onUpdate: (constraints: Constraints, objectives: OptimizationObjectives) => void
  isOptimizing: boolean
  hasResults: boolean
}

// Unified metric configuration - each metric can be MIN, MAX, or LIMIT (with hard constraint)
type MetricMode = 'minimize' | 'maximize' | 'limit'

// Numeric constraint keys only (excludes allow_unrealistic_taper)
type NumericConstraintKey = 'min_range_nm' | 'max_cost_usd' | 'max_mtow_lbm' | 'min_endurance_hr' | 'max_wingtip_deflection_in'

interface MetricConfig {
  key: string
  label: string
  unit: string
  defaultMode: MetricMode
  // Constraint info when in target mode
  constraintKey: NumericConstraintKey
  constraintType: 'min' | 'max'  // Whether the target sets a min or max constraint
  sliderMin: number
  sliderMax: number
  sliderStep: number
  sliderDefault: number
}

const metricConfigs: MetricConfig[] = [
  {
    key: 'range_nm',
    label: 'Range',
    unit: 'nm',
    defaultMode: 'maximize',
    constraintKey: 'min_range_nm',
    constraintType: 'min',
    sliderMin: 0,
    sliderMax: 6000,
    sliderStep: 100,
    sliderDefault: 1500
  },
  {
    key: 'endurance_hr',
    label: 'Endurance',
    unit: 'hr',
    defaultMode: 'maximize',
    constraintKey: 'min_endurance_hr',
    constraintType: 'min',
    sliderMin: 0,
    sliderMax: 40,
    sliderStep: 1,
    sliderDefault: 8
  },
  {
    key: 'mtow_lbm',
    label: 'MTOW',
    unit: 'lbm',
    defaultMode: 'minimize',
    constraintKey: 'max_mtow_lbm',
    constraintType: 'max',
    sliderMin: 0,
    sliderMax: 10000,
    sliderStep: 100,
    sliderDefault: 3000
  },
  {
    key: 'cost_usd',
    label: 'Cost',
    unit: '$',
    defaultMode: 'minimize',
    constraintKey: 'max_cost_usd',
    constraintType: 'max',
    sliderMin: 0,
    sliderMax: 100000,
    sliderStep: 1000,
    sliderDefault: 35000
  },
  {
    key: 'wingtip_deflection_in',
    label: 'Deflection',
    unit: 'in',
    defaultMode: 'minimize',
    constraintKey: 'max_wingtip_deflection_in',
    constraintType: 'max',
    sliderMin: 0,
    sliderMax: 2,
    sliderStep: 0.05,
    sliderDefault: 1.0
  }
]

// Preset configurations with modes and enabled state
interface PresetConfig {
  modes: Record<string, MetricMode>
  enabled: Record<string, boolean>
  constraints: Constraints
}

const presets: Record<string, PresetConfig> = {
  'Long Range': {
    modes: {
      range_nm: 'maximize',
      endurance_hr: 'limit',
      mtow_lbm: 'limit',
      cost_usd: 'limit',
      wingtip_deflection_in: 'limit'
    },
    enabled: {
      range_nm: true,
      endurance_hr: true,
      mtow_lbm: true,
      cost_usd: true,
      wingtip_deflection_in: true
    },
    constraints: {
      min_range_nm: undefined,
      min_endurance_hr: 15,
      max_mtow_lbm: 5000,
      max_cost_usd: 50000,
      max_wingtip_deflection_in: 1.5
    }
  },
  'Low Cost': {
    modes: {
      range_nm: 'limit',
      endurance_hr: 'limit',
      mtow_lbm: 'minimize',
      cost_usd: 'minimize',
      wingtip_deflection_in: 'limit'
    },
    enabled: {
      range_nm: true,
      endurance_hr: true,
      mtow_lbm: true,
      cost_usd: true,
      wingtip_deflection_in: true
    },
    constraints: {
      min_range_nm: 1000,
      min_endurance_hr: 5,
      max_mtow_lbm: undefined,
      max_cost_usd: undefined,
      max_wingtip_deflection_in: 0.8
    }
  },
  'Balanced': {
    modes: {
      range_nm: 'limit',
      endurance_hr: 'limit',
      mtow_lbm: 'limit',
      cost_usd: 'limit',
      wingtip_deflection_in: 'limit'
    },
    enabled: {
      range_nm: true,
      endurance_hr: true,
      mtow_lbm: true,
      cost_usd: true,
      wingtip_deflection_in: true
    },
    constraints: {
      min_range_nm: 1500,
      min_endurance_hr: 8,
      max_mtow_lbm: 3000,
      max_cost_usd: 35000,
      max_wingtip_deflection_in: 1.0
    }
  }
}

// Derive the current mode for a metric based on constraints and objectives
function getModeForMetric(
  config: MetricConfig,
  constraints: Constraints,
  objectives: OptimizationObjectives
): MetricMode {
  const constraintValue = constraints[config.constraintKey]
  if (constraintValue !== undefined) {
    return 'limit'
  }
  const objectiveKey = config.key as keyof OptimizationObjectives
  return objectives[objectiveKey] || config.defaultMode as OptimizationDirection
}

// Default objectives (what the backend uses if not specified)
const defaultObjectives: OptimizationObjectives = {
  range_nm: 'maximize',
  endurance_hr: 'maximize',
  mtow_lbm: 'minimize',
  cost_usd: 'minimize',
  wingtip_deflection_in: 'minimize'
}

// Constraint mode type (applies to angles, taper ratios, root chord, panel break)
type ConstraintMode = 'fixed' | 'range'

// Generic constraint configuration for UI
interface ConstraintConfig {
  key: string
  label: string
  fixedKey: keyof Constraints
  minKey: keyof Constraints
  maxKey: keyof Constraints
  defaultValue: number
  defaultMin: number
  defaultMax: number
  min: number
  max: number
  step: number
  unit?: string
}

// Angle configurations
const angleConfigs: ConstraintConfig[] = [
  { key: 'le_sweep_p1', label: 'LE Sweep P1', fixedKey: 'le_sweep_p1_fixed', minKey: 'le_sweep_p1_min', maxKey: 'le_sweep_p1_max', defaultValue: 30, defaultMin: 0, defaultMax: 65, min: 0, max: 65, step: 1, unit: '°' },
  { key: 'le_sweep_p2', label: 'LE Sweep P2', fixedKey: 'le_sweep_p2_fixed', minKey: 'le_sweep_p2_min', maxKey: 'le_sweep_p2_max', defaultValue: 40, defaultMin: 0, defaultMax: 65, min: 0, max: 65, step: 1, unit: '°' },
  { key: 'te_sweep_p1', label: 'TE Sweep P1', fixedKey: 'te_sweep_p1_fixed', minKey: 'te_sweep_p1_min', maxKey: 'te_sweep_p1_max', defaultValue: 0, defaultMin: -60, defaultMax: 60, min: -60, max: 60, step: 1, unit: '°' },
  { key: 'te_sweep_p2', label: 'TE Sweep P2', fixedKey: 'te_sweep_p2_fixed', minKey: 'te_sweep_p2_min', maxKey: 'te_sweep_p2_max', defaultValue: 10, defaultMin: -60, defaultMax: 60, min: -60, max: 60, step: 1, unit: '°' },
]

// Taper ratio configurations
const taperConfigs: ConstraintConfig[] = [
  { key: 'taper_ratio_p1', label: 'Panel 1', fixedKey: 'taper_ratio_p1_fixed', minKey: 'min_taper_ratio_p1', maxKey: 'max_taper_ratio_p1', defaultValue: 0.5, defaultMin: 0.1, defaultMax: 1.0, min: 0.1, max: 1.5, step: 0.05 },
  { key: 'taper_ratio_p2', label: 'Panel 2', fixedKey: 'taper_ratio_p2_fixed', minKey: 'min_taper_ratio_p2', maxKey: 'max_taper_ratio_p2', defaultValue: 0.5, defaultMin: 0.1, defaultMax: 0.8, min: 0.1, max: 1.5, step: 0.05 },
]

// Root chord ratio configuration
const rootChordConfig: ConstraintConfig = {
  key: 'root_chord_ratio', label: 'Chord/Span', fixedKey: 'root_chord_ratio_fixed', minKey: 'min_root_chord_ratio', maxKey: 'max_root_chord_ratio', defaultValue: 0.9, defaultMin: 0.8, defaultMax: 1.2, min: 0.3, max: 2.0, step: 0.05
}

// Panel break configuration
const panelBreakConfig: ConstraintConfig = {
  key: 'panel_break', label: 'Panel Break %', fixedKey: 'panel_break_fixed', minKey: 'min_panel_break', maxKey: 'max_panel_break', defaultValue: 0.4, defaultMin: 0.2, defaultMax: 0.5, min: 0.1, max: 0.65, step: 0.05
}

export default function ConstraintForm({ constraints, objectives, onUpdate, isOptimizing, hasResults }: ConstraintFormProps) {
  // Local state for immediate feedback
  // Default allow_unrealistic_taper to true for better optimization results
  const [localConstraints, setLocalConstraints] = useState<Constraints>({
    ...constraints,
    allow_unrealistic_taper: constraints.allow_unrealistic_taper ?? true
  })
  const [localObjectives, setLocalObjectives] = useState<OptimizationObjectives>({ ...defaultObjectives, ...objectives })
  const [localModes, setLocalModes] = useState<Record<string, MetricMode>>(() => {
    const modes: Record<string, MetricMode> = {}
    metricConfigs.forEach(config => {
      modes[config.key] = getModeForMetric(config, constraints, objectives)
    })
    return modes
  })
  const [enabledMetrics, setEnabledMetrics] = useState<Record<string, boolean>>(() => {
    const enabled: Record<string, boolean> = {}
    metricConfigs.forEach(config => {
      enabled[config.key] = true // All enabled by default
    })
    return enabled
  })
  const [hasChanges, setHasChanges] = useState(false)

  // Geometric constraints state
  const [geometricExpanded, setGeometricExpanded] = useState(true)

  // Taper ratio enabled state and modes (default ON with Range)
  const [taperEnabled, setTaperEnabled] = useState<Record<string, boolean>>({
    taper_ratio_p1: true,
    taper_ratio_p2: true
  })
  const [taperModes, setTaperModes] = useState<Record<string, ConstraintMode>>({
    taper_ratio_p1: 'range',
    taper_ratio_p2: 'range'
  })

  // Angle enabled states and modes (default OFF with Fixed)
  const [angleEnabled, setAngleEnabled] = useState<Record<string, boolean>>({
    le_sweep_p1: false,
    le_sweep_p2: false,
    te_sweep_p1: false,
    te_sweep_p2: false
  })
  const [angleModes, setAngleModes] = useState<Record<string, ConstraintMode>>({
    le_sweep_p1: 'fixed',
    le_sweep_p2: 'fixed',
    te_sweep_p1: 'fixed',
    te_sweep_p2: 'fixed'
  })

  // Root chord ratio enabled state and mode (default OFF with Range)
  const [rootChordEnabled, setRootChordEnabled] = useState(false)
  const [rootChordMode, setRootChordMode] = useState<ConstraintMode>('range')

  // Panel break enabled state and mode (default OFF with Range)
  const [panelBreakEnabled, setPanelBreakEnabled] = useState(false)
  const [panelBreakMode, setPanelBreakMode] = useState<ConstraintMode>('range')

  // Advanced options expanded state (default collapsed)
  const [advancedExpanded, setAdvancedExpanded] = useState(false)

  // Sync local state when external props change
  useEffect(() => {
    setLocalConstraints({
      ...constraints,
      allow_unrealistic_taper: constraints.allow_unrealistic_taper ?? true
    })
    setLocalObjectives({ ...defaultObjectives, ...objectives })
    const modes: Record<string, MetricMode> = {}
    metricConfigs.forEach(config => {
      modes[config.key] = getModeForMetric(config, constraints, objectives)
    })
    setLocalModes(modes)
    setHasChanges(false)
  }, [constraints, objectives])

  const checkForChanges = (
    newConstraints: Constraints,
    newObjectives: OptimizationObjectives,
    newModes: Record<string, MetricMode>,
    newEnabled?: Record<string, boolean>
  ) => {
    // Check if anything changed from the submitted state
    const constraintsChanged = JSON.stringify(newConstraints) !== JSON.stringify(constraints)
    const objectivesChanged = JSON.stringify(newObjectives) !== JSON.stringify(objectives)
    const modesChanged = metricConfigs.some(config => {
      const originalMode = getModeForMetric(config, constraints, objectives)
      return newModes[config.key] !== originalMode
    })
    const enabledChanged = newEnabled ? JSON.stringify(newEnabled) !== JSON.stringify(enabledMetrics) : false
    setHasChanges(constraintsChanged || objectivesChanged || modesChanged || enabledChanged)
  }

  const handleEnabledChange = (metricKey: string, checked: boolean) => {
    const newEnabled = { ...enabledMetrics, [metricKey]: checked }
    setEnabledMetrics(newEnabled)

    // If disabling, clear the constraint and objective for this metric
    if (!checked) {
      const config = metricConfigs.find(c => c.key === metricKey)!
      const objectiveKey = metricKey as keyof OptimizationObjectives
      const newConstraints = { ...localConstraints, [config.constraintKey]: undefined }
      const newObjectives = { ...localObjectives }
      delete newObjectives[objectiveKey]
      setLocalConstraints(newConstraints)
      setLocalObjectives(newObjectives)
      checkForChanges(newConstraints, newObjectives, localModes, newEnabled)
    } else {
      checkForChanges(localConstraints, localObjectives, localModes, newEnabled)
    }
  }

  const handleModeChange = (metricKey: string, newMode: MetricMode | null) => {
    if (newMode === null) return // Don't allow deselection

    const config = metricConfigs.find(c => c.key === metricKey)!
    const objectiveKey = metricKey as keyof OptimizationObjectives

    const newModes = { ...localModes, [metricKey]: newMode }
    let newConstraints = { ...localConstraints }
    let newObjectives = { ...localObjectives }

    if (newMode === 'limit') {
      // Set a hard constraint with the default value
      newConstraints = {
        ...newConstraints,
        [config.constraintKey]: config.sliderDefault
      }
      // Remove from objectives (will use default direction)
      newObjectives = { ...newObjectives }
      delete newObjectives[objectiveKey]
    } else {
      // Clear the constraint
      newConstraints = {
        ...newConstraints,
        [config.constraintKey]: undefined
      }
      // Set the objective direction
      newObjectives = {
        ...newObjectives,
        [objectiveKey]: newMode as OptimizationDirection
      }
    }

    setLocalModes(newModes)
    setLocalConstraints(newConstraints)
    setLocalObjectives(newObjectives)
    checkForChanges(newConstraints, newObjectives, newModes)
  }

  const handleSliderChange = (constraintKey: keyof Constraints) => (
    _event: Event,
    value: number | number[]
  ) => {
    const newConstraints = {
      ...localConstraints,
      [constraintKey]: value as number
    }
    setLocalConstraints(newConstraints)
    checkForChanges(newConstraints, localObjectives, localModes)
  }

  const handlePreset = (presetName: keyof typeof presets) => {
    const preset = presets[presetName]
    setLocalConstraints(preset.constraints)
    setLocalModes(preset.modes)
    setEnabledMetrics(preset.enabled)
    // Build objectives from modes (only for min/max modes on enabled metrics)
    const newObjectives: OptimizationObjectives = {}
    metricConfigs.forEach(config => {
      const mode = preset.modes[config.key]
      const isEnabled = preset.enabled[config.key]
      if (isEnabled && mode !== 'limit') {
        const key = config.key as keyof OptimizationObjectives
        newObjectives[key] = mode as OptimizationDirection
      }
    })
    setLocalObjectives(newObjectives)
    checkForChanges(preset.constraints, newObjectives, preset.modes, preset.enabled)
  }

  const handleClear = () => {
    const clearedConstraints: Constraints = {
      min_range_nm: undefined,
      max_cost_usd: undefined,
      max_mtow_lbm: undefined,
      min_endurance_hr: undefined,
      max_wingtip_deflection_in: undefined,
      allow_unrealistic_taper: true  // Default to true for better optimization results
    }
    const defaultModes: Record<string, MetricMode> = {}
    const allEnabled: Record<string, boolean> = {}
    metricConfigs.forEach(config => {
      defaultModes[config.key] = config.defaultMode
      allEnabled[config.key] = true
    })
    setLocalConstraints(clearedConstraints)
    setLocalObjectives({ ...defaultObjectives })
    setLocalModes(defaultModes)
    setEnabledMetrics(allEnabled)
    checkForChanges(clearedConstraints, defaultObjectives, defaultModes, allEnabled)
  }

  const handleAllowUnrealisticTaperChange = (checked: boolean) => {
    const newConstraints = { ...localConstraints, allow_unrealistic_taper: checked }
    setLocalConstraints(newConstraints)
    checkForChanges(newConstraints, localObjectives, localModes)
  }

  // Generic constraint value change handler
  const handleConstraintValueChange = (key: keyof Constraints, value: string) => {
    const numValue = value === '' ? undefined : parseFloat(value)
    const newConstraints = { ...localConstraints, [key]: numValue }
    setLocalConstraints(newConstraints)
    checkForChanges(newConstraints, localObjectives, localModes)
  }

  // Taper ratio handlers
  const handleTaperEnabledChange = (taperKey: string, checked: boolean) => {
    setTaperEnabled(prev => ({ ...prev, [taperKey]: checked }))
    const config = taperConfigs.find(c => c.key === taperKey)!
    if (!checked) {
      const newConstraints = {
        ...localConstraints,
        [config.fixedKey]: undefined,
        [config.minKey]: undefined,
        [config.maxKey]: undefined
      }
      setLocalConstraints(newConstraints)
      checkForChanges(newConstraints, localObjectives, localModes)
    } else {
      // Set default values based on current mode
      const mode = taperModes[taperKey]
      let newConstraints: Constraints
      if (mode === 'fixed') {
        newConstraints = {
          ...localConstraints,
          [config.fixedKey]: config.defaultValue,
          [config.minKey]: undefined,
          [config.maxKey]: undefined
        }
      } else {
        newConstraints = {
          ...localConstraints,
          [config.fixedKey]: undefined,
          [config.minKey]: config.defaultMin,
          [config.maxKey]: config.defaultMax
        }
      }
      setLocalConstraints(newConstraints)
      checkForChanges(newConstraints, localObjectives, localModes)
    }
  }

  const handleTaperModeChange = (taperKey: string, mode: ConstraintMode) => {
    setTaperModes(prev => ({ ...prev, [taperKey]: mode }))
    const config = taperConfigs.find(c => c.key === taperKey)!
    if (taperEnabled[taperKey]) {
      let newConstraints: Constraints
      if (mode === 'fixed') {
        newConstraints = {
          ...localConstraints,
          [config.fixedKey]: config.defaultValue,
          [config.minKey]: undefined,
          [config.maxKey]: undefined
        }
      } else {
        newConstraints = {
          ...localConstraints,
          [config.fixedKey]: undefined,
          [config.minKey]: config.defaultMin,
          [config.maxKey]: config.defaultMax
        }
      }
      setLocalConstraints(newConstraints)
      checkForChanges(newConstraints, localObjectives, localModes)
    }
  }

  // Angle handlers
  const handleAngleEnabledChange = (angleKey: string, checked: boolean) => {
    setAngleEnabled(prev => ({ ...prev, [angleKey]: checked }))
    const config = angleConfigs.find(c => c.key === angleKey)!
    if (!checked) {
      const newConstraints = {
        ...localConstraints,
        [config.fixedKey]: undefined,
        [config.minKey]: undefined,
        [config.maxKey]: undefined
      }
      setLocalConstraints(newConstraints)
      checkForChanges(newConstraints, localObjectives, localModes)
    } else {
      // Set default values based on current mode
      const mode = angleModes[angleKey]
      let newConstraints: Constraints
      if (mode === 'fixed') {
        newConstraints = {
          ...localConstraints,
          [config.fixedKey]: config.defaultValue,
          [config.minKey]: undefined,
          [config.maxKey]: undefined
        }
      } else {
        newConstraints = {
          ...localConstraints,
          [config.fixedKey]: undefined,
          [config.minKey]: config.defaultMin,
          [config.maxKey]: config.defaultMax
        }
      }
      setLocalConstraints(newConstraints)
      checkForChanges(newConstraints, localObjectives, localModes)
    }
  }

  const handleAngleModeChange = (angleKey: string, mode: ConstraintMode) => {
    setAngleModes(prev => ({ ...prev, [angleKey]: mode }))
    const config = angleConfigs.find(c => c.key === angleKey)!
    if (angleEnabled[angleKey]) {
      let newConstraints: Constraints
      if (mode === 'fixed') {
        newConstraints = {
          ...localConstraints,
          [config.fixedKey]: config.defaultValue,
          [config.minKey]: undefined,
          [config.maxKey]: undefined
        }
      } else {
        newConstraints = {
          ...localConstraints,
          [config.fixedKey]: undefined,
          [config.minKey]: config.defaultMin,
          [config.maxKey]: config.defaultMax
        }
      }
      setLocalConstraints(newConstraints)
      checkForChanges(newConstraints, localObjectives, localModes)
    }
  }

  // Root chord ratio handlers
  const handleRootChordEnabledChange = (checked: boolean) => {
    setRootChordEnabled(checked)
    const config = rootChordConfig
    if (!checked) {
      const newConstraints = {
        ...localConstraints,
        [config.fixedKey]: undefined,
        [config.minKey]: undefined,
        [config.maxKey]: undefined
      }
      setLocalConstraints(newConstraints)
      checkForChanges(newConstraints, localObjectives, localModes)
    } else {
      let newConstraints: Constraints
      if (rootChordMode === 'fixed') {
        newConstraints = {
          ...localConstraints,
          [config.fixedKey]: config.defaultValue,
          [config.minKey]: undefined,
          [config.maxKey]: undefined
        }
      } else {
        newConstraints = {
          ...localConstraints,
          [config.fixedKey]: undefined,
          [config.minKey]: config.defaultMin,
          [config.maxKey]: config.defaultMax
        }
      }
      setLocalConstraints(newConstraints)
      checkForChanges(newConstraints, localObjectives, localModes)
    }
  }

  const handleRootChordModeChange = (mode: ConstraintMode) => {
    setRootChordMode(mode)
    const config = rootChordConfig
    if (rootChordEnabled) {
      let newConstraints: Constraints
      if (mode === 'fixed') {
        newConstraints = {
          ...localConstraints,
          [config.fixedKey]: config.defaultValue,
          [config.minKey]: undefined,
          [config.maxKey]: undefined
        }
      } else {
        newConstraints = {
          ...localConstraints,
          [config.fixedKey]: undefined,
          [config.minKey]: config.defaultMin,
          [config.maxKey]: config.defaultMax
        }
      }
      setLocalConstraints(newConstraints)
      checkForChanges(newConstraints, localObjectives, localModes)
    }
  }

  // Panel break handlers
  const handlePanelBreakEnabledChange = (checked: boolean) => {
    setPanelBreakEnabled(checked)
    const config = panelBreakConfig
    if (!checked) {
      const newConstraints = {
        ...localConstraints,
        [config.fixedKey]: undefined,
        [config.minKey]: undefined,
        [config.maxKey]: undefined
      }
      setLocalConstraints(newConstraints)
      checkForChanges(newConstraints, localObjectives, localModes)
    } else {
      let newConstraints: Constraints
      if (panelBreakMode === 'fixed') {
        newConstraints = {
          ...localConstraints,
          [config.fixedKey]: config.defaultValue,
          [config.minKey]: undefined,
          [config.maxKey]: undefined
        }
      } else {
        newConstraints = {
          ...localConstraints,
          [config.fixedKey]: undefined,
          [config.minKey]: config.defaultMin,
          [config.maxKey]: config.defaultMax
        }
      }
      setLocalConstraints(newConstraints)
      checkForChanges(newConstraints, localObjectives, localModes)
    }
  }

  const handlePanelBreakModeChange = (mode: ConstraintMode) => {
    setPanelBreakMode(mode)
    const config = panelBreakConfig
    if (panelBreakEnabled) {
      let newConstraints: Constraints
      if (mode === 'fixed') {
        newConstraints = {
          ...localConstraints,
          [config.fixedKey]: config.defaultValue,
          [config.minKey]: undefined,
          [config.maxKey]: undefined
        }
      } else {
        newConstraints = {
          ...localConstraints,
          [config.fixedKey]: undefined,
          [config.minKey]: config.defaultMin,
          [config.maxKey]: config.defaultMax
        }
      }
      setLocalConstraints(newConstraints)
      checkForChanges(newConstraints, localObjectives, localModes)
    }
  }

  const handleRunOptimization = () => {
    // Only include constraints and objectives for enabled metrics
    const filteredConstraints: Constraints = { ...localConstraints }
    const filteredObjectives: OptimizationObjectives = { ...localObjectives }

    metricConfigs.forEach(config => {
      if (!enabledMetrics[config.key]) {
        // Clear constraint for disabled metric
        filteredConstraints[config.constraintKey] = undefined
        // Clear objective for disabled metric
        const objectiveKey = config.key as keyof OptimizationObjectives
        delete filteredObjectives[objectiveKey]
      }
    })

    onUpdate(filteredConstraints, filteredObjectives)
    setHasChanges(false)
  }

  // Determine button text and style based on state
  const getButtonConfig = () => {
    if (isOptimizing) {
      return {
        text: 'OPTIMIZING...',
        bgcolor: '#cccccc',
        disabled: true
      }
    }
    if (hasChanges) {
      return {
        text: 'RUN OPTIMIZATION',
        bgcolor: '#1565c0',
        disabled: false
      }
    }
    if (hasResults) {
      return {
        text: 'OPTIMIZATION COMPLETE',
        bgcolor: '#2e7d32',
        disabled: true
      }
    }
    return {
      text: 'RUN OPTIMIZATION',
      bgcolor: '#000000',
      disabled: false
    }
  }

  const buttonConfig = getButtonConfig()

  return (
    <Box
      sx={{
        bgcolor: '#f5f5f5',
        border: '2px solid #cccccc',
        p: 3
      }}
    >
      <Typography
        variant="h6"
        sx={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontWeight: 600,
          color: '#000000',
          mb: 2,
          fontSize: '1.2em',
          borderBottom: '2px solid #000000',
          pb: 1
        }}
      >
        Optimization Setup
      </Typography>

      {/* Preset Buttons */}
      <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap' }}>
        {Object.keys(presets).map((preset) => (
          <Button
            key={preset}
            variant="outlined"
            onClick={() => handlePreset(preset as keyof typeof presets)}
            disabled={isOptimizing}
            sx={{
              borderColor: '#000000',
              color: '#000000',
              fontFamily: 'monospace',
              '&:hover': {
                borderColor: '#000000',
                bgcolor: '#e0e0e0'
              }
            }}
          >
            {preset}
          </Button>
        ))}
        <Button
          variant="outlined"
          onClick={handleClear}
          disabled={isOptimizing}
          sx={{
            borderColor: '#666666',
            color: '#666666',
            fontFamily: 'monospace',
            '&:hover': {
              borderColor: '#666666',
              bgcolor: '#e0e0e0'
            }
          }}
        >
          Clear All
        </Button>
      </Box>

      {/* Metric Controls */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        {metricConfigs.map((config) => {
          const mode = localModes[config.key]
          const constraintValue = localConstraints[config.constraintKey]
          const showSlider = mode === 'limit'
          const isEnabled = enabledMetrics[config.key]

          return (
            <Box
              key={config.key}
              sx={{
                display: 'grid',
                gridTemplateColumns: showSlider ? '32px 110px 200px 1fr 80px' : '32px 110px 200px 1fr',
                gap: 2,
                alignItems: 'center',
                py: 0.5,
                borderBottom: '1px solid #e0e0e0',
                opacity: isEnabled ? 1 : 0.4,
                bgcolor: isEnabled ? 'transparent' : '#f9f9f9'
              }}
            >
              {/* Enable Checkbox */}
              <Checkbox
                checked={isEnabled}
                onChange={(e) => handleEnabledChange(config.key, e.target.checked)}
                disabled={isOptimizing}
                size="small"
                sx={{
                  p: 0,
                  color: '#666666',
                  '&.Mui-checked': {
                    color: '#1565c0'
                  }
                }}
              />

              {/* Metric Label */}
              <Typography
                sx={{
                  fontFamily: 'monospace',
                  fontSize: '0.9em',
                  color: isEnabled ? '#000000' : '#888888',
                  fontWeight: 500
                }}
              >
                {config.label}
                <Typography
                  component="span"
                  sx={{
                    fontFamily: 'monospace',
                    fontSize: '0.8em',
                    color: '#666666',
                    ml: 0.5
                  }}
                >
                  ({config.unit})
                </Typography>
              </Typography>

              {/* Mode Toggle */}
              <ToggleButtonGroup
                value={mode}
                exclusive
                onChange={(_, value) => handleModeChange(config.key, value)}
                disabled={isOptimizing || !isEnabled}
                size="small"
                sx={{
                  '& .MuiToggleButton-root': {
                    fontFamily: 'monospace',
                    fontSize: '0.75em',
                    px: 1.5,
                    py: 0.5,
                    borderColor: '#cccccc',
                    color: '#666666',
                    '&.Mui-selected': {
                      color: '#ffffff',
                      '&:hover': {
                        opacity: 0.9
                      }
                    },
                    '&.Mui-selected[value="minimize"]': {
                      bgcolor: isEnabled ? '#2196f3' : '#b0bec5',
                    },
                    '&.Mui-selected[value="maximize"]': {
                      bgcolor: isEnabled ? '#4caf50' : '#b0bec5',
                    },
                    '&.Mui-selected[value="limit"]': {
                      bgcolor: isEnabled ? '#ff9800' : '#b0bec5',
                    },
                    '&:hover': {
                      bgcolor: '#e0e0e0'
                    }
                  }
                }}
              >
                <ToggleButton value="minimize">MIN</ToggleButton>
                <ToggleButton value="maximize">MAX</ToggleButton>
                <ToggleButton value="limit">LIMIT</ToggleButton>
              </ToggleButtonGroup>

              {/* Slider (only shown in limit mode) */}
              {showSlider ? (
                <>
                  <Slider
                    value={constraintValue ?? config.sliderDefault}
                    min={config.sliderMin}
                    max={config.sliderMax}
                    step={config.sliderStep}
                    onChange={handleSliderChange(config.constraintKey)}
                    disabled={isOptimizing || !isEnabled}
                    valueLabelDisplay="auto"
                    sx={{
                      color: isEnabled ? '#ff9800' : '#bdbdbd',
                      '& .MuiSlider-thumb': {
                        bgcolor: isEnabled ? '#ff9800' : '#bdbdbd',
                        border: `2px solid ${isEnabled ? '#e65100' : '#9e9e9e'}`
                      },
                      '& .MuiSlider-track': {
                        bgcolor: isEnabled ? '#ff9800' : '#bdbdbd'
                      },
                      '& .MuiSlider-rail': {
                        bgcolor: '#cccccc'
                      },
                      '& .MuiSlider-valueLabel': {
                        fontFamily: 'monospace',
                        fontSize: '0.75rem',
                        bgcolor: isEnabled ? '#ff9800' : '#9e9e9e'
                      }
                    }}
                  />
                  <Typography
                    sx={{
                      fontFamily: 'monospace',
                      fontSize: '0.9em',
                      color: isEnabled ? '#000000' : '#888888',
                      fontWeight: 'bold',
                      textAlign: 'right'
                    }}
                  >
                    {config.constraintType === 'min' ? '≥' : '≤'} {
                      config.sliderStep < 1
                        ? (constraintValue ?? config.sliderDefault).toFixed(2)
                        : (constraintValue ?? config.sliderDefault).toLocaleString()
                    }
                  </Typography>
                </>
              ) : (
                <Typography
                  sx={{
                    fontFamily: 'monospace',
                    fontSize: '0.85em',
                    color: '#888888',
                    fontStyle: 'italic'
                  }}
                >
                  {isEnabled
                    ? (mode === 'minimize' ? 'Find lowest possible value' : 'Find highest possible value')
                    : 'Not included in optimization'
                  }
                </Typography>
              )}
            </Box>
          )
        })}
      </Box>

      {/* Geometric Constraints - Collapsible Section */}
      <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid #e0e0e0' }}>
        <Box
          onClick={() => setGeometricExpanded(!geometricExpanded)}
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            cursor: 'pointer',
            mb: 1,
            '&:hover': { bgcolor: '#f0f0f0' },
            p: 0.5,
            mx: -0.5,
            borderRadius: 1
          }}
        >
          <Typography
            sx={{
              fontFamily: 'monospace',
              fontSize: '0.95em',
              color: '#333333',
              fontWeight: 600
            }}
          >
            Geometric Constraints
          </Typography>
          {geometricExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        </Box>

        <Collapse in={geometricExpanded}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, py: 1 }}>

            {/* LOA (Root Chord) Section - moved to top */}
            <Box>
              <Typography
                sx={{
                  fontFamily: 'monospace',
                  fontSize: '0.8em',
                  color: '#666666',
                  mb: 1,
                  fontWeight: 500
                }}
              >
                LOA (ratio to span)
              </Typography>

              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: '32px 100px 80px 1fr',
                  gap: 1,
                  alignItems: 'center',
                  opacity: rootChordEnabled ? 1 : 0.5
                }}
              >
                <Checkbox
                  checked={rootChordEnabled}
                  onChange={(e) => handleRootChordEnabledChange(e.target.checked)}
                  disabled={isOptimizing}
                  size="small"
                  sx={{ p: 0 }}
                />
                <Typography sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>
                  LOA/Span ratio
                </Typography>
                <Select
                  size="small"
                  value={rootChordMode}
                  onChange={(e) => handleRootChordModeChange(e.target.value as ConstraintMode)}
                  disabled={isOptimizing || !rootChordEnabled}
                  sx={{
                    fontFamily: 'monospace',
                    fontSize: '0.8em',
                    '& .MuiSelect-select': { py: 0.5 }
                  }}
                >
                  <MenuItem value="fixed" sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>Fixed</MenuItem>
                  <MenuItem value="range" sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>Range</MenuItem>
                </Select>

                {rootChordMode === 'fixed' ? (
                  <TextField
                    size="small"
                    type="number"
                    value={localConstraints.root_chord_ratio_fixed ?? ''}
                    onChange={(e) => handleConstraintValueChange('root_chord_ratio_fixed', e.target.value)}
                    disabled={isOptimizing || !rootChordEnabled}
                    placeholder="Value"
                    inputProps={{ min: rootChordConfig.min, max: rootChordConfig.max, step: rootChordConfig.step }}
                    sx={{
                      width: 100,
                      '& .MuiInputBase-input': { fontFamily: 'monospace', fontSize: '0.85em', py: 0.5 }
                    }}
                  />
                ) : (
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                    <TextField
                      size="small"
                      type="number"
                      value={localConstraints.min_root_chord_ratio ?? ''}
                      onChange={(e) => handleConstraintValueChange('min_root_chord_ratio', e.target.value)}
                      disabled={isOptimizing || !rootChordEnabled}
                      placeholder="Min"
                      inputProps={{ min: rootChordConfig.min, max: rootChordConfig.max, step: rootChordConfig.step }}
                      sx={{
                        width: 70,
                        '& .MuiInputBase-input': { fontFamily: 'monospace', fontSize: '0.85em', py: 0.5 }
                      }}
                    />
                    <Typography sx={{ fontFamily: 'monospace', fontSize: '0.85em', color: '#666' }}>—</Typography>
                    <TextField
                      size="small"
                      type="number"
                      value={localConstraints.max_root_chord_ratio ?? ''}
                      onChange={(e) => handleConstraintValueChange('max_root_chord_ratio', e.target.value)}
                      disabled={isOptimizing || !rootChordEnabled}
                      placeholder="Max"
                      inputProps={{ min: rootChordConfig.min, max: rootChordConfig.max, step: rootChordConfig.step }}
                      sx={{
                        width: 70,
                        '& .MuiInputBase-input': { fontFamily: 'monospace', fontSize: '0.85em', py: 0.5 }
                      }}
                    />
                  </Box>
                )}
              </Box>
            </Box>

            {/* Taper Ratios Section */}
            <Box>
              <Typography
                sx={{
                  fontFamily: 'monospace',
                  fontSize: '0.8em',
                  color: '#666666',
                  mb: 1,
                  fontWeight: 500
                }}
              >
                Taper Ratios (tip chord / root chord)
              </Typography>

              {taperConfigs.map((config) => (
                <Box
                  key={config.key}
                  sx={{
                    display: 'grid',
                    gridTemplateColumns: '32px 100px 80px 1fr',
                    gap: 1,
                    alignItems: 'center',
                    mb: 1,
                    opacity: taperEnabled[config.key] ? 1 : 0.5
                  }}
                >
                  <Checkbox
                    checked={taperEnabled[config.key]}
                    onChange={(e) => handleTaperEnabledChange(config.key, e.target.checked)}
                    disabled={isOptimizing}
                    size="small"
                    sx={{ p: 0 }}
                  />
                  <Typography sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>
                    {config.label}
                  </Typography>
                  <Select
                    size="small"
                    value={taperModes[config.key]}
                    onChange={(e) => handleTaperModeChange(config.key, e.target.value as ConstraintMode)}
                    disabled={isOptimizing || !taperEnabled[config.key]}
                    sx={{
                      fontFamily: 'monospace',
                      fontSize: '0.8em',
                      '& .MuiSelect-select': { py: 0.5 }
                    }}
                  >
                    <MenuItem value="fixed" sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>Fixed</MenuItem>
                    <MenuItem value="range" sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>Range</MenuItem>
                  </Select>

                  {taperModes[config.key] === 'fixed' ? (
                    <TextField
                      size="small"
                      type="number"
                      value={localConstraints[config.fixedKey] ?? ''}
                      onChange={(e) => handleConstraintValueChange(config.fixedKey, e.target.value)}
                      disabled={isOptimizing || !taperEnabled[config.key]}
                      placeholder="Value"
                      inputProps={{ min: config.min, max: config.max, step: config.step }}
                      sx={{
                        width: 100,
                        '& .MuiInputBase-input': { fontFamily: 'monospace', fontSize: '0.85em', py: 0.5 }
                      }}
                    />
                  ) : (
                    <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                      <TextField
                        size="small"
                        type="number"
                        value={localConstraints[config.minKey] ?? ''}
                        onChange={(e) => handleConstraintValueChange(config.minKey, e.target.value)}
                        disabled={isOptimizing || !taperEnabled[config.key]}
                        placeholder="Min"
                        inputProps={{ min: config.min, max: config.max, step: config.step }}
                        sx={{
                          width: 70,
                          '& .MuiInputBase-input': { fontFamily: 'monospace', fontSize: '0.85em', py: 0.5 }
                        }}
                      />
                      <Typography sx={{ fontFamily: 'monospace', fontSize: '0.85em', color: '#666' }}>—</Typography>
                      <TextField
                        size="small"
                        type="number"
                        value={localConstraints[config.maxKey] ?? ''}
                        onChange={(e) => handleConstraintValueChange(config.maxKey, e.target.value)}
                        disabled={isOptimizing || !taperEnabled[config.key]}
                        placeholder="Max"
                        inputProps={{ min: config.min, max: config.max, step: config.step }}
                        sx={{
                          width: 70,
                          '& .MuiInputBase-input': { fontFamily: 'monospace', fontSize: '0.85em', py: 0.5 }
                        }}
                      />
                    </Box>
                  )}
                </Box>
              ))}
            </Box>

            {/* Sweep Angles Section */}
            <Box>
              <Typography
                sx={{
                  fontFamily: 'monospace',
                  fontSize: '0.8em',
                  color: '#666666',
                  mb: 1,
                  fontWeight: 500
                }}
              >
                Sweep Angles
              </Typography>

              {angleConfigs.map((config) => (
                <Box
                  key={config.key}
                  sx={{
                    display: 'grid',
                    gridTemplateColumns: '32px 100px 80px 1fr',
                    gap: 1,
                    alignItems: 'center',
                    mb: 1,
                    opacity: angleEnabled[config.key] ? 1 : 0.5
                  }}
                >
                  <Checkbox
                    checked={angleEnabled[config.key]}
                    onChange={(e) => handleAngleEnabledChange(config.key, e.target.checked)}
                    disabled={isOptimizing}
                    size="small"
                    sx={{ p: 0 }}
                  />
                  <Typography sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>
                    {config.label}
                  </Typography>
                  <Select
                    size="small"
                    value={angleModes[config.key]}
                    onChange={(e) => handleAngleModeChange(config.key, e.target.value as ConstraintMode)}
                    disabled={isOptimizing || !angleEnabled[config.key]}
                    sx={{
                      fontFamily: 'monospace',
                      fontSize: '0.8em',
                      '& .MuiSelect-select': { py: 0.5 }
                    }}
                  >
                    <MenuItem value="fixed" sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>Fixed</MenuItem>
                    <MenuItem value="range" sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>Range</MenuItem>
                  </Select>

                  {angleModes[config.key] === 'fixed' ? (
                    <TextField
                      size="small"
                      type="number"
                      value={localConstraints[config.fixedKey] ?? ''}
                      onChange={(e) => handleConstraintValueChange(config.fixedKey, e.target.value)}
                      disabled={isOptimizing || !angleEnabled[config.key]}
                      placeholder="Value"
                      inputProps={{ min: config.min, max: config.max, step: config.step }}
                      InputProps={{
                        endAdornment: <InputAdornment position="end">{config.unit}</InputAdornment>
                      }}
                      sx={{
                        width: 100,
                        '& .MuiInputBase-input': { fontFamily: 'monospace', fontSize: '0.85em', py: 0.5 }
                      }}
                    />
                  ) : (
                    <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                      <TextField
                        size="small"
                        type="number"
                        value={localConstraints[config.minKey] ?? ''}
                        onChange={(e) => handleConstraintValueChange(config.minKey, e.target.value)}
                        disabled={isOptimizing || !angleEnabled[config.key]}
                        placeholder="Min"
                        inputProps={{ min: config.min, max: config.max, step: config.step }}
                        sx={{
                          width: 70,
                          '& .MuiInputBase-input': { fontFamily: 'monospace', fontSize: '0.85em', py: 0.5 }
                        }}
                      />
                      <Typography sx={{ fontFamily: 'monospace', fontSize: '0.85em', color: '#666' }}>—</Typography>
                      <TextField
                        size="small"
                        type="number"
                        value={localConstraints[config.maxKey] ?? ''}
                        onChange={(e) => handleConstraintValueChange(config.maxKey, e.target.value)}
                        disabled={isOptimizing || !angleEnabled[config.key]}
                        placeholder="Max"
                        inputProps={{ min: config.min, max: config.max, step: config.step }}
                        InputProps={{
                          endAdornment: <InputAdornment position="end">{config.unit}</InputAdornment>
                        }}
                        sx={{
                          width: 85,
                          '& .MuiInputBase-input': { fontFamily: 'monospace', fontSize: '0.85em', py: 0.5 }
                        }}
                      />
                    </Box>
                  )}
                </Box>
              ))}
            </Box>

            {/* Panel Break Section */}
            <Box>
              <Typography
                sx={{
                  fontFamily: 'monospace',
                  fontSize: '0.8em',
                  color: '#666666',
                  mb: 1,
                  fontWeight: 500
                }}
              >
                Panel Break (fraction of half-span)
              </Typography>

              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: '32px 100px 80px 1fr',
                  gap: 1,
                  alignItems: 'center',
                  opacity: panelBreakEnabled ? 1 : 0.5
                }}
              >
                <Checkbox
                  checked={panelBreakEnabled}
                  onChange={(e) => handlePanelBreakEnabledChange(e.target.checked)}
                  disabled={isOptimizing}
                  size="small"
                  sx={{ p: 0 }}
                />
                <Typography sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>
                  Panel Break %
                </Typography>
                <Select
                  size="small"
                  value={panelBreakMode}
                  onChange={(e) => handlePanelBreakModeChange(e.target.value as ConstraintMode)}
                  disabled={isOptimizing || !panelBreakEnabled}
                  sx={{
                    fontFamily: 'monospace',
                    fontSize: '0.8em',
                    '& .MuiSelect-select': { py: 0.5 }
                  }}
                >
                  <MenuItem value="fixed" sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>Fixed</MenuItem>
                  <MenuItem value="range" sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>Range</MenuItem>
                </Select>

                {panelBreakMode === 'fixed' ? (
                  <TextField
                    size="small"
                    type="number"
                    value={localConstraints.panel_break_fixed ?? ''}
                    onChange={(e) => handleConstraintValueChange('panel_break_fixed', e.target.value)}
                    disabled={isOptimizing || !panelBreakEnabled}
                    placeholder="Value"
                    inputProps={{ min: panelBreakConfig.min, max: panelBreakConfig.max, step: panelBreakConfig.step }}
                    sx={{
                      width: 100,
                      '& .MuiInputBase-input': { fontFamily: 'monospace', fontSize: '0.85em', py: 0.5 }
                    }}
                  />
                ) : (
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                    <TextField
                      size="small"
                      type="number"
                      value={localConstraints.min_panel_break ?? ''}
                      onChange={(e) => handleConstraintValueChange('min_panel_break', e.target.value)}
                      disabled={isOptimizing || !panelBreakEnabled}
                      placeholder="Min"
                      inputProps={{ min: panelBreakConfig.min, max: panelBreakConfig.max, step: panelBreakConfig.step }}
                      sx={{
                        width: 70,
                        '& .MuiInputBase-input': { fontFamily: 'monospace', fontSize: '0.85em', py: 0.5 }
                      }}
                    />
                    <Typography sx={{ fontFamily: 'monospace', fontSize: '0.85em', color: '#666' }}>—</Typography>
                    <TextField
                      size="small"
                      type="number"
                      value={localConstraints.max_panel_break ?? ''}
                      onChange={(e) => handleConstraintValueChange('max_panel_break', e.target.value)}
                      disabled={isOptimizing || !panelBreakEnabled}
                      placeholder="Max"
                      inputProps={{ min: panelBreakConfig.min, max: panelBreakConfig.max, step: panelBreakConfig.step }}
                      sx={{
                        width: 70,
                        '& .MuiInputBase-input': { fontFamily: 'monospace', fontSize: '0.85em', py: 0.5 }
                      }}
                    />
                  </Box>
                )}
              </Box>
            </Box>

          </Box>
        </Collapse>
      </Box>

      {/* Advanced Options - Collapsible Section */}
      <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid #e0e0e0' }}>
        <Box
          onClick={() => setAdvancedExpanded(!advancedExpanded)}
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            cursor: 'pointer',
            mb: 1,
            '&:hover': { bgcolor: '#f0f0f0' },
            p: 0.5,
            mx: -0.5,
            borderRadius: 1
          }}
        >
          <Typography
            sx={{
              fontFamily: 'monospace',
              fontSize: '0.85em',
              color: '#666666',
              fontWeight: 500
            }}
          >
            Advanced Options
          </Typography>
          {advancedExpanded ? <ExpandLessIcon sx={{ color: '#666666' }} /> : <ExpandMoreIcon sx={{ color: '#666666' }} />}
        </Box>

        <Collapse in={advancedExpanded}>
          <Tooltip
            title="When enabled, allows wing geometries with expanding chord sections or very thin tips. These designs may be structurally challenging to manufacture."
            placement="right"
            arrow
          >
            <FormControlLabel
              control={
                <Checkbox
                  checked={localConstraints.allow_unrealistic_taper ?? true}
                  onChange={(e) => handleAllowUnrealisticTaperChange(e.target.checked)}
                  disabled={isOptimizing}
                  size="small"
                  sx={{
                    color: '#666666',
                    '&.Mui-checked': {
                      color: '#ff5722'
                    }
                  }}
                />
              }
              label={
                <Typography
                  sx={{
                    fontFamily: 'monospace',
                    fontSize: '0.85em',
                    color: localConstraints.allow_unrealistic_taper ? '#ff5722' : '#666666'
                  }}
                >
                  Allow unrealistic taper ratios
                </Typography>
              }
            />
          </Tooltip>
        </Collapse>
      </Box>

      <Button
        variant="contained"
        fullWidth
        disabled={buttonConfig.disabled}
        onClick={handleRunOptimization}
        sx={{
          mt: 4,
          bgcolor: buttonConfig.bgcolor,
          color: '#ffffff',
          fontFamily: 'monospace',
          fontWeight: 'bold',
          fontSize: '1.1em',
          py: 1.5,
          '&:hover': {
            bgcolor: hasChanges ? '#0d47a1' : '#333333'
          },
          '&:disabled': {
            bgcolor: buttonConfig.bgcolor,
            color: '#ffffff'
          }
        }}
      >
        {buttonConfig.text}
      </Button>

      {hasChanges && hasResults && (
        <Typography
          sx={{
            fontFamily: 'monospace',
            fontSize: '0.85em',
            color: '#1565c0',
            mt: 1,
            textAlign: 'center'
          }}
        >
          Settings changed - click to re-run optimization
        </Typography>
      )}
    </Box>
  )
}
