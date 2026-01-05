import { useState, useEffect } from 'react'
import { Box, Button, Slider, Typography, ToggleButton, ToggleButtonGroup, Checkbox, FormControlLabel, Tooltip } from '@mui/material'
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
    sliderMax: 100,
    sliderStep: 1,
    sliderDefault: 30
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
      max_wingtip_deflection_in: 40
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
      max_wingtip_deflection_in: 25
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
      max_wingtip_deflection_in: 30
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

export default function ConstraintForm({ constraints, objectives, onUpdate, isOptimizing, hasResults }: ConstraintFormProps) {
  // Local state for immediate feedback
  const [localConstraints, setLocalConstraints] = useState<Constraints>(constraints)
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

  // Sync local state when external props change
  useEffect(() => {
    setLocalConstraints(constraints)
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
      allow_unrealistic_taper: false
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
                    {config.constraintType === 'min' ? '≥' : '≤'} {(constraintValue ?? config.sliderDefault).toLocaleString()}
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

      {/* Advanced Options */}
      <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid #e0e0e0' }}>
        <Typography
          sx={{
            fontFamily: 'monospace',
            fontSize: '0.85em',
            color: '#666666',
            mb: 1,
            fontWeight: 500
          }}
        >
          Advanced Options
        </Typography>
        <Tooltip
          title="When enabled, allows wing geometries with expanding chord sections or very thin tips. These designs may be structurally challenging to manufacture."
          placement="right"
          arrow
        >
          <FormControlLabel
            control={
              <Checkbox
                checked={localConstraints.allow_unrealistic_taper ?? false}
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
