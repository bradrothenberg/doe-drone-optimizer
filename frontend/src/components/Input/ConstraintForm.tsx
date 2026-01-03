import { Box, Button, Slider, Typography } from '@mui/material'
import type { Constraints } from '../../types'

interface ConstraintFormProps {
  constraints: Constraints
  onUpdate: (constraints: Constraints) => void
  isOptimizing: boolean
}

const constraintConfig = [
  {
    key: 'min_range_nm' as keyof Constraints,
    label: 'Minimum Range (nm)',
    min: 0,
    max: 6000,
    step: 100,
    default: 1500
  },
  {
    key: 'max_cost_usd' as keyof Constraints,
    label: 'Maximum Cost ($)',
    min: 0,
    max: 100000,
    step: 1000,
    default: 35000
  },
  {
    key: 'max_mtow_lbm' as keyof Constraints,
    label: 'Maximum MTOW (lbm)',
    min: 0,
    max: 10000,
    step: 100,
    default: 3000
  },
  {
    key: 'min_endurance_hr' as keyof Constraints,
    label: 'Minimum Endurance (hr)',
    min: 0,
    max: 40,
    step: 1,
    default: 8
  }
]

const presets = {
  'Long Range': {
    min_range_nm: 2500,
    max_cost_usd: 50000,
    max_mtow_lbm: 5000,
    min_endurance_hr: 15
  },
  'Low Cost': {
    min_range_nm: 1000,
    max_cost_usd: 25000,
    max_mtow_lbm: 2500,
    min_endurance_hr: 5
  },
  'Balanced': {
    min_range_nm: 1500,
    max_cost_usd: 35000,
    max_mtow_lbm: 3000,
    min_endurance_hr: 8
  }
}

export default function ConstraintForm({ constraints, onUpdate, isOptimizing }: ConstraintFormProps) {
  const handleSliderChange = (key: keyof Constraints) => (
    _event: Event,
    value: number | number[]
  ) => {
    onUpdate({
      ...constraints,
      [key]: value as number
    })
  }

  const handlePreset = (presetName: keyof typeof presets) => {
    onUpdate(presets[presetName])
  }

  const handleClear = () => {
    onUpdate({
      min_range_nm: undefined,
      max_cost_usd: undefined,
      max_mtow_lbm: undefined,
      min_endurance_hr: undefined
    })
  }

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
        Optimization Constraints
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

      {/* Constraint Sliders */}
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
        {constraintConfig.map((config) => {
          const value = constraints[config.key] ?? config.default
          return (
            <Box key={config.key}>
              <Typography
                sx={{
                  fontFamily: 'monospace',
                  fontSize: '0.85em',
                  color: '#666666',
                  mb: 1
                }}
              >
                {config.label}
              </Typography>
              <Slider
                value={value}
                min={config.min}
                max={config.max}
                step={config.step}
                onChange={handleSliderChange(config.key)}
                disabled={isOptimizing}
                valueLabelDisplay="on"
                sx={{
                  color: '#000000',
                  '& .MuiSlider-thumb': {
                    bgcolor: '#000000',
                    border: '2px solid #000000'
                  },
                  '& .MuiSlider-track': {
                    bgcolor: '#000000'
                  },
                  '& .MuiSlider-rail': {
                    bgcolor: '#cccccc'
                  },
                  '& .MuiSlider-valueLabel': {
                    fontFamily: 'monospace',
                    fontSize: '0.75rem',
                    bgcolor: '#000000'
                  }
                }}
              />
              <Typography
                sx={{
                  fontFamily: 'monospace',
                  fontSize: '0.9em',
                  color: '#000000',
                  fontWeight: 'bold',
                  mt: 1
                }}
              >
                {value.toLocaleString()}
              </Typography>
            </Box>
          )
        })}
      </Box>

      <Button
        variant="contained"
        fullWidth
        disabled={isOptimizing}
        onClick={() => onUpdate(constraints)}
        sx={{
          mt: 4,
          bgcolor: '#000000',
          color: '#ffffff',
          fontFamily: 'monospace',
          fontWeight: 'bold',
          fontSize: '1.1em',
          py: 1.5,
          '&:hover': {
            bgcolor: '#333333'
          },
          '&:disabled': {
            bgcolor: '#cccccc',
            color: '#666666'
          }
        }}
      >
        {isOptimizing ? 'OPTIMIZING...' : 'RUN OPTIMIZATION'}
      </Button>
    </Box>
  )
}
