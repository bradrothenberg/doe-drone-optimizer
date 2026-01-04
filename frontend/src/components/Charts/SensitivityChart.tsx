import { useState, useEffect } from 'react'
import { Box, Typography, CircularProgress, ToggleButton, ToggleButtonGroup } from '@mui/material'
import Plot from 'react-plotly.js'
import { computeSensitivity } from '../../services/api'
import type { DesignResult, InputSensitivity } from '../../types'

interface SensitivityChartProps {
  design: DesignResult
}

type OutputMetric = 'range_nm' | 'endurance_hr' | 'mtow_lbm' | 'cost_usd' | 'wingtip_deflection_in'

const OUTPUT_CONFIG: Record<OutputMetric, { label: string; unit: string; color: string }> = {
  range_nm: { label: 'Range', unit: 'nm', color: '#1565c0' },
  endurance_hr: { label: 'Endurance', unit: 'hr', color: '#2e7d32' },
  mtow_lbm: { label: 'MTOW', unit: 'lbm', color: '#c62828' },
  cost_usd: { label: 'Cost', unit: '$', color: '#6a1b9a' },
  wingtip_deflection_in: { label: 'Deflection', unit: 'in', color: '#e65100' }
}

export default function SensitivityChart({ design }: SensitivityChartProps) {
  const [sensitivities, setSensitivities] = useState<InputSensitivity[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedOutput, setSelectedOutput] = useState<OutputMetric>('range_nm')

  useEffect(() => {
    const fetchSensitivity = async () => {
      setLoading(true)
      setError(null)
      try {
        const response = await computeSensitivity({
          design: {
            loa: design.loa,
            span: design.span,
            le_sweep_p1: design.le_sweep_p1,
            le_sweep_p2: design.le_sweep_p2,
            te_sweep_p1: design.te_sweep_p1,
            te_sweep_p2: design.te_sweep_p2,
            panel_break: design.panel_break
          },
          perturbation_pct: 10
        })
        setSensitivities(response.sensitivities)
      } catch (err) {
        setError('Failed to compute sensitivity')
        console.error(err)
      } finally {
        setLoading(false)
      }
    }

    fetchSensitivity()
  }, [design])

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
        <CircularProgress size={24} />
        <Typography sx={{ ml: 2, fontFamily: 'monospace' }}>Computing sensitivity...</Typography>
      </Box>
    )
  }

  if (error || !sensitivities) {
    return (
      <Box sx={{ p: 2, bgcolor: '#ffebee', border: '1px solid #ef5350', fontFamily: 'monospace' }}>
        {error || 'No sensitivity data available'}
      </Box>
    )
  }

  // Get delta values for selected output
  const getDelta = (s: InputSensitivity): number => {
    const key = `${selectedOutput}_delta` as keyof InputSensitivity
    return s[key] as number
  }

  // Sort by absolute impact for tornado chart (largest at top)
  const sortedData = [...sensitivities].sort((a, b) => Math.abs(getDelta(b)) - Math.abs(getDelta(a)))

  // Create tornado chart data - bars extend from center (0) in both directions
  const config = OUTPUT_CONFIG[selectedOutput]
  const positiveValues = sortedData.map(s => Math.max(0, getDelta(s)))
  const negativeValues = sortedData.map(s => Math.min(0, getDelta(s)))

  // Find max absolute value for symmetric axis
  const maxAbsValue = Math.max(...sortedData.map(s => Math.abs(getDelta(s))))

  return (
    <Box
      sx={{
        bgcolor: '#f5f5f5',
        border: '2px solid #cccccc',
        p: 2,
        mt: 3
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2, flexWrap: 'wrap', gap: 2 }}>
        <Typography
          variant="h6"
          sx={{
            fontFamily: "'IBM Plex Mono', monospace",
            fontWeight: 600,
            color: '#000000',
            fontSize: '1.1em',
            borderBottom: '2px solid #000000',
            pb: 1
          }}
        >
          Sensitivity Analysis (±10% perturbation)
        </Typography>

        {/* Interactive output metric selector - click to switch */}
        <ToggleButtonGroup
          value={selectedOutput}
          exclusive
          onChange={(_, value) => value && setSelectedOutput(value)}
          size="small"
          sx={{
            '& .MuiToggleButton-root': {
              fontFamily: 'monospace',
              fontSize: '0.75em',
              padding: '4px 10px',
              textTransform: 'none',
              border: '1px solid #cccccc',
              '&.Mui-selected': {
                fontWeight: 'bold'
              }
            }
          }}
        >
          {Object.entries(OUTPUT_CONFIG).map(([key, { label, color }]) => (
            <ToggleButton
              key={key}
              value={key}
              sx={{
                '&.Mui-selected': {
                  bgcolor: `${color}20`,
                  borderColor: color,
                  color: color,
                  '&:hover': {
                    bgcolor: `${color}30`
                  }
                }
              }}
            >
              {label}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
      </Box>

      {/* Tornado Chart */}
      <Plot
        data={[
          // Positive changes (right side)
          {
            type: 'bar',
            orientation: 'h',
            x: positiveValues,
            y: sortedData.map(s => s.input_name),
            marker: { color: '#4caf50' },
            name: 'Increase',
            hovertemplate: '%{y}: +%{x:.2f} ' + config.unit + '<extra></extra>'
          },
          // Negative changes (left side)
          {
            type: 'bar',
            orientation: 'h',
            x: negativeValues,
            y: sortedData.map(s => s.input_name),
            marker: { color: '#f44336' },
            name: 'Decrease',
            hovertemplate: '%{y}: %{x:.2f} ' + config.unit + '<extra></extra>'
          }
        ]}
        layout={{
          title: {
            text: `<b>Impact on ${config.label}</b> (${config.unit})`,
            font: { family: 'Courier New', size: 14, color: config.color },
            y: 0.98
          },
          barmode: 'overlay',
          xaxis: {
            title: `Change in ${config.label} (${config.unit})`,
            titlefont: { size: 11, family: 'monospace' },
            tickfont: { family: 'monospace', size: 10 },
            zeroline: true,
            zerolinecolor: '#000000',
            zerolinewidth: 2,
            gridcolor: '#e0e0e0',
            range: [-maxAbsValue * 1.1, maxAbsValue * 1.1] // Symmetric range
          },
          yaxis: {
            tickfont: { family: 'monospace', size: 10 },
            automargin: true
          },
          margin: { l: 140, r: 30, t: 50, b: 50 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: '#ffffff',
          showlegend: true,
          legend: {
            x: 1,
            y: 1,
            xanchor: 'right',
            font: { family: 'monospace', size: 10 }
          }
        }}
        config={{
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d'],
          toImageButtonOptions: {
            format: 'png',
            filename: `sensitivity_${selectedOutput}`,
            height: 400,
            width: 700,
            scale: 2
          }
        }}
        style={{ width: '100%', height: '350px' }}
      />

      {/* Input summary table */}
      <Box sx={{ mt: 2, overflowX: 'auto' }}>
        <Typography
          sx={{
            fontFamily: 'monospace',
            fontSize: '0.85em',
            fontWeight: 'bold',
            mb: 1,
            color: '#333333'
          }}
        >
          Top Influencing Inputs (+10% of range)
        </Typography>
        <Box
          component="table"
          sx={{
            width: '100%',
            borderCollapse: 'collapse',
            fontFamily: 'monospace',
            fontSize: '0.75em',
            '& th, & td': {
              padding: '4px 8px',
              textAlign: 'left',
              borderBottom: '1px solid #e0e0e0'
            },
            '& th': {
              fontWeight: 'bold',
              bgcolor: '#f0f0f0'
            }
          }}
        >
          <thead>
            <tr>
              <th>Input</th>
              <th style={{ textAlign: 'right' }}>Base</th>
              <th style={{ textAlign: 'right' }}>Perturbed</th>
              <th style={{ textAlign: 'right' }}>Δ {config.label}</th>
            </tr>
          </thead>
          <tbody>
            {sortedData.slice(0, 4).map((s, idx) => {
              const delta = getDelta(s)
              return (
                <tr key={idx}>
                  <td>{s.input_name}</td>
                  <td style={{ textAlign: 'right' }}>{s.base_value.toFixed(1)}</td>
                  <td style={{ textAlign: 'right' }}>{s.perturbed_value.toFixed(1)}</td>
                  <td style={{
                    textAlign: 'right',
                    color: delta >= 0 ? '#4caf50' : '#f44336',
                    fontWeight: 'bold'
                  }}>
                    {delta >= 0 ? '+' : ''}{delta.toFixed(2)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </Box>
      </Box>

      <Typography
        sx={{
          fontFamily: 'monospace',
          fontSize: '0.75em',
          color: '#666666',
          mt: 2,
          textAlign: 'center'
        }}
      >
        Click output buttons above to see how each input affects different metrics
      </Typography>
    </Box>
  )
}
