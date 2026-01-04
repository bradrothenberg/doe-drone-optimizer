import { useState, useEffect } from 'react'
import { Box, Typography, FormControl, Select, MenuItem, CircularProgress } from '@mui/material'
import Plot from 'react-plotly.js'
import { computeSensitivity } from '../../services/api'
import type { DesignResult, InputSensitivity } from '../../types'

interface SensitivityChartProps {
  design: DesignResult
}

type OutputMetric = 'range_nm' | 'endurance_hr' | 'mtow_lbm' | 'cost_usd' | 'wingtip_deflection_in'

const OUTPUT_LABELS: Record<OutputMetric, string> = {
  range_nm: 'Range (nm)',
  endurance_hr: 'Endurance (hr)',
  mtow_lbm: 'MTOW (lbm)',
  cost_usd: 'Cost ($)',
  wingtip_deflection_in: 'Tip Deflection (in)'
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

  // Sort by absolute impact
  const sortedData = [...sensitivities].sort((a, b) => Math.abs(getDelta(b)) - Math.abs(getDelta(a)))

  const colors = sortedData.map(s => getDelta(s) >= 0 ? '#4caf50' : '#f44336')

  return (
    <Box
      sx={{
        bgcolor: '#f5f5f5',
        border: '2px solid #cccccc',
        p: 2,
        mt: 3
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
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
          Sensitivity Analysis (10% perturbation)
        </Typography>

        <FormControl size="small" sx={{ minWidth: 180 }}>
          <Select
            value={selectedOutput}
            onChange={(e) => setSelectedOutput(e.target.value as OutputMetric)}
            sx={{ fontFamily: 'monospace', fontSize: '0.9em' }}
          >
            {Object.entries(OUTPUT_LABELS).map(([key, label]) => (
              <MenuItem key={key} value={key} sx={{ fontFamily: 'monospace' }}>
                {label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      <Plot
        data={[
          {
            type: 'bar',
            orientation: 'h',
            x: sortedData.map(s => getDelta(s)),
            y: sortedData.map(s => s.input_name),
            marker: { color: colors },
            hovertemplate: '%{y}: %{x:.2f}<extra></extra>'
          }
        ]}
        layout={{
          title: {
            text: `Impact on ${OUTPUT_LABELS[selectedOutput]}`,
            font: { family: 'monospace', size: 14 }
          },
          xaxis: {
            title: `Change in ${OUTPUT_LABELS[selectedOutput]}`,
            titlefont: { size: 11 },
            tickfont: { family: 'monospace', size: 10 },
            zeroline: true,
            zerolinecolor: '#000000',
            zerolinewidth: 2,
            gridcolor: '#e0e0e0'
          },
          yaxis: {
            tickfont: { family: 'monospace', size: 10 },
            automargin: true
          },
          margin: { l: 150, r: 30, t: 50, b: 50 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: '#ffffff',
          showlegend: false
        }}
        config={{
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d'],
          toImageButtonOptions: {
            format: 'png',
            filename: 'sensitivity_chart',
            height: 400,
            width: 600,
            scale: 2
          }
        }}
        style={{ width: '100%', height: '350px' }}
      />

      <Typography
        sx={{
          fontFamily: 'monospace',
          fontSize: '0.8em',
          color: '#666666',
          mt: 1,
          textAlign: 'center'
        }}
      >
        Green = positive impact, Red = negative impact when input increases by 10%
      </Typography>
    </Box>
  )
}
