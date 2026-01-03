import Plot from 'react-plotly.js'
import { Box, Typography } from '@mui/material'
import { chartLayout, chartConfig } from './chartConfig'
import type { DesignResult } from '../../types'

interface ParetoChart3DProps {
  data: DesignResult[]
}

export default function ParetoChart3D({ data }: ParetoChart3DProps) {
  return (
    <Box
      sx={{
        bgcolor: '#f5f5f5',
        border: '2px solid #cccccc',
        p: 2
      }}
    >
      <Typography
        variant="h6"
        sx={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontWeight: 600,
          color: '#000000',
          mb: 2,
          fontSize: '1.1em',
          borderBottom: '2px solid #000000',
          pb: 1
        }}
      >
        Pareto 3D: Range × Endurance × MTOW ({data.length} optimal)
      </Typography>

      <Plot
        data={[
          {
            x: data.map(d => d.mtow_lbm),
            y: data.map(d => d.range_nm),
            z: data.map(d => d.endurance_hr),
            mode: 'markers',
            type: 'scatter3d',
            name: 'Pareto Optimal',
            marker: {
              size: 5,
              color: data.map(d => d.cost_usd),
              colorscale: [[0, '#e3f2fd'], [1, '#1565c0']],
              colorbar: {
                title: 'Cost ($)',
                thickness: 15,
                tickfont: { size: 9 },
                tickformat: ',.0f'
              },
              symbol: 'diamond',
              line: { color: '#000000', width: 0.5 }
            },
            hovertemplate:
              '<b>Design</b><br>' +
              'MTOW: %{x:.0f} lbm<br>' +
              'Range: %{y:.0f} nm<br>' +
              'Endurance: %{z:.1f} hr<extra></extra>'
          }
        ]}
        layout={{
          ...chartLayout,
          title: { text: '', font: { size: 13 } },
          scene: {
            xaxis: {
              title: 'MTOW (lbm)',
              titlefont: { size: 10 },
              tickfont: { size: 8 },
              gridcolor: '#e0e0e0',
              linecolor: '#cccccc'
            },
            yaxis: {
              title: 'Range (nm)',
              titlefont: { size: 10 },
              tickfont: { size: 8 },
              gridcolor: '#e0e0e0',
              linecolor: '#cccccc'
            },
            zaxis: {
              title: 'Endurance (hr)',
              titlefont: { size: 10 },
              tickfont: { size: 8 },
              gridcolor: '#e0e0e0',
              linecolor: '#cccccc'
            },
            camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } },
            bgcolor: '#ffffff'
          },
          margin: { t: 40, r: 10, b: 10, l: 10 },
          showlegend: false
        }}
        config={chartConfig}
        style={{ width: '100%', height: '600px' }}
      />
    </Box>
  )
}
