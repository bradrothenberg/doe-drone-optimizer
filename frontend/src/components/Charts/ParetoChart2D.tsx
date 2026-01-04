import Plot from 'react-plotly.js'
import { Box, Typography } from '@mui/material'
import { chartLayout, chartConfig, paretoMarker, paretoLine } from './chartConfig'
import type { DesignResult } from '../../types'

interface ParetoChart2DProps {
  data: DesignResult[]
  xKey: keyof DesignResult
  yKey: keyof DesignResult
  title: string
  xLabel: string
  yLabel: string
}

export default function ParetoChart2D({
  data,
  xKey,
  yKey,
  title,
  xLabel,
  yLabel
}: ParetoChart2DProps) {
  // Sort data for line connection (ascending x)
  const sortedData = [...data].sort((a, b) => (a[xKey] as number) - (b[xKey] as number))

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
        {title} ({data.length} optimal)
      </Typography>

      <Plot
        data={[
          {
            x: sortedData.map(d => d[xKey]),
            y: sortedData.map(d => d[yKey]),
            mode: 'markers+lines',
            type: 'scatter',
            name: 'Pareto Optimal',
            marker: paretoMarker,
            line: paretoLine,
            hovertemplate:
              `<b>Design</b><br>` +
              `${xLabel}: %{x:.0f}<br>` +
              `${yLabel}: %{y:.0f}<extra></extra>`
          }
        ]}
        layout={{
          ...chartLayout,
          title: { text: '', font: { size: 13 } },
          xaxis: {
            ...chartLayout.xaxis,
            title: xLabel,
            tickformat: xKey === 'cost_usd' ? ',.0f' : undefined
          },
          yaxis: {
            ...chartLayout.yaxis,
            title: yLabel,
            tickformat: yKey === 'cost_usd' ? ',.0f' : undefined
          },
          showlegend: false
        }}
        config={chartConfig}
        style={{ width: '100%', height: '400px' }}
      />
    </Box>
  )
}
