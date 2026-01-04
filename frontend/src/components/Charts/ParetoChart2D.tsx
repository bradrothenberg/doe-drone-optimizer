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
  selectedIndex?: number | null
  onSelectDesign?: (index: number) => void
}

export default function ParetoChart2D({
  data,
  xKey,
  yKey,
  title,
  xLabel,
  yLabel,
  selectedIndex,
  onSelectDesign
}: ParetoChart2DProps) {
  // Sort data for line connection (ascending x) and track original indices
  const indexedData = data.map((d, i) => ({ ...d, originalIndex: i }))
  const sortedData = [...indexedData].sort((a, b) => (a[xKey] as number) - (b[xKey] as number))

  // Create marker colors - highlight selected point
  const markerColors = sortedData.map(d =>
    d.originalIndex === selectedIndex ? '#1565c0' : paretoMarker.color
  )
  const markerSizes = sortedData.map(d =>
    d.originalIndex === selectedIndex ? 14 : paretoMarker.size
  )

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleClick = (event: any) => {
    if (event.points && event.points.length > 0 && onSelectDesign) {
      const pointIndex = event.points[0].pointIndex
      const originalIndex = sortedData[pointIndex].originalIndex
      onSelectDesign(originalIndex)
    }
  }

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
          mb: 1,
          fontSize: '1.1em',
          borderBottom: '2px solid #000000',
          pb: 1
        }}
      >
        {title} ({data.length} optimal)
      </Typography>
      <Typography
        sx={{
          fontFamily: 'monospace',
          fontSize: '0.8em',
          color: '#666666',
          mb: 1
        }}
      >
        Click any data point to view planform
      </Typography>

      <Plot
        data={[
          {
            x: sortedData.map(d => d[xKey]),
            y: sortedData.map(d => d[yKey]),
            mode: 'markers+lines',
            type: 'scatter',
            name: 'Pareto Optimal',
            marker: {
              ...paretoMarker,
              color: markerColors,
              size: markerSizes
            },
            line: paretoLine,
            hovertemplate:
              `<b>Design</b><br>` +
              `${xLabel}: %{x:.0f}<br>` +
              `${yLabel}: %{y:.0f}<extra></extra>`,
            customdata: sortedData.map(d => d.originalIndex)
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
          showlegend: false,
          hovermode: 'closest'
        }}
        config={chartConfig}
        style={{ width: '100%', height: '400px' }}
        onClick={handleClick}
      />
    </Box>
  )
}
