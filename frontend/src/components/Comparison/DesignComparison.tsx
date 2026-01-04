import { useMemo, useState } from 'react'
import { Box, Typography, IconButton, Tabs, Tab } from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import Plot from 'react-plotly.js'
import type { DesignResult } from '../../types'
import SensitivityChart from '../Charts/SensitivityChart'

/**
 * PlanformOverlay - Renders multiple wing planforms overlaid on the same SVG
 * for side-by-side shape comparison
 */
interface PlanformOverlayProps {
  designs: DesignResult[]
  colors: string[]
  width?: number
  height?: number
}

function PlanformOverlay({ designs, colors, width = 400, height = 300 }: PlanformOverlayProps) {
  // SVG viewBox dimensions
  const viewWidth = 500
  const viewHeight = 450
  const padding = 40
  const centerX = viewWidth / 2
  const centerY = viewHeight / 2

  // Find the largest span and LOA to scale all designs to the same reference
  const maxSpan = Math.max(...designs.map(d => d.span))
  const maxLoa = Math.max(...designs.map(d => d.loa))

  // Available drawing area
  const drawWidth = viewWidth - 2 * padding
  const drawHeight = viewHeight - 2 * padding

  // Global scale based on largest design
  const scaleX = drawWidth / maxSpan
  const scaleY = drawHeight / maxLoa
  const globalScale = Math.min(scaleX, scaleY) * 0.9

  // Generate wing outline for a single design
  const getWingPath = (design: DesignResult, scale: number) => {
    const { loa, span, le_sweep_p1, le_sweep_p2, te_sweep_p1, te_sweep_p2, panel_break } = design

    const halfSpan = span / 2
    const breakSpan = halfSpan * panel_break
    const remainingSpan = halfSpan - breakSpan

    // Convert sweep angles to Y offsets
    const leOffset1 = Math.tan((le_sweep_p1 * Math.PI) / 180) * breakSpan * scale
    const leOffset2 = leOffset1 + Math.tan((le_sweep_p2 * Math.PI) / 180) * remainingSpan * scale
    let teOffset1 = Math.tan((te_sweep_p1 * Math.PI) / 180) * breakSpan * scale
    let teOffset2 = teOffset1 + Math.tan((te_sweep_p2 * Math.PI) / 180) * remainingSpan * scale

    const loaScaled = loa * scale
    const noseY = centerY - loaScaled / 2

    // Apply 2" gap constraint to prevent bowtie
    const MIN_GAP = 2.0
    const breakLEY = noseY + leOffset1
    const breakTEY = noseY + loaScaled - teOffset1
    if (breakTEY - breakLEY < MIN_GAP * scale) {
      teOffset1 = loaScaled - leOffset1 - MIN_GAP * scale
      teOffset2 = teOffset1 + Math.tan((te_sweep_p2 * Math.PI) / 180) * remainingSpan * scale
    }

    const tipLEY = noseY + leOffset2
    const tipTEY = noseY + loaScaled - teOffset2
    if (tipTEY - tipLEY < MIN_GAP * scale) {
      teOffset2 = loaScaled - leOffset2 - MIN_GAP * scale
    }

    const scaledBreakSpan = breakSpan * scale
    const scaledHalfSpan = halfSpan * scale

    // Build path for both wings
    const rightPoints = [
      `${centerX},${noseY}`,
      `${centerX + scaledBreakSpan},${noseY + leOffset1}`,
      `${centerX + scaledHalfSpan},${noseY + leOffset2}`,
      `${centerX + scaledHalfSpan},${noseY + loaScaled - teOffset2}`,
      `${centerX + scaledBreakSpan},${noseY + loaScaled - teOffset1}`,
      `${centerX},${noseY + loaScaled}`
    ].join(' ')

    const leftPoints = [
      `${centerX},${noseY}`,
      `${centerX - scaledBreakSpan},${noseY + leOffset1}`,
      `${centerX - scaledHalfSpan},${noseY + leOffset2}`,
      `${centerX - scaledHalfSpan},${noseY + loaScaled - teOffset2}`,
      `${centerX - scaledBreakSpan},${noseY + loaScaled - teOffset1}`,
      `${centerX},${noseY + loaScaled}`
    ].join(' ')

    return { rightPoints, leftPoints }
  }

  return (
    <Box
      sx={{
        bgcolor: '#f9f9f9',
        border: '1px solid #cccccc',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center'
      }}
    >
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${viewWidth} ${viewHeight}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ background: '#f9f9f9' }}
      >
        {/* Grid lines */}
        <line
          x1={padding}
          y1={centerY}
          x2={viewWidth - padding}
          y2={centerY}
          stroke="#cccccc"
          strokeWidth="1"
          strokeDasharray="5,5"
        />
        <line
          x1={centerX}
          y1={padding}
          x2={centerX}
          y2={viewHeight - padding}
          stroke="#cccccc"
          strokeWidth="1"
          strokeDasharray="5,5"
        />

        {/* Render each design's planform */}
        {designs.map((design, idx) => {
          const { rightPoints, leftPoints } = getWingPath(design, globalScale)
          const color = colors[idx]

          return (
            <g key={idx}>
              <polygon
                points={rightPoints}
                fill="none"
                stroke={color}
                strokeWidth="2.5"
                strokeOpacity={0.9}
              />
              <polygon
                points={leftPoints}
                fill="none"
                stroke={color}
                strokeWidth="2.5"
                strokeOpacity={0.9}
              />
            </g>
          )
        })}

        {/* Centerline */}
        <line
          x1={centerX}
          y1={centerY - (maxLoa * globalScale) / 2}
          x2={centerX}
          y2={centerY + (maxLoa * globalScale) / 2}
          stroke="#666666"
          strokeWidth="1"
          strokeDasharray="3,3"
        />

        {/* Axis labels */}
        <text
          x={viewWidth - padding + 5}
          y={centerY + 4}
          fill="#666666"
          fontFamily="Courier New"
          fontSize="10"
        >
          +Y
        </text>
        <text
          x={centerX + 5}
          y={padding - 5}
          fill="#666666"
          fontFamily="Courier New"
          fontSize="10"
        >
          +X
        </text>
      </svg>
    </Box>
  )
}

interface DesignComparisonProps {
  designs: DesignResult[]
  onRemove: (index: number) => void
  onClear: () => void
}

const COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

const METRICS = [
  { key: 'range_nm', label: 'Range (nm)', higher: true },
  { key: 'endurance_hr', label: 'Endurance (hr)', higher: true },
  { key: 'mtow_lbm', label: 'MTOW (lbm)', higher: false },
  { key: 'cost_usd', label: 'Cost ($)', higher: false },
  { key: 'wingtip_deflection_in', label: 'Tip Deflection (in)', higher: false }
] as const

type MetricKey = typeof METRICS[number]['key']

export default function DesignComparison({ designs, onRemove, onClear }: DesignComparisonProps) {
  const [sensitivityDesignIndex, setSensitivityDesignIndex] = useState(0)

  // Normalize values for radar chart (0-1 scale)
  const normalizedData = useMemo(() => {
    if (designs.length === 0) return []

    // Find min/max for each metric
    const ranges: Record<MetricKey, { min: number; max: number }> = {} as Record<MetricKey, { min: number; max: number }>

    METRICS.forEach(({ key }) => {
      const values = designs.map(d => d[key])
      ranges[key] = {
        min: Math.min(...values),
        max: Math.max(...values)
      }
    })

    // Normalize each design
    return designs.map(design => {
      const normalized: Record<MetricKey, number> = {} as Record<MetricKey, number>

      METRICS.forEach(({ key, higher }) => {
        const { min, max } = ranges[key]
        const range = max - min

        if (range === 0) {
          normalized[key] = 0.5
        } else {
          // For metrics where higher is better, normalize directly
          // For metrics where lower is better, invert
          const raw = (design[key] - min) / range
          normalized[key] = higher ? raw : 1 - raw
        }
      })

      return normalized
    })
  }, [designs])

  // Prepare radar chart data
  const radarData = useMemo(() => {
    const categories = METRICS.map(m => m.label)

    return designs.map((design, idx) => ({
      type: 'scatterpolar' as const,
      r: [...METRICS.map(m => normalizedData[idx]?.[m.key] ?? 0), normalizedData[idx]?.[METRICS[0].key] ?? 0],
      theta: [...categories, categories[0]], // Close the polygon
      fill: 'toself' as const,
      fillcolor: `${COLORS[idx]}20`,
      line: { color: COLORS[idx], width: 2 },
      name: `Design ${idx + 1}`,
      hovertemplate: METRICS.map(m =>
        `${m.label}: ${design[m.key].toLocaleString(undefined, { maximumFractionDigits: 1 })}`
      ).join('<br>') + '<extra></extra>'
    }))
  }, [designs, normalizedData])

  if (designs.length === 0) {
    return null
  }

  return (
    <Box
      sx={{
        bgcolor: '#f5f5f5',
        border: '2px solid #cccccc',
        p: 2,
        mt: 4
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
          Design Comparison ({designs.length} selected)
        </Typography>
        <IconButton onClick={onClear} size="small" title="Clear all selections">
          <CloseIcon />
        </IconButton>
      </Box>

      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 3 }}>
        {/* Radar Chart */}
        <Box>
          <Plot
            data={radarData}
            layout={{
              polar: {
                radialaxis: {
                  visible: true,
                  range: [0, 1],
                  tickvals: [0, 0.25, 0.5, 0.75, 1],
                  ticktext: ['Worst', '', 'Mid', '', 'Best'],
                  tickfont: { family: 'monospace', size: 10 }
                },
                angularaxis: {
                  tickfont: { family: 'monospace', size: 11 }
                }
              },
              showlegend: true,
              legend: {
                x: 1,
                y: 1,
                font: { family: 'monospace', size: 11 }
              },
              margin: { l: 60, r: 60, t: 40, b: 40 },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent'
            }}
            config={{ displayModeBar: false }}
            style={{ width: '100%', height: '350px' }}
          />
        </Box>

        {/* Planform Overlay Comparison */}
        <Box>
          <Typography
            sx={{
              fontFamily: 'monospace',
              fontSize: '0.9em',
              fontWeight: 'bold',
              mb: 1
            }}
          >
            Planform Overlay
          </Typography>
          <PlanformOverlay designs={designs} colors={COLORS} width={400} height={300} />
          {/* Legend */}
          <Box sx={{ display: 'flex', gap: 2, mt: 1, justifyContent: 'center' }}>
            {designs.map((_, idx) => (
              <Box key={idx} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Box sx={{ width: 16, height: 3, bgcolor: COLORS[idx] }} />
                <Typography sx={{ fontFamily: 'monospace', fontSize: '0.75em' }}>Design {idx + 1}</Typography>
              </Box>
            ))}
          </Box>
        </Box>
      </Box>

      {/* Metrics Table */}
      <Box sx={{ mt: 3, overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'monospace', fontSize: '0.85em' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #000000' }}>
              <th style={{ textAlign: 'left', padding: '8px', fontWeight: 'bold' }}>Metric</th>
              {designs.map((_, idx) => (
                <th key={idx} style={{ textAlign: 'right', padding: '8px' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 1 }}>
                    <Box sx={{ width: 12, height: 12, bgcolor: COLORS[idx], borderRadius: '50%' }} />
                    Design {idx + 1}
                    <IconButton size="small" onClick={() => onRemove(idx)} sx={{ p: 0.25 }}>
                      <CloseIcon sx={{ fontSize: 14 }} />
                    </IconButton>
                  </Box>
                </th>
              ))}
              {designs.length > 1 && <th style={{ textAlign: 'right', padding: '8px' }}>Delta</th>}
            </tr>
          </thead>
          <tbody>
            {METRICS.map(({ key, label, higher }) => {
              const values = designs.map(d => d[key])
              const best = higher ? Math.max(...values) : Math.min(...values)
              const delta = designs.length > 1 ? Math.max(...values) - Math.min(...values) : null

              return (
                <tr key={key} style={{ borderBottom: '1px solid #cccccc' }}>
                  <td style={{ padding: '8px', fontWeight: 'bold' }}>{label}</td>
                  {designs.map((design, idx) => {
                    const isBest = design[key] === best
                    return (
                      <td
                        key={idx}
                        style={{
                          textAlign: 'right',
                          padding: '8px',
                          fontWeight: isBest ? 'bold' : 'normal',
                          color: isBest ? '#2e7d32' : 'inherit'
                        }}
                      >
                        {key === 'cost_usd' ? '$' : ''}{design[key].toLocaleString(undefined, { maximumFractionDigits: 1 })}
                      </td>
                    )
                  })}
                  {delta !== null && (
                    <td style={{ textAlign: 'right', padding: '8px', color: '#666666' }}>
                      {key === 'cost_usd' ? '$' : ''}{delta.toLocaleString(undefined, { maximumFractionDigits: 1 })}
                    </td>
                  )}
                </tr>
              )
            })}
            {/* Design Parameters */}
            <tr style={{ borderTop: '2px solid #000000' }}>
              <td colSpan={designs.length + 2} style={{ padding: '8px', fontWeight: 'bold', color: '#666666' }}>
                Design Parameters
              </td>
            </tr>
            {[
              { key: 'loa', label: 'LOA (in)' },
              { key: 'span', label: 'Span (in)' },
              { key: 'panel_break', label: 'Panel Break', format: (v: number) => `${(v * 100).toFixed(0)}%` }
            ].map(({ key, label, format }) => (
              <tr key={key} style={{ borderBottom: '1px solid #cccccc' }}>
                <td style={{ padding: '8px' }}>{label}</td>
                {designs.map((design, idx) => (
                  <td key={idx} style={{ textAlign: 'right', padding: '8px' }}>
                    {format ? format(design[key as keyof DesignResult] as number) : (design[key as keyof DesignResult] as number).toFixed(1)}
                  </td>
                ))}
                {designs.length > 1 && <td style={{ textAlign: 'right', padding: '8px', color: '#666666' }}>-</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </Box>

      {/* Sensitivity Analysis */}
      <Box sx={{ mt: 3 }}>
        {designs.length > 1 && (
          <Box sx={{ mb: 2 }}>
            <Typography
              sx={{
                fontFamily: 'monospace',
                fontSize: '0.9em',
                fontWeight: 'bold',
                mb: 1
              }}
            >
              Sensitivity Analysis for:
            </Typography>
            <Tabs
              value={sensitivityDesignIndex}
              onChange={(_, newValue) => setSensitivityDesignIndex(newValue)}
              sx={{
                minHeight: 32,
                '& .MuiTab-root': {
                  minHeight: 32,
                  fontFamily: 'monospace',
                  fontSize: '0.85em',
                  textTransform: 'none'
                }
              }}
            >
              {designs.map((_, idx) => (
                <Tab
                  key={idx}
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      <Box sx={{ width: 10, height: 10, bgcolor: COLORS[idx], borderRadius: '50%' }} />
                      Design {idx + 1}
                    </Box>
                  }
                />
              ))}
            </Tabs>
          </Box>
        )}
        <SensitivityChart design={designs[sensitivityDesignIndex]} />
      </Box>
    </Box>
  )
}
