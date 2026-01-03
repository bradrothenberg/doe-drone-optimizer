import { Box } from '@mui/material'
import type { DesignResult } from '../../types'

interface PlanformProps {
  design: DesignResult
  width?: number
  height?: number
}

/**
 * Renders a top-down planform view of the wing design
 * Shows the wing shape based on sweep angles and panel break
 */
export default function Planform({ design, width = 200, height = 150 }: PlanformProps) {
  const { loa, span, le_sweep_p1, le_sweep_p2, te_sweep_p1, te_sweep_p2, panel_break } = design

  // Scale factors to fit in SVG viewport
  const scaleX = width / span
  const scaleY = height / loa
  const scale = Math.min(scaleX, scaleY) * 0.8 // 80% to leave margins

  // Centerline position
  const centerX = width / 2
  const startY = height * 0.1

  // Panel break position (fraction of half-span)
  const halfSpan = span / 2
  const breakSpan = halfSpan * panel_break

  // Calculate sweep offsets (convert degrees to offset)
  // Positive sweep = leading edge swept back
  const leOffset1 = Math.tan((le_sweep_p1 * Math.PI) / 180) * breakSpan * scale
  const teOffset1 = Math.tan((te_sweep_p1 * Math.PI) / 180) * breakSpan * scale

  const remainingSpan = halfSpan - breakSpan
  const leOffset2 = leOffset1 + Math.tan((le_sweep_p2 * Math.PI) / 180) * remainingSpan * scale
  const teOffset2 = teOffset1 + Math.tan((te_sweep_p2 * Math.PI) / 180) * remainingSpan * scale

  // Wing outline points (right half, then mirrored for left)
  const rightWing = [
    // Root leading edge
    { x: centerX, y: startY },
    // Panel break leading edge
    { x: centerX + breakSpan * scale, y: startY + leOffset1 },
    // Tip leading edge
    { x: centerX + halfSpan * scale, y: startY + leOffset2 },
    // Tip trailing edge
    { x: centerX + halfSpan * scale, y: startY + loa * scale - teOffset2 },
    // Panel break trailing edge
    { x: centerX + breakSpan * scale, y: startY + loa * scale - teOffset1 },
    // Root trailing edge
    { x: centerX, y: startY + loa * scale }
  ]

  const leftWing = rightWing.map(p => ({ x: centerX - (p.x - centerX), y: p.y })).reverse()

  const pathD = [
    `M ${rightWing[0].x} ${rightWing[0].y}`,
    ...rightWing.slice(1).map(p => `L ${p.x} ${p.y}`),
    ...leftWing.map(p => `L ${p.x} ${p.y}`),
    'Z'
  ].join(' ')

  // Panel break line
  const breakLineRight = `M ${centerX + breakSpan * scale} ${startY + leOffset1} L ${centerX + breakSpan * scale} ${startY + loa * scale - teOffset2 + teOffset1}`
  const breakLineLeft = `M ${centerX - breakSpan * scale} ${startY + leOffset1} L ${centerX - breakSpan * scale} ${startY + loa * scale - teOffset2 + teOffset1}`

  return (
    <Box
      sx={{
        bgcolor: '#ffffff',
        border: '1px solid #cccccc',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        p: 0.5
      }}
    >
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {/* Wing planform */}
        <path
          d={pathD}
          fill="#f5f5f5"
          stroke="#000000"
          strokeWidth="2"
        />

        {/* Panel break lines */}
        <path d={breakLineRight} stroke="#666666" strokeWidth="1" strokeDasharray="4 2" />
        <path d={breakLineLeft} stroke="#666666" strokeWidth="1" strokeDasharray="4 2" />

        {/* Centerline */}
        <line
          x1={centerX}
          y1={startY}
          x2={centerX}
          y2={startY + loa * scale}
          stroke="#cccccc"
          strokeWidth="1"
          strokeDasharray="2 2"
        />

        {/* Labels */}
        <text
          x={centerX}
          y={height - 5}
          textAnchor="middle"
          fontSize="10"
          fontFamily="monospace"
          fill="#666666"
        >
          Span: {span.toFixed(0)}"
        </text>

        <text
          x={5}
          y={height / 2}
          textAnchor="start"
          fontSize="10"
          fontFamily="monospace"
          fill="#666666"
          transform={`rotate(-90, 5, ${height / 2})`}
        >
          LOA: {loa.toFixed(0)}"
        </text>
      </svg>
    </Box>
  )
}
