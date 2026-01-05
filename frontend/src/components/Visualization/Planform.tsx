import { Box } from '@mui/material'
import type { DesignResult } from '../../types'

interface PlanformProps {
  design: DesignResult
  width?: number
  height?: number
  color?: string
  showLabels?: boolean
}

/**
 * Renders a top-down planform view of the wing design
 * Matches the DOE report style with:
 * - Light blue wing fill
 * - Yellow elevons
 * - Red/pink control surfaces
 * - Grid lines and axis labels
 */
export default function Planform({ design, width = 400, height = 350, color, showLabels = true }: PlanformProps) {
  const { loa, span, le_sweep_p1, le_sweep_p2, te_sweep_p1, te_sweep_p2, panel_break } = design

  // SVG viewBox dimensions (matching DOE report style)
  const viewWidth = 500
  const viewHeight = 450
  const padding = 40
  const centerX = viewWidth / 2
  const centerY = viewHeight / 2

  // Available drawing area
  const drawWidth = viewWidth - 2 * padding
  const drawHeight = viewHeight - 2 * padding

  // Scale to fit the wing in the drawing area
  const scaleX = drawWidth / span
  const scaleY = drawHeight / loa
  const scale = Math.min(scaleX, scaleY) * 0.9

  // Wing geometry calculations
  const halfSpan = span / 2
  const breakSpan = halfSpan * panel_break
  const remainingSpan = halfSpan - breakSpan

  // Convert sweep angles to Y offsets
  // Positive LE sweep = leading edge moves aft (down in view, +Y offset)
  // Positive TE sweep = trailing edge moves aft (down in view, +Y offset from root TE)
  // Negative TE sweep = trailing edge moves forward (up in view, -Y offset from root TE)
  const leOffset1 = Math.tan((le_sweep_p1 * Math.PI) / 180) * breakSpan * scale
  const leOffset2 = leOffset1 + Math.tan((le_sweep_p2 * Math.PI) / 180) * remainingSpan * scale
  // TE offset is ADDED to root TE position (positive = aft, negative = forward)
  let teOffset1 = Math.tan((te_sweep_p1 * Math.PI) / 180) * breakSpan * scale
  let teOffset2 = teOffset1 + Math.tan((te_sweep_p2 * Math.PI) / 180) * remainingSpan * scale

  // Scaled LOA
  const loaScaled = loa * scale

  // Starting Y position (nose at top)
  const noseY = centerY - loaScaled / 2

  // Apply 2" gap constraint to prevent bowtie at panel break (P1)
  // With new convention: TE Y = noseY + loaScaled + teOffset
  // Chord at break = (noseY + loaScaled + teOffset1) - (noseY + leOffset1) = loaScaled + teOffset1 - leOffset1
  const MIN_GAP = 2.0
  const chordAtBreak = loaScaled + teOffset1 - leOffset1
  if (chordAtBreak < MIN_GAP * scale) {
    // Adjust teOffset1 to maintain minimum chord
    teOffset1 = leOffset1 - loaScaled + MIN_GAP * scale
    // Recalculate teOffset2 based on corrected teOffset1
    teOffset2 = teOffset1 + Math.tan((te_sweep_p2 * Math.PI) / 180) * remainingSpan * scale
  }

  // Apply 2" gap constraint to prevent bowtie at wingtip (P2)
  // Chord at tip = loaScaled + teOffset2 - leOffset2
  const chordAtTip = loaScaled + teOffset2 - leOffset2
  if (chordAtTip < MIN_GAP * scale) {
    teOffset2 = leOffset2 - loaScaled + MIN_GAP * scale
  }

  // Wing colors (can be overridden for comparison overlay)
  const wingFill = color ? `${color}40` : '#add8e6'
  const wingFillOpacity = color ? 0.7 : 0.4
  const wingStroke = color || '#000000'

  // Calculate wing outline points for right wing
  const scaledBreakSpan = breakSpan * scale
  const scaledHalfSpan = halfSpan * scale

  // Right wing polygon points
  // TE Y position = root TE (noseY + loaScaled) + teOffset
  // Positive teOffset = TE moves aft (down), negative teOffset = TE moves forward (up)
  const rightWingPoints = [
    // Root LE
    `${centerX},${noseY}`,
    // Panel break LE (interpolate)
    `${centerX + scaledBreakSpan * 0.25},${noseY + leOffset1 * 0.25}`,
    `${centerX + scaledBreakSpan * 0.5},${noseY + leOffset1 * 0.5}`,
    `${centerX + scaledBreakSpan * 0.75},${noseY + leOffset1 * 0.75}`,
    `${centerX + scaledBreakSpan},${noseY + leOffset1}`,
    // Tip LE
    `${centerX + scaledHalfSpan},${noseY + leOffset2}`,
    // Tip TE (+ teOffset: positive = aft/down, negative = forward/up)
    `${centerX + scaledHalfSpan},${noseY + loaScaled + teOffset2}`,
    // Panel break TE
    `${centerX + scaledBreakSpan},${noseY + loaScaled + teOffset1}`,
    // Root TE (interpolate back)
    `${centerX + scaledBreakSpan * 0.75},${noseY + loaScaled + teOffset1 * 0.75}`,
    `${centerX + scaledBreakSpan * 0.5},${noseY + loaScaled + teOffset1 * 0.5}`,
    `${centerX + scaledBreakSpan * 0.25},${noseY + loaScaled + teOffset1 * 0.25}`,
    `${centerX},${noseY + loaScaled}`
  ].join(' ')

  // Left wing polygon points (mirrored)
  const leftWingPoints = [
    `${centerX},${noseY}`,
    `${centerX - scaledBreakSpan * 0.25},${noseY + leOffset1 * 0.25}`,
    `${centerX - scaledBreakSpan * 0.5},${noseY + leOffset1 * 0.5}`,
    `${centerX - scaledBreakSpan * 0.75},${noseY + leOffset1 * 0.75}`,
    `${centerX - scaledBreakSpan},${noseY + leOffset1}`,
    `${centerX - scaledHalfSpan},${noseY + leOffset2}`,
    `${centerX - scaledHalfSpan},${noseY + loaScaled + teOffset2}`,
    `${centerX - scaledBreakSpan},${noseY + loaScaled + teOffset1}`,
    `${centerX - scaledBreakSpan * 0.75},${noseY + loaScaled + teOffset1 * 0.75}`,
    `${centerX - scaledBreakSpan * 0.5},${noseY + loaScaled + teOffset1 * 0.5}`,
    `${centerX - scaledBreakSpan * 0.25},${noseY + loaScaled + teOffset1 * 0.25}`,
    `${centerX},${noseY + loaScaled}`
  ].join(' ')

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

        {/* Right wing */}
        <polygon
          points={rightWingPoints}
          fill={wingFill}
          fillOpacity={wingFillOpacity}
          stroke={wingStroke}
          strokeWidth="2"
        />

        {/* Left wing */}
        <polygon
          points={leftWingPoints}
          fill={wingFill}
          fillOpacity={wingFillOpacity}
          stroke={wingStroke}
          strokeWidth="2"
        />


        {/* Centerline */}
        <line
          x1={centerX}
          y1={noseY}
          x2={centerX}
          y2={noseY + loaScaled}
          stroke="#666666"
          strokeWidth="1"
          strokeDasharray="3,3"
        />

        {/* Axis labels */}
        {showLabels && (
          <>
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
          </>
        )}

        {/* Dimension labels */}
        {showLabels && (
          <>
            <text
              x={centerX}
              y={viewHeight - 10}
              textAnchor="middle"
              fill="#333333"
              fontFamily="Courier New"
              fontSize="11"
            >
              Span: {span.toFixed(0)}" | LOA: {loa.toFixed(0)}"
            </text>
          </>
        )}
      </svg>
    </Box>
  )
}
