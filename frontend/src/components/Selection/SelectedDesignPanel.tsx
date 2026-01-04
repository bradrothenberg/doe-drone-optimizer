import { Box, Typography } from '@mui/material'
import type { DesignResult } from '../../types'
import Planform from '../Visualization/Planform'

interface SelectedDesignPanelProps {
  design: DesignResult | null
  designIndex?: number
}

interface MetricItemProps {
  label: string
  value: string | number
  bgColor?: string
  borderColor?: string
}

function MetricItem({ label, value, bgColor = '#f9f9f9', borderColor = '#cccccc' }: MetricItemProps) {
  return (
    <Box
      sx={{
        bgcolor: bgColor,
        border: `1px solid ${borderColor}`,
        p: 1.25
      }}
    >
      <Typography
        sx={{
          color: '#666666',
          fontSize: '0.75em',
          fontFamily: 'monospace'
        }}
      >
        {label}
      </Typography>
      <Typography
        sx={{
          color: '#000000',
          fontSize: '1.1em',
          fontWeight: 'bold',
          fontFamily: 'monospace'
        }}
      >
        {value}
      </Typography>
    </Box>
  )
}

export default function SelectedDesignPanel({ design, designIndex }: SelectedDesignPanelProps) {
  if (!design) {
    return (
      <Box
        sx={{
          bgcolor: '#f5f5f5',
          border: '2px solid #cccccc',
          p: 2,
          height: '100%',
          minHeight: 500
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
          Wing Planform View
        </Typography>
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            height: 300,
            color: '#666666',
            fontFamily: 'monospace',
            fontSize: '0.9em',
            textAlign: 'center'
          }}
        >
          Click a data point on the chart<br />to view the wing planform
        </Box>
      </Box>
    )
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
          mb: 2,
          fontSize: '1.1em',
          borderBottom: '2px solid #000000',
          pb: 1
        }}
      >
        Wing Planform View
      </Typography>

      {/* Planform visualization */}
      <Box sx={{ mb: 2 }}>
        <Planform design={design} width={350} height={300} />
      </Box>

      {/* Metrics grid */}
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 1.25
        }}
      >
        {designIndex !== undefined && (
          <MetricItem label="Design #" value={designIndex + 1} />
        )}
        <MetricItem label="Range" value={`${design.range_nm.toFixed(0)} nm`} />
        <MetricItem label="MTOW" value={`${design.mtow_lbm.toFixed(0)} lbm`} />
        <MetricItem label="Endurance" value={`${design.endurance_hr.toFixed(1)} hr`} />
        <MetricItem label="Wingspan" value={`${(design.span / 12).toFixed(1)} ft`} />
        <MetricItem label="LOA" value={`${(design.loa / 12).toFixed(1)} ft`} />
        <MetricItem label="Panel Break" value={`${(design.panel_break * 100).toFixed(0)}%`} />
        <MetricItem
          label="Material Cost"
          value={`$${design.cost_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
          bgColor="#e8f5e9"
          borderColor="#a5d6a7"
        />
        <MetricItem
          label="Tip Deflection"
          value={`${design.wingtip_deflection_in.toFixed(3)} in`}
          bgColor="#fff3e0"
          borderColor="#ffcc80"
        />
      </Box>

      {/* Sweep angles */}
      <Typography
        sx={{
          fontFamily: 'monospace',
          fontSize: '0.8em',
          color: '#666666',
          mt: 2,
          pt: 1,
          borderTop: '1px solid #cccccc'
        }}
      >
        LE Sweep: {design.le_sweep_p1.toFixed(1)}° / {design.le_sweep_p2.toFixed(1)}° |
        TE Sweep: {design.te_sweep_p1.toFixed(1)}° / {design.te_sweep_p2.toFixed(1)}°
      </Typography>
    </Box>
  )
}
