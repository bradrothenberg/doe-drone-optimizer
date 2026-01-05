import { useState, useEffect, useRef, useCallback } from 'react'
import { Box, Typography, TextField, Button } from '@mui/material'
import type { DesignResult } from '../../types'
import Planform from '../Visualization/Planform'
import { launchNtop } from '../../services/api'

interface SelectedDesignPanelProps {
  design: DesignResult | null
  designIndex?: number
  totalDesigns?: number
  onNavigate?: (newIndex: number) => void
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

// Navigation arrow button styled like the reference
function NavArrowButton({ direction, onClick, disabled }: { direction: 'prev' | 'next', onClick: () => void, disabled: boolean }) {
  return (
    <Box
      component="button"
      onClick={onClick}
      disabled={disabled}
      title={direction === 'prev' ? 'Previous design (Left Arrow)' : 'Next design (Right Arrow)'}
      sx={{
        background: '#ffffff',
        color: disabled ? '#cccccc' : '#333333',
        border: '1px solid #cccccc',
        padding: '8px 14px',
        fontFamily: "'Courier New', monospace",
        fontSize: '1.3em',
        cursor: disabled ? 'default' : 'pointer',
        transition: 'background-color 0.15s, border-color 0.15s',
        '&:hover:not(:disabled)': {
          background: '#e0e0e0',
          borderColor: '#999999'
        },
        '&:disabled': {
          opacity: 0.5
        }
      }}
    >
      {direction === 'prev' ? '←' : '→'}
    </Box>
  )
}

export default function SelectedDesignPanel({ design, designIndex, totalDesigns, onNavigate }: SelectedDesignPanelProps) {
  const [isEditing, setIsEditing] = useState(false)
  const [editValue, setEditValue] = useState('')
  const [ntopStatus, setNtopStatus] = useState<{ message: string; color: string } | null>(null)
  const [isLaunchingNtop, setIsLaunchingNtop] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!design || !onNavigate || totalDesigns === undefined || designIndex === undefined) return
      if (isEditing) return // Don't navigate while editing

      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        if (designIndex > 0) {
          onNavigate(designIndex - 1)
        }
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        if (designIndex < totalDesigns - 1) {
          onNavigate(designIndex + 1)
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [design, designIndex, totalDesigns, onNavigate, isEditing])

  // Focus input when editing starts
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [isEditing])

  const handleDoubleClick = useCallback(() => {
    if (designIndex !== undefined) {
      setEditValue(String(designIndex + 1))
      setIsEditing(true)
    }
  }, [designIndex])

  const handleEditSubmit = useCallback(() => {
    const newIndex = parseInt(editValue, 10) - 1
    if (!isNaN(newIndex) && newIndex >= 0 && totalDesigns !== undefined && newIndex < totalDesigns && onNavigate) {
      onNavigate(newIndex)
    }
    setIsEditing(false)
  }, [editValue, totalDesigns, onNavigate])

  const handleEditKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleEditSubmit()
    } else if (e.key === 'Escape') {
      setIsEditing(false)
    }
  }, [handleEditSubmit])

  const handlePrev = useCallback(() => {
    if (designIndex !== undefined && designIndex > 0 && onNavigate) {
      onNavigate(designIndex - 1)
    }
  }, [designIndex, onNavigate])

  const handleNext = useCallback(() => {
    if (designIndex !== undefined && totalDesigns !== undefined && designIndex < totalDesigns - 1 && onNavigate) {
      onNavigate(designIndex + 1)
    }
  }, [designIndex, totalDesigns, onNavigate])

  const handleLaunchNtop = useCallback(async () => {
    if (!design) return

    setIsLaunchingNtop(true)
    setNtopStatus({ message: 'Launching nTop...', color: '#1565c0' })

    try {
      const response = await launchNtop({
        run_id: designIndex !== undefined ? `Design ${designIndex + 1}` : undefined,
        loa: design.loa,
        span: design.span,
        le_sweep_p1: design.le_sweep_p1,
        le_sweep_p2: design.le_sweep_p2,
        te_sweep_p1: design.te_sweep_p1,
        te_sweep_p2: design.te_sweep_p2,
        panel_break: design.panel_break
      })

      if (response.status === 'ok') {
        setNtopStatus({ message: response.message, color: '#2e7d32' })
      } else {
        setNtopStatus({ message: `Error: ${response.message}`, color: '#c62828' })
      }
    } catch (error) {
      setNtopStatus({ message: 'Failed to launch nTop. Check server connection.', color: '#c62828' })
    } finally {
      setIsLaunchingNtop(false)
    }
  }, [design, designIndex])

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

  const canNavigatePrev = designIndex !== undefined && designIndex > 0
  const canNavigateNext = designIndex !== undefined && totalDesigns !== undefined && designIndex < totalDesigns - 1

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

      {/* Navigation controls */}
      {designIndex !== undefined && totalDesigns !== undefined && onNavigate && (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 2,
            mb: 2
          }}
        >
          <NavArrowButton direction="prev" onClick={handlePrev} disabled={!canNavigatePrev} />

          {isEditing ? (
            <TextField
              inputRef={inputRef}
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              onBlur={handleEditSubmit}
              onKeyDown={handleEditKeyDown}
              size="small"
              sx={{
                width: 80,
                '& .MuiInputBase-input': {
                  fontFamily: 'monospace',
                  fontWeight: 'bold',
                  textAlign: 'center',
                  padding: '6px 8px'
                }
              }}
            />
          ) : (
            <Typography
              onDoubleClick={handleDoubleClick}
              sx={{
                fontFamily: 'monospace',
                fontSize: '1.1em',
                fontWeight: 'bold',
                color: '#000000',
                cursor: 'pointer',
                padding: '4px 12px',
                border: '1px solid transparent',
                borderRadius: '4px',
                '&:hover': {
                  bgcolor: '#e0e0e0',
                  border: '1px solid #cccccc'
                }
              }}
              title="Double-click to enter design number"
            >
              Design {designIndex + 1} of {totalDesigns}
            </Typography>
          )}

          <NavArrowButton direction="next" onClick={handleNext} disabled={!canNavigateNext} />
        </Box>
      )}

      {/* Planform visualization */}
      <Box sx={{ mb: 2 }}>
        <Planform design={design} width={350} height={300} />
      </Box>

      {/* Open in nTop button */}
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mb: 2 }}>
        <Button
          variant="contained"
          onClick={handleLaunchNtop}
          disabled={isLaunchingNtop}
          sx={{
            bgcolor: '#2e7d32',
            color: 'white',
            fontFamily: "'Courier New', monospace",
            textTransform: 'none',
            px: 3,
            py: 1,
            '&:hover': {
              bgcolor: '#1b5e20'
            },
            '&:disabled': {
              bgcolor: '#81c784',
              color: 'white'
            }
          }}
        >
          {isLaunchingNtop ? 'Launching...' : 'Open in nTop'}
        </Button>
        {ntopStatus && (
          <Typography
            sx={{
              mt: 1,
              fontSize: '0.85em',
              color: ntopStatus.color,
              fontFamily: 'monospace',
              textAlign: 'center'
            }}
          >
            {ntopStatus.message}
          </Typography>
        )}
      </Box>

      {/* Metrics grid */}
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 1.25
        }}
      >
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
