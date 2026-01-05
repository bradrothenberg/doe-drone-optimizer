import { useState } from 'react'
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TableSortLabel,
  Button,
  Checkbox,
  IconButton,
  Tooltip
} from '@mui/material'
import OpenInNewIcon from '@mui/icons-material/OpenInNew'
import Planform from '../Visualization/Planform'
import type { DesignResult } from '../../types'
import { launchNtop } from '../../services/api'

interface DesignTableProps {
  designs: DesignResult[]
  selectedDesigns?: DesignResult[]
  onSelectionChange?: (designs: DesignResult[]) => void
  maxSelections?: number
  highlightedIndex?: number | null
  onRowClick?: (index: number) => void
}

// Numeric keys that can be safely sorted
type NumericSortKey = 'loa' | 'span' | 'le_sweep_p1' | 'le_sweep_p2' | 'te_sweep_p1' | 'te_sweep_p2' |
  'panel_break' | 'range_nm' | 'endurance_hr' | 'mtow_lbm' | 'cost_usd' | 'wingtip_deflection_in'
type SortOrder = 'asc' | 'desc'

export default function DesignTable({
  designs,
  selectedDesigns = [],
  onSelectionChange,
  maxSelections = 4,
  highlightedIndex,
  onRowClick
}: DesignTableProps) {
  const [page, setPage] = useState(0)
  const [rowsPerPage, setRowsPerPage] = useState(10)
  const [sortKey, setSortKey] = useState<NumericSortKey>('range_nm')
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc')
  const [launchingIndex, setLaunchingIndex] = useState<number | null>(null)

  const handleSort = (key: NumericSortKey) => {
    const isAsc = sortKey === key && sortOrder === 'asc'
    setSortOrder(isAsc ? 'desc' : 'asc')
    setSortKey(key)
  }

  // Check if a design is selected (by comparing key properties)
  const isSelected = (design: DesignResult) => {
    return selectedDesigns.some(
      d => d.loa === design.loa && d.span === design.span &&
           d.le_sweep_p1 === design.le_sweep_p1 && d.range_nm === design.range_nm
    )
  }

  const handleToggleSelection = (design: DesignResult, e: React.MouseEvent) => {
    e.stopPropagation() // Don't trigger row click
    if (!onSelectionChange) return

    if (isSelected(design)) {
      // Remove from selection
      onSelectionChange(selectedDesigns.filter(
        d => !(d.loa === design.loa && d.span === design.span &&
               d.le_sweep_p1 === design.le_sweep_p1 && d.range_nm === design.range_nm)
      ))
    } else if (selectedDesigns.length < maxSelections) {
      // Add to selection
      onSelectionChange([...selectedDesigns, design])
    }
  }

  const handleLaunchNtop = async (design: DesignResult & { originalIndex: number }, e: React.MouseEvent) => {
    e.stopPropagation() // Don't trigger row click

    setLaunchingIndex(design.originalIndex)
    try {
      await launchNtop({
        run_id: `Design ${design.originalIndex + 1}`,
        loa: design.loa,
        span: design.span,
        le_sweep_p1: design.le_sweep_p1,
        le_sweep_p2: design.le_sweep_p2,
        te_sweep_p1: design.te_sweep_p1,
        te_sweep_p2: design.te_sweep_p2,
        panel_break: design.panel_break
      })
    } catch (error) {
      console.error('Failed to launch nTop:', error)
    } finally {
      setLaunchingIndex(null)
    }
  }

  // Track original indices for highlighting
  const indexedDesigns = designs.map((d, i) => ({ ...d, originalIndex: i }))

  const sortedDesigns = [...indexedDesigns].sort((a, b) => {
    const aVal = a[sortKey]
    const bVal = b[sortKey]
    return sortOrder === 'asc' ? aVal - bVal : bVal - aVal
  })

  const paginatedDesigns = sortedDesigns.slice(
    page * rowsPerPage,
    page * rowsPerPage + rowsPerPage
  )

  const handleExportCSV = () => {
    // Export in the current sorted order with design number
    const csv = [
      ['Design #', 'LOA In', 'Span', 'LE Sweep P1', 'LE Sweep P2', 'TE Sweep P1', 'TE Sweep P2', 'Panel Break %'].join(','),
      ...sortedDesigns.map(d =>
        [
          d.originalIndex + 1,  // Design number (1-indexed)
          d.loa.toFixed(2),
          d.span.toFixed(2),
          d.le_sweep_p1.toFixed(2),
          d.le_sweep_p2.toFixed(2),
          d.te_sweep_p1.toFixed(2),
          d.te_sweep_p2.toFixed(2),
          (d.panel_break * 100).toFixed(2) // Convert to percentage
        ].join(',')
      )
    ].join('\n')

    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'ntop_input_parameters.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <Box
      sx={{
        bgcolor: '#f5f5f5',
        border: '2px solid #cccccc',
        p: 2
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
            pb: 1,
            flex: 1
          }}
        >
          Pareto-Optimal Designs ({designs.length} total)
        </Typography>

        <Button
          variant="outlined"
          onClick={handleExportCSV}
          sx={{
            borderColor: '#000000',
            color: '#000000',
            fontFamily: 'monospace',
            ml: 2,
            '&:hover': {
              borderColor: '#000000',
              bgcolor: '#e0e0e0'
            }
          }}
        >
          Export CSV
        </Button>
      </Box>

      <Typography
        sx={{
          fontFamily: 'monospace',
          fontSize: '0.8em',
          color: '#666666',
          mb: 1
        }}
      >
        Click a row to view planform details
      </Typography>

      <TableContainer>
        <Table size="small" sx={{ '& *': { fontFamily: 'monospace !important' } }}>
          <TableHead>
            <TableRow sx={{ bgcolor: '#e0e0e0' }}>
              {onSelectionChange && (
                <TableCell sx={{ fontWeight: 'bold', borderBottom: '2px solid #000000', width: 50 }}>
                  Compare
                </TableCell>
              )}
              <TableCell sx={{ fontWeight: 'bold', borderBottom: '2px solid #000000', width: 60 }}>
                #
              </TableCell>
              <TableCell sx={{ fontWeight: 'bold', borderBottom: '2px solid #000000' }}>
                Planform
              </TableCell>
              {[
                { key: 'loa', label: 'LOA (in)' },
                { key: 'span', label: 'Span (in)' },
                { key: 'range_nm', label: 'Range (nm)' },
                { key: 'endurance_hr', label: 'Endurance (hr)' },
                { key: 'mtow_lbm', label: 'MTOW (lbm)' },
                { key: 'cost_usd', label: 'Cost ($)' },
                { key: 'wingtip_deflection_in', label: 'Tip Defl (in)' }
              ].map(({ key, label }) => (
                <TableCell key={key} sx={{ fontWeight: 'bold', borderBottom: '2px solid #000000' }}>
                  <TableSortLabel
                    active={sortKey === key}
                    direction={sortKey === key ? sortOrder : 'asc'}
                    onClick={() => handleSort(key as NumericSortKey)}
                  >
                    {label}
                  </TableSortLabel>
                </TableCell>
              ))}
              <TableCell sx={{ fontWeight: 'bold', borderBottom: '2px solid #000000', width: 60 }}>
                nTop
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {paginatedDesigns.map((design, idx) => {
              const selected = isSelected(design)
              const disabled = !selected && selectedDesigns.length >= maxSelections
              const isHighlighted = design.originalIndex === highlightedIndex
              const isLaunching = launchingIndex === design.originalIndex
              return (
              <TableRow
                key={idx}
                onClick={() => onRowClick?.(design.originalIndex)}
                sx={{
                  cursor: onRowClick ? 'pointer' : 'default',
                  '&:nth-of-type(odd)': {
                    bgcolor: isHighlighted ? '#bbdefb' : selected ? '#e3f2fd' : '#ffffff'
                  },
                  '&:nth-of-type(even)': {
                    bgcolor: isHighlighted ? '#bbdefb' : selected ? '#e3f2fd' : '#f5f5f5'
                  },
                  '&:hover': { bgcolor: isHighlighted ? '#90caf9' : selected ? '#bbdefb' : '#e0e0e0' },
                  border: isHighlighted ? '2px solid #1565c0' : 'none'
                }}
              >
                {onSelectionChange && (
                  <TableCell sx={{ p: 0.5 }}>
                    <Checkbox
                      checked={selected}
                      disabled={disabled}
                      onClick={(e) => handleToggleSelection(design, e)}
                      size="small"
                      sx={{
                        color: '#666666',
                        '&.Mui-checked': { color: '#1976d2' },
                        '&.Mui-disabled': { color: '#cccccc' }
                      }}
                    />
                  </TableCell>
                )}
                <TableCell sx={{ p: 0.5, textAlign: 'center', fontWeight: 'bold' }}>
                  {design.originalIndex + 1}
                </TableCell>
                <TableCell sx={{ p: 0.5 }}>
                  <Planform design={design} width={140} height={100} />
                </TableCell>
                <TableCell>{design.loa.toFixed(1)}</TableCell>
                <TableCell>{design.span.toFixed(1)}</TableCell>
                <TableCell>{design.range_nm.toFixed(0)}</TableCell>
                <TableCell>{design.endurance_hr.toFixed(1)}</TableCell>
                <TableCell>{design.mtow_lbm.toFixed(0)}</TableCell>
                <TableCell>${design.cost_usd.toLocaleString()}</TableCell>
                <TableCell>{design.wingtip_deflection_in.toFixed(1)}</TableCell>
                <TableCell sx={{ p: 0.5 }}>
                  <Tooltip title="Open in nTop">
                    <IconButton
                      size="small"
                      onClick={(e) => handleLaunchNtop(design, e)}
                      disabled={isLaunching}
                      sx={{
                        color: '#2e7d32',
                        '&:hover': {
                          bgcolor: '#e8f5e9'
                        },
                        '&:disabled': {
                          color: '#81c784'
                        }
                      }}
                    >
                      <OpenInNewIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </TableCell>
              </TableRow>
            )})}
          </TableBody>
        </Table>
      </TableContainer>

      <TablePagination
        rowsPerPageOptions={[10, 20, 50]}
        component="div"
        count={designs.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={(_e, newPage) => setPage(newPage)}
        onRowsPerPageChange={(e) => {
          setRowsPerPage(parseInt(e.target.value, 10))
          setPage(0)
        }}
        sx={{ borderTop: '1px solid #cccccc', mt: 2 }}
      />
    </Box>
  )
}
