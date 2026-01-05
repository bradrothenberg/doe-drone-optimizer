import { useState, useCallback } from 'react'
import { Box, Container } from '@mui/material'
import Header from './components/Layout/Header'
import ConstraintForm from './components/Input/ConstraintForm'
import ParetoChart2D from './components/Charts/ParetoChart2D'
import ParetoChart3D from './components/Charts/ParetoChart3D'
import DesignTable from './components/Table/DesignTable'
import DesignComparison from './components/Comparison/DesignComparison'
import SelectedDesignPanel from './components/Selection/SelectedDesignPanel'
import { useOptimization } from './hooks/useOptimization'
import type { Constraints, OptimizationObjectives, DesignResult } from './types'

// Default objectives
const defaultObjectives: OptimizationObjectives = {
  range_nm: 'maximize',
  endurance_hr: 'maximize',
  mtow_lbm: 'minimize',
  cost_usd: 'minimize',
  wingtip_deflection_in: 'minimize'
}

function App() {
  // Constraints that have been submitted for optimization
  const [submittedConstraints, setSubmittedConstraints] = useState<Constraints>({
    min_range_nm: undefined,
    max_cost_usd: undefined,
    max_mtow_lbm: undefined,
    min_endurance_hr: undefined,
    max_wingtip_deflection_in: undefined,
    // Default taper ratio constraints (enabled by default)
    min_taper_ratio_p1: 0.1,
    max_taper_ratio_p1: 1.0,
    min_taper_ratio_p2: 0.1,
    max_taper_ratio_p2: 0.8
  })
  const [submittedObjectives, setSubmittedObjectives] = useState<OptimizationObjectives>(defaultObjectives)
  const [selectedDesigns, setSelectedDesigns] = useState<DesignResult[]>([])
  const [shouldRun, setShouldRun] = useState(false)
  const [highlightedDesignIndex, setHighlightedDesignIndex] = useState<number | null>(null)

  const { data, isLoading, error } = useOptimization(submittedConstraints, submittedObjectives, shouldRun)

  // Handle optimization trigger
  const handleRunOptimization = useCallback((newConstraints: Constraints, newObjectives: OptimizationObjectives) => {
    setSelectedDesigns([])
    setHighlightedDesignIndex(null)
    setSubmittedConstraints(newConstraints)
    setSubmittedObjectives(newObjectives)
    setShouldRun(true)
  }, [])

  const handleRemoveSelection = (index: number) => {
    setSelectedDesigns(prev => prev.filter((_, i) => i !== index))
  }

  const handleClearSelections = () => {
    setSelectedDesigns([])
  }

  const handleSelectDesign = (index: number) => {
    setHighlightedDesignIndex(index)
  }

  // Get the highlighted design
  const highlightedDesign = data && highlightedDesignIndex !== null
    ? data.pareto_designs[highlightedDesignIndex]
    : null

  return (
    <Container maxWidth="xl">
      <Header />

      <Box sx={{ mt: 4 }}>
        <ConstraintForm
          constraints={submittedConstraints}
          objectives={submittedObjectives}
          onUpdate={handleRunOptimization}
          isOptimizing={isLoading}
          hasResults={!!data}
        />
      </Box>

      {error && (
        <Box sx={{
          mt: 3,
          p: 2,
          bgcolor: '#ffebee',
          border: '1px solid #ef5350',
          fontFamily: 'monospace'
        }}>
          Error: {error.message}
        </Box>
      )}

      {data && (
        <>
          {/* Charts and Planform Panel - Side by Side Layout */}
          <Box sx={{ mt: 4, display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 3 }}>
            {/* Left side - Charts */}
            <Box>
              <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 3 }}>
                <ParetoChart2D
                  data={data.pareto_designs}
                  xKey="mtow_lbm"
                  yKey="range_nm"
                  title="Range vs MTOW"
                  xLabel="MTOW (lbm)"
                  yLabel="Range (nm)"
                  selectedIndex={highlightedDesignIndex}
                  onSelectDesign={handleSelectDesign}
                />
                <ParetoChart2D
                  data={data.pareto_designs}
                  xKey="cost_usd"
                  yKey="range_nm"
                  title="Range vs Cost"
                  xLabel="Cost ($)"
                  yLabel="Range (nm)"
                  selectedIndex={highlightedDesignIndex}
                  onSelectDesign={handleSelectDesign}
                />
              </Box>
            </Box>

            {/* Right side - Planform Panel */}
            <SelectedDesignPanel
              design={highlightedDesign}
              designIndex={highlightedDesignIndex ?? undefined}
              totalDesigns={data.pareto_designs.length}
              onNavigate={handleSelectDesign}
            />
          </Box>

          <Box sx={{ mt: 4 }}>
            <ParetoChart3D
              data={data.pareto_designs}
              selectedIndex={highlightedDesignIndex}
              onSelectDesign={handleSelectDesign}
            />
          </Box>

          <Box sx={{ mt: 4 }}>
            <DesignTable
              designs={data.pareto_designs}
              selectedDesigns={selectedDesigns}
              onSelectionChange={setSelectedDesigns}
              maxSelections={4}
              highlightedIndex={highlightedDesignIndex}
              onRowClick={handleSelectDesign}
            />
          </Box>

          {selectedDesigns.length > 0 && (
            <DesignComparison
              designs={selectedDesigns}
              onRemove={handleRemoveSelection}
              onClear={handleClearSelections}
            />
          )}
        </>
      )}

      {isLoading && (
        <Box sx={{ mt: 4, textAlign: 'center', fontFamily: 'monospace' }}>
          Optimizing... This may take up to 30 seconds.
        </Box>
      )}
    </Container>
  )
}

export default App
