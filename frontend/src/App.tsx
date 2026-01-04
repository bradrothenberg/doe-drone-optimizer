import { useState } from 'react'
import { Box, Container } from '@mui/material'
import Header from './components/Layout/Header'
import ConstraintForm from './components/Input/ConstraintForm'
import ParetoChart2D from './components/Charts/ParetoChart2D'
import ParetoChart3D from './components/Charts/ParetoChart3D'
import DesignTable from './components/Table/DesignTable'
import { useOptimization } from './hooks/useOptimization'
import type { Constraints } from './types'

function App() {
  const [constraints, setConstraints] = useState<Constraints>({
    min_range_nm: undefined,
    max_cost_usd: undefined,
    max_mtow_lbm: undefined,
    min_endurance_hr: undefined,
    max_wingtip_deflection_in: undefined
  })

  const { data, isLoading, error } = useOptimization(constraints)

  return (
    <Container maxWidth="xl">
      <Header />

      <Box sx={{ mt: 4 }}>
        <ConstraintForm
          constraints={constraints}
          onUpdate={setConstraints}
          isOptimizing={isLoading}
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
          <Box sx={{ mt: 4, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 3 }}>
            <ParetoChart2D
              data={data.pareto_designs}
              xKey="mtow_lbm"
              yKey="range_nm"
              title="Range vs MTOW"
              xLabel="MTOW (lbm)"
              yLabel="Range (nm)"
            />
            <ParetoChart2D
              data={data.pareto_designs}
              xKey="cost_usd"
              yKey="range_nm"
              title="Range vs Cost"
              xLabel="Cost ($)"
              yLabel="Range (nm)"
            />
          </Box>

          <Box sx={{ mt: 4 }}>
            <ParetoChart3D data={data.pareto_designs} />
          </Box>

          <Box sx={{ mt: 4 }}>
            <DesignTable designs={data.pareto_designs} />
          </Box>
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
