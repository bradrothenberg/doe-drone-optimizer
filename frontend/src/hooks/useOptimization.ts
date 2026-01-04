import { useQuery } from '@tanstack/react-query'
import { optimizeDesigns } from '../services/api'
import type { Constraints } from '../types'

export function useOptimization(constraints: Constraints) {
  // Only run optimization if at least one constraint is set
  const hasConstraints = Object.values(constraints).some(v => v !== undefined)

  return useQuery({
    queryKey: ['optimize', constraints],
    queryFn: () => optimizeDesigns({
      constraints,
      population_size: 200,
      n_generations: 100,
      n_designs: 50
    }),
    enabled: hasConstraints,
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: 1
  })
}
