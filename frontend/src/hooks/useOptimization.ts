import { useQuery } from '@tanstack/react-query'
import { optimizeDesigns } from '../services/api'
import type { Constraints, OptimizationObjectives } from '../types'

export function useOptimization(constraints: Constraints, objectives: OptimizationObjectives, enabled: boolean) {
  return useQuery({
    queryKey: ['optimize', constraints, objectives],
    queryFn: () => optimizeDesigns({
      constraints,
      objectives,
      population_size: 200,
      n_generations: 100,
      n_designs: 50
    }),
    enabled: enabled,
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: 1
  })
}
