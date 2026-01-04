# PR #5 Review: Week 4 - React + TypeScript + Vite Frontend

## Overview

This PR adds wingtip deflection as a 5th model output and introduces a complete React frontend for the DOE Drone Design Optimizer. The implementation is generally well-structured with good TypeScript usage and clean component architecture, but there are several issues that should be addressed before merging.

---

## 🔴 Critical Issues

### 1. Missing `max_wingtip_deflection_in` in App.tsx initial state

**File:** `frontend/src/App.tsx` (lines 12-17)

The `max_wingtip_deflection_in` constraint is used in the ConstraintForm but not included in the initial state:

```typescript
const [constraints, setConstraints] = useState<Constraints>({
  min_range_nm: undefined,
  max_cost_usd: undefined,
  max_mtow_lbm: undefined,
  min_endurance_hr: undefined
  // ❌ Missing: max_wingtip_deflection_in: undefined
})
```

**Impact:** The wingtip deflection constraint may not be properly tracked in state.

### 2. Missing `max_wingtip_deflection_in` in handleClear function

**File:** `frontend/src/components/Input/ConstraintForm.tsx` (lines 78-84)

The "Clear All" button doesn't reset the wingtip deflection constraint:

```typescript
const handleClear = () => {
  onUpdate({
    min_range_nm: undefined,
    max_cost_usd: undefined,
    max_mtow_lbm: undefined,
    min_endurance_hr: undefined
    // ❌ Missing: max_wingtip_deflection_in: undefined
  })
}
```

**Impact:** Clicking "Clear All" leaves wingtip deflection constraint active.

### 3. Extremely High Uncertainty Values for Wingtip Deflection

**File:** `backend/data/models/training_results.json`

The uncertainty statistics for wingtip deflection are clearly broken:

```json
"wingtip_deflection_in": {
  "mean": 12003114.0,
  "median": 10084304.0,
  "std": 10503950.0,
  "max": 49909348.0
}
```

These values (12 million inches mean uncertainty!) indicate the neural network's uncertainty estimation is failing for this output, likely due to the large value range before/after clamping.

**Impact:** Uncertainty estimates shown to users will be meaningless for wingtip deflection.

---

## 🟡 Medium Priority Issues

### 4. No Debouncing on Slider Changes

**File:** `frontend/src/components/Input/ConstraintForm.tsx`

Every slider drag event fires `onUpdate`, which updates state and potentially triggers re-renders. While React Query's `staleTime` prevents excessive API calls, consider adding debounce:

```typescript
import { useDebouncedCallback } from 'use-debounce'

const debouncedUpdate = useDebouncedCallback(onUpdate, 300)
```

### 5. Inconsistent Penalty Weights in NSGA Optimization

**File:** `backend/app/optimization/nsga.py` (lines 119-121)

```python
if 'max_wingtip_deflection_in' in self.user_constraints:
    max_deflection = self.user_constraints['max_wingtip_deflection_in']
    violation = np.maximum(0, wingtip_deflection_in - max_deflection)
    penalty += violation * 10.0  # ⚠️ Why 10.0 vs 100 for other constraints?
```

Other constraints use penalty multiplier of 100. The choice of 10.0 for deflection isn't documented.

**Suggestion:** Add a comment explaining the rationale, or use consistent weights.

### 6. Hardcoded API Base URL

**File:** `frontend/src/services/api.ts` (line 5)

```typescript
const api = axios.create({
  baseURL: '/api',  // Hardcoded
  ...
})
```

**Suggestion:** Use environment variable for flexibility:
```typescript
baseURL: import.meta.env.VITE_API_URL || '/api',
```

### 7. Type Safety in DesignTable Sorting

**File:** `frontend/src/components/Table/DesignTable.tsx` (lines 35-38)

```typescript
const sortedDesigns = [...designs].sort((a, b) => {
  const aVal = a[sortKey] as number  // Unsafe type assertion
  const bVal = b[sortKey] as number
  return sortOrder === 'asc' ? aVal - bVal : bVal - aVal
})
```

If `sortKey` accidentally points to a non-numeric field, this could fail silently or produce incorrect results.

---

## 🟢 Minor Suggestions

### 8. Consider React.memo for Planform Component

**File:** `frontend/src/components/Visualization/Planform.tsx`

The Planform component performs geometric calculations on every render. Wrap in `React.memo` to prevent unnecessary recalculations:

```typescript
export default React.memo(function Planform({ design, width = 200, height = 150 }: PlanformProps) {
  // ...
})
```

### 9. Loading State UX

**File:** `frontend/src/App.tsx` (lines 71-75)

Currently shows only text during optimization. Consider adding a progress indicator or skeleton loader for better UX.

### 10. Accessibility

No ARIA labels on sliders or interactive elements. Consider adding:
- `aria-label` on sliders
- `aria-busy` on loading states
- Keyboard navigation support verification

---

## ✅ What's Good

1. **Clean Component Architecture** - Good separation of concerns between Layout, Charts, Input, Table, and Visualization components

2. **TypeScript Usage** - Comprehensive type definitions in `types/index.ts` matching backend schemas

3. **React Query Integration** - Proper use of `useQuery` with caching (`staleTime: 5 minutes`) and error handling

4. **SVG Planform Visualization** - Clever implementation with proper wing geometry calculations and bowtie prevention

5. **Data Clamping** - Smart handling of failed simulation outliers in `data_loader.py`:
   ```python
   MAX_WINGTIP_DEFLECTION = 100.0
   self.df_clean['wingtip_deflection_in'] = self.df_clean['wingtip_deflection_in'].clip(upper=100.0)
   ```

6. **Consistent Backend Updates** - All schemas, API endpoints, and optimization code properly updated for 5th output

7. **Preset Configurations** - Good UX with "Long Range", "Low Cost", "Balanced" presets

8. **CSV Export** - Clean implementation for nTop input parameters

---

## Testing Recommendations

1. **Unit Tests for Planform Geometry** - Test sweep angle calculations and bowtie prevention logic

2. **Constraint Form Edge Cases** - Test with missing constraints, extreme values

3. **API Integration Tests** - Verify all 5 constraints handled correctly in optimization endpoint

4. **CSV Export Validation** - Verify exported values match displayed values

---

## Summary

| Priority | Issue | Status |
|----------|-------|--------|
| 🔴 Critical | Missing `max_wingtip_deflection_in` in initial state | Must fix |
| 🔴 Critical | Missing `max_wingtip_deflection_in` in handleClear | Must fix |
| 🔴 Critical | Broken uncertainty values for wingtip deflection | Should investigate |
| 🟡 Medium | No debouncing on sliders | Recommended |
| 🟡 Medium | Inconsistent penalty weights | Should document |
| 🟡 Medium | Hardcoded API URL | Recommended |
| 🟢 Minor | React.memo, accessibility, loading UX | Nice to have |

**Recommendation:** Fix the critical issues (missing constraint in state/clear) before merging. The uncertainty values for wingtip deflection should also be investigated as they appear to be broken.
