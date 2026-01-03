# Week 4: React + TypeScript + Vite Frontend

## Summary

Complete React + TypeScript + Vite frontend for DOE Drone Design Optimizer with styling matching the existing DOE report.

## Features

### Interactive Constraint Form
- Sliders for 4 constraints: min_range_nm, max_cost_usd, max_mtow_lbm, min_endurance_hr
- Preset buttons: "Long Range", "Low Cost", "Balanced", "Clear All"
- Real-time validation with disabled state during optimization
- Default values pre-filled for quick testing

### 2D Pareto Charts (Plotly.js)
- Range vs MTOW and Range vs Cost with red star markers
- Hover tooltips with design details
- Exact colors from DOE report (#d32f2f pareto optimal)
- Responsive and interactive (zoom, pan)

### 3D Pareto Visualization
- Interactive 3D scatter: Range × Endurance × MTOW
- Color-coded by cost (blue gradient)
- Diamond markers for Pareto designs
- Rotatable camera with preset angle

### Sortable Design Table
- All 7 design parameters + 4 performance metrics
- Click column headers to sort
- Pagination: 10, 20, or 50 rows per page
- CSV export functionality

### API Integration
- React Query for state management
- Axios API client with TypeScript types
- 5-minute result caching
- Loading states and error handling

## Build Results

✅ TypeScript compilation successful
✅ Vite build successful (31.91s)
✅ 594 packages installed
✅ Bundle size: 5.2 MB (includes Plotly.js)

## Testing Instructions

### Start Backend (Terminal 1)
```bash
cd backend
python -m uvicorn app.main:app --reload
```

### Start Frontend (Terminal 2)
```bash
cd frontend
npm run dev
```

### Test at http://localhost:3000
1. Click "Balanced" preset
2. Click "RUN OPTIMIZATION"
3. Wait ~10-15 seconds
4. Verify charts and table render correctly
5. Test sorting, pagination, CSV export

🤖 Generated with [Claude Code](https://claude.com/claude-code)
