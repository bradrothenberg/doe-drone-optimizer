# Frontend Design Specification

## Extracted Styles from DOE Report

### Color Scheme

**Background & Text:**
- Background: `#ffffff` (white)
- Text: `#000000` (black)
- Secondary text/labels: `#666666` (gray)
- Card background: `#f5f5f5` (light gray)
- Borders: `#cccccc` (medium gray), `#e0e0e0` (light gray)

**Chart Colors:**

1. **Range colorscale** (red-yellow-green gradient):
   - Low: `#ffcccc` (light red)
   - Mid: `#ffffcc` (light yellow)
   - High: `#c8e6c9` (light green)

2. **MTOW colorscale** (blue gradient):
   - Low: `#e3f2fd` (light blue)
   - High: `#1565c0` (dark blue)

3. **Pareto markers:**
   - Pareto optimal: `#d32f2f` (red), border `#b71c1c` (dark red)
   - Dominated designs: `#cccccc` (light gray), border `#999999` (medium gray)

### Typography

**Fonts:**
- Primary: `'Courier New', 'Consolas', 'Monaco', monospace`
- Headers: `'IBM Plex Mono', monospace`
- Chart text: `Courier New, monospace`

**Font Sizes:**
- Main heading (h1): `2em` (600 weight)
- Section heading (h2): `1.1em` (400 weight)
- Card heading (h3): `1.2em` (600 weight)
- Chart title: `13px`
- Chart axis labels: `10px`
- Chart tick labels: `10px`
- Colorbar ticks: `9px`
- Legend: `9px`
- Metric values: `1.8em` (bold)
- Metric labels: `0.8em`
- Info labels: `0.75em`

### Chart Configuration

**Base Layout (chartLayout):**
```javascript
{
  paper_bgcolor: '#ffffff',
  plot_bgcolor: '#ffffff',
  font: { family: 'Courier New, monospace', color: '#000000', size: 11 },
  margin: { t: 30, r: 20, b: 50, l: 60 },
  hovermode: 'closest',
  xaxis: {
    gridcolor: '#e0e0e0',
    linecolor: '#cccccc',
    tickfont: { size: 10 }
  },
  yaxis: {
    gridcolor: '#e0e0e0',
    linecolor: '#cccccc',
    tickfont: { size: 10 }
  }
}
```

**Config (chartConfig):**
```javascript
{
  displayModeBar: false,
  responsive: true
}
```

### 2D Scatter Chart Pattern

**Dominated designs trace:**
```javascript
{
  x: data.map(d => d.mtow),
  y: data.map(d => d.range),
  mode: 'markers',
  type: 'scatter',
  name: 'Dominated',
  customdata: data.map(d => d.run_id),
  marker: {
    size: 4,
    color: '#cccccc',
    line: { color: '#999999', width: 0.5 }
  },
  hovertemplate: '<b>Run %{customdata}</b><br>MTOW: %{x:.0f} lbm<br>Range: %{y:.0f} nm<extra></extra>'
}
```

**Pareto optimal trace:**
```javascript
{
  x: paretoData.map(d => d.mtow),
  y: paretoData.map(d => d.range),
  mode: 'markers+lines',
  type: 'scatter',
  name: 'Pareto Optimal',
  customdata: paretoData.map(d => d.run_id),
  marker: {
    size: 10,
    symbol: 'star',
    color: '#d32f2f',
    line: { color: '#b71c1c', width: 1 }
  },
  line: { color: '#d32f2f', width: 2, dash: 'dot' },
  hovertemplate: '<b>Run %{customdata} *</b><br>MTOW: %{x:.0f} lbm<br>Range: %{y:.0f} nm<extra></extra>'
}
```

### 3D Scatter Chart Pattern

**Dominated designs:**
```javascript
{
  x: dominatedData.map(d => d.mtow),
  y: dominatedData.map(d => d.range),
  z: dominatedData.map(d => d.cost),
  mode: 'markers',
  type: 'scatter3d',
  name: 'Dominated',
  customdata: dominatedData.map(d => d.run_id),
  marker: {
    size: 2,
    color: '#cccccc',
    opacity: 0.5
  },
  hovertemplate: '<b>Run %{customdata}</b><br>MTOW: %{x:.0f}<br>Range: %{y:.0f}<br>Cost: $%{z:,.0f}<extra></extra>'
}
```

**Pareto optimal:**
```javascript
{
  x: paretoData.map(d => d.mtow),
  y: paretoData.map(d => d.range),
  z: paretoData.map(d => d.cost),
  mode: 'markers',
  type: 'scatter3d',
  name: 'Pareto Optimal',
  customdata: paretoData.map(d => d.run_id),
  marker: {
    size: 5,
    color: '#d32f2f',
    symbol: 'diamond'
  },
  hovertemplate: '<b>Run %{customdata} *</b><br>MTOW: %{x:.0f}<br>Range: %{y:.0f}<br>Cost: $%{z:,.0f}<extra></extra>'
}
```

**3D Scene Layout:**
```javascript
scene: {
  xaxis: { title: 'MTOW (lbm)', titlefont: { size: 10 }, tickfont: { size: 8 } },
  yaxis: { title: 'Range (nm)', titlefont: { size: 10 }, tickfont: { size: 8 } },
  zaxis: { title: 'Cost ($)', titlefont: { size: 10 }, tickfont: { size: 8 } },
  camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } }
},
margin: { t: 40, r: 10, b: 10, l: 10 }
```

### Colorscale Patterns

**For continuous color mapping:**
```javascript
// Range (red-yellow-green)
marker: {
  color: data.map(d => d.range),
  colorscale: [[0, '#ffcccc'], [0.5, '#ffffcc'], [1, '#c8e6c9']],
  colorbar: { title: 'Range', thickness: 12, tickfont: { size: 9 } },
  line: { color: '#000000', width: 0.5 }
}

// MTOW (blue gradient)
marker: {
  color: data.map(d => d.mtow),
  colorscale: [[0, '#e3f2fd'], [1, '#1565c0']],
  colorbar: { title: 'MTOW', thickness: 12, tickfont: { size: 9 } },
  line: { color: '#000000', width: 0.5 }
}
```

### Interactive Features

1. **Click handlers:** `plotly_click` event to select designs
2. **Hover tooltips:** Custom templates with design ID and metrics
3. **Legend:** Positioned at top-left (x: 0.02, y: 0.98) with small font (9px)
4. **Responsive:** All charts resize with container

## Component Structure

```
src/
├── components/
│   ├── Layout/
│   │   ├── Header.tsx          # App header with title
│   │   ├── Card.tsx            # Reusable card wrapper
│   │   └── MetricDisplay.tsx   # Metric value/label display
│   ├── Input/
│   │   ├── ConstraintForm.tsx  # Constraint input sliders
│   │   └── PresetButtons.tsx   # "Long Range", "Low Cost", "Balanced"
│   ├── Charts/
│   │   ├── ParetoChart2D.tsx   # 2D scatter (Range vs MTOW, Range vs Cost)
│   │   ├── ParetoChart3D.tsx   # 3D scatter (Range × MTOW × Cost)
│   │   └── chartConfig.ts      # Shared Plotly config
│   ├── Table/
│   │   ├── DesignTable.tsx     # Sortable, paginated table
│   │   └── DesignRow.tsx       # Individual design row
│   └── Modal/
│       └── DesignDetail.tsx    # Full design details modal
├── services/
│   └── api.ts                  # Axios API client
├── hooks/
│   ├── useOptimization.ts      # React Query hook for optimization
│   └── usePrediction.ts        # React Query hook for prediction
├── types/
│   └── index.ts                # TypeScript interfaces
├── theme/
│   └── colors.ts               # Color constants
└── App.tsx                     # Main application
```

## Implementation Notes

1. **Use exact color hex codes** from the DOE report for consistency
2. **Monospace fonts** throughout to match terminal aesthetic
3. **Minimal styling** - no gradients, shadows, or animations
4. **Black borders** on all markers for definition
5. **Small markers** for dominated designs (size: 2-4), **large markers** for Pareto (size: 5-10)
6. **Star/diamond symbols** for Pareto optimal designs
7. **Responsive charts** that fill their container
8. **No mode bar** on charts (displayModeBar: false)
9. **Hover mode: 'closest'** for best UX
10. **Currency formatting** with commas for cost values
