# DOE Drone Optimizer - Frontend

React + TypeScript + Vite frontend for the DOE Drone Design Optimizer.

## Features

- **Interactive Constraint Form**: Set min/max constraints with sliders
- **Preset Buttons**: Quick access to "Long Range", "Low Cost", "Balanced" configurations
- **2D Pareto Charts**: Range vs MTOW, Range vs Cost with Plotly.js
- **3D Pareto Visualization**: Interactive 3D scatter plot (Range × Endurance × MTOW)
- **Sortable Table**: View all Pareto-optimal designs with sorting and pagination
- **CSV Export**: Download designs for further analysis
- **Monospace Design**: Clean terminal aesthetic matching original DOE report

## Tech Stack

- **React 18.3** with TypeScript
- **Vite** for fast development and building
- **Material-UI (MUI)** for components
- **Plotly.js** for interactive charts
- **React Query** for API state management
- **Axios** for HTTP requests

## Development

### Install dependencies

```bash
npm install
```

### Run development server

```bash
npm run dev
```

Frontend will be available at http://localhost:3000

API requests proxy to backend at http://localhost:8000

### Build for production

```bash
npm run build
```

### Preview production build

```bash
npm run preview
```

## Project Structure

```
src/
├── components/
│   ├── Layout/
│   │   └── Header.tsx           # App header
│   ├── Input/
│   │   └── ConstraintForm.tsx   # Constraint sliders and presets
│   ├── Charts/
│   │   ├── chartConfig.ts       # Shared Plotly config
│   │   ├── ParetoChart2D.tsx    # 2D scatter charts
│   │   └── ParetoChart3D.tsx    # 3D scatter chart
│   └── Table/
│       └── DesignTable.tsx      # Sortable design table
├── hooks/
│   └── useOptimization.ts       # React Query hook
├── services/
│   └── api.ts                   # Axios API client
├── theme/
│   └── colors.ts                # Color constants from DOE report
├── types/
│   └── index.ts                 # TypeScript interfaces
├── App.tsx                      # Main app component
├── main.tsx                     # Entry point
└── index.css                    # Global styles

```

## API Integration

The frontend connects to the FastAPI backend via `/api` proxy:

- `POST /api/optimize` - Run NSGA-II optimization with constraints
- `POST /api/predict` - Get predictions for specific designs
- `GET /api/health` - Check backend health and model status

## Design System

Colors and typography match the original DOE report:

- **Fonts**: Courier New, IBM Plex Mono (monospace)
- **Colors**: Black/white/gray palette with minimal accents
- **Pareto markers**: Red stars (#d32f2f) for optimal designs
- **Borders**: 2px solid black borders on cards and headers

See [DESIGN.md](./DESIGN.md) for full design specification.

## Performance

- Charts render <500ms for 1000 points
- Optimization requests have 35s timeout
- Results cached for 5 minutes (React Query)
- Responsive design scales to mobile
