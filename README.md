# DOE Drone Design Optimizer

Multi-objective optimization web app for Group 3 drone designs using ML-based Pareto front exploration.

## Features
- Ensemble ML models (XGBoost + Neural Network) for performance prediction
- NSGA-II multi-objective optimizer
- Interactive Pareto front visualization (Plotly.js)
- FastAPI backend + React frontend
- Docker deployment

## Dataset
Based on 9,998 DOE samples from Latin Hypercube Sampling study

## Project Structure
```
DOE_Optimize/
├── backend/                    # FastAPI Python API
│   ├── app/
│   │   ├── models/            # ML models (XGBoost + Neural Net ensemble)
│   │   ├── optimization/      # NSGA-II multi-objective optimizer
│   │   ├── api/               # REST endpoints
│   │   ├── schemas/           # Pydantic data models
│   │   └── utils/             # Utility functions
│   ├── data/
│   │   ├── doe_summary.csv    # Training dataset (9,996 samples)
│   │   └── models/            # Trained model artifacts (.pkl files)
│   ├── tests/                 # Backend tests
│   └── scripts/               # Training and utility scripts
└── frontend/                  # React + TypeScript UI
    └── src/
        ├── components/        # Input forms, Plotly charts, design tables
        ├── services/          # API client (Axios)
        ├── types/             # TypeScript type definitions
        └── hooks/             # Custom React hooks
```

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (optional)

### Development Setup

#### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train models (Week 1)
python scripts/train_models.py

# Run API server
uvicorn app.main:app --reload
```
Backend API: http://localhost:8000
API Docs: http://localhost:8000/docs

#### Frontend
```bash
cd frontend
npm install
npm start
```
Frontend UI: http://localhost:3000

### Docker Deployment
```bash
docker-compose up
```
- Backend API: http://localhost:8000
- Frontend UI: http://localhost:3000
- API Docs: http://localhost:8000/docs

## Usage

1. Open the web app at http://localhost:3000
2. Set your target constraints:
   - Minimum range (nautical miles)
   - Maximum cost (USD)
   - Maximum MTOW (lbm)
   - Minimum endurance (hours)
3. Click "Optimize Designs"
4. Explore the Pareto front in interactive 2D/3D charts
5. Select designs to view detailed parameters
6. Download design specifications as JSON

## Technical Approach

### Machine Learning
- **Input:** 7 geometric parameters (LOA, span, sweep angles, panel break)
- **Output:** 4 primary metrics (range, endurance, MTOW, cost)
- **Models:** XGBoost (60%) + Neural Network (40%) ensemble
- **Uncertainty:** Standard deviation between model predictions

### Optimization
- **Algorithm:** NSGA-II (Non-dominated Sorting Genetic Algorithm)
- **Objectives:** Maximize range & endurance, minimize MTOW & cost
- **Constraints:** User-specified targets with smart relaxation
- **Output:** 20-50 Pareto-optimal designs

## API Endpoints

- `POST /api/predict` - Predict performance for given design parameters
- `POST /api/optimize` - Find Pareto-optimal designs given constraints
- `GET /api/design/{run_id}` - Get detailed data for specific DOE run
- `GET /api/health` - Health check and model status

## Development Roadmap

- [x] Week 0: Project setup, git initialization
- [ ] Week 1: Data & ML Pipeline (train ensemble models)
- [ ] Week 2: Optimization Engine (NSGA-II implementation)
- [ ] Week 3: Backend API (FastAPI endpoints)
- [ ] Week 4: Frontend UI (React + Plotly charts)
- [ ] Week 5: Integration & Testing
- [ ] Week 6: Deployment & Documentation

## Performance Targets

| Metric | Target |
|--------|--------|
| Model inference (single) | <50ms |
| Model inference (batch 100) | <200ms |
| NSGA-II optimization | <10s |
| API response time | <12s |
| Frontend initial load | <2s |

## References

- Dataset: `D:\nTop\DOERunner\gcp_10000sample_study_v3`
- Pareto logic: `generate_report.py`
- Visualization patterns: `doe_report.html`

## License

Internal nTop project

## Contact

For questions or issues, contact the nTop engineering team.
