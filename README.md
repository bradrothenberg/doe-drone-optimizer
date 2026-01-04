# DOE Drone Design Optimizer

Multi-objective optimization web app for Group 3 drone designs using ML-based Pareto front exploration.

## Features
- **ML Ensemble Models**: XGBoost + Neural Network for 5 performance metrics (range, endurance, MTOW, cost, wingtip deflection)
- **NSGA-II Optimizer**: Multi-objective optimization with geometric constraints
- **Interactive Visualization**: 2D/3D Pareto charts, planform viewer, design comparison
- **Sensitivity Analysis**: Tornado charts showing input-output relationships
- **Design Comparison**: Side-by-side radar charts for comparing designs
- **Export Options**: CSV export for nTop, PNG chart export
- **Docker Deployment**: Production-ready containerization

## Dataset
Based on 9,996 DOE samples from Latin Hypercube Sampling study (nTop parametric model)

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

#### Local Development
```bash
# Build and run with hot reload
docker-compose up --build

# Or run in detached mode
docker-compose up -d
```
- Backend API: http://localhost:8000
- Frontend UI: http://localhost:3000
- API Docs: http://localhost:8000/docs

#### Production Deployment
```bash
# Build and run production containers
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f
```
Production uses:
- Gunicorn with 4 Uvicorn workers
- nginx for static file serving
- Resource limits (2 CPU, 4GB RAM for backend)
- Auto-restart on failure

#### Cloud Deployment (GitHub Actions)
The repository includes a CI/CD pipeline (`.github/workflows/deploy.yml`) that:
1. Runs backend tests on every push/PR
2. Builds Docker images
3. Pushes to GitHub Container Registry (ghcr.io)

To deploy to your cloud provider:
1. Fork/clone the repository
2. Configure repository secrets for your cloud (AWS, Azure, or GCP)
3. Uncomment and configure the deploy job in the workflow

**AWS ECS Example:**
```yaml
# Add to .github/workflows/deploy.yml
- name: Configure AWS credentials
  uses: aws-actions/configure-aws-credentials@v4
  with:
    aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
    aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    aws-region: us-east-1

- name: Deploy to ECS
  run: aws ecs update-service --cluster my-cluster --service my-service --force-new-deployment
```

**Container Images:**
```bash
# Pull pre-built images (after CI/CD runs)
docker pull ghcr.io/YOUR_ORG/doe-drone-optimizer-backend:main
docker pull ghcr.io/YOUR_ORG/doe-drone-optimizer-frontend:main
```

## Usage

1. **Open the web app** at http://localhost:3000
2. **Configure optimization objectives** (for each metric: Range, Endurance, MTOW, Cost, Wingtip Deflection):
   - **MIN**: Minimize this metric
   - **MAX**: Maximize this metric
   - **LIMIT**: Set a hard constraint (slider appears to set threshold)
   - Uncheck to exclude from optimization
3. **Click "Optimize Designs"** to run NSGA-II optimization
4. **Explore results**:
   - 2D Pareto charts (Range vs Cost, MTOW vs Endurance, etc.)
   - 3D interactive Pareto visualization
   - Design table with sortable columns
5. **Click any design point** to view:
   - Wing planform visualization
   - Full parameter details
   - Sensitivity analysis (tornado chart)
6. **Compare designs**: Select 2-4 designs for radar chart comparison
7. **Export**:
   - CSV export for nTop (generates input file for parametric model)
   - PNG export from chart toolbar (camera icon)

## Technical Approach

### Machine Learning
- **Input:** 7 geometric parameters (LOA, span, LE/TE sweep angles for 2 panels, panel break)
- **Output:** 5 performance metrics (range, endurance, MTOW, cost, wingtip deflection)
- **Models:** XGBoost ensemble with engineered features
- **R² scores:** >0.85 for all outputs on test set

### Optimization
- **Algorithm:** NSGA-II (Non-dominated Sorting Genetic Algorithm)
- **Population:** 200 individuals, 100 generations
- **Objectives:** User-configurable (MIN/MAX/LIMIT for each metric)
- **Geometric constraints:**
  - Taper ratio enforcement (TE sweep - LE sweep ≥ -5°)
  - Bowtie wing filtering (minimum chord ≥ 2")
  - Wingtip deflection clamping (≥ 0)
- **Output:** Up to 50 Pareto-optimal designs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Predict performance for design parameters |
| `/api/optimize` | POST | Run NSGA-II optimization with constraints |
| `/api/sensitivity` | POST | Compute sensitivity analysis for a design |
| `/api/design/{run_id}` | GET | Get DOE data for specific run ID |
| `/health` | GET | Health check and model status |

Full API documentation available at http://localhost:8000/docs (Swagger UI)

## Development Roadmap

- [x] Week 0: Project setup, git initialization
- [x] Week 1: Data & ML Pipeline (XGBoost ensemble, feature engineering)
- [x] Week 2: Optimization Engine (NSGA-II with geometric constraints)
- [x] Week 3: Backend API (FastAPI, sensitivity analysis)
- [x] Week 4: Frontend UI (React + Plotly, design comparison)
- [x] Week 5: Docker containerization, CI/CD pipeline
- [x] Week 6: Polish, chart export, v1.0.0 release

## Performance Targets

| Metric | Target |
|--------|--------|
| Model inference (single) | <50ms |
| Model inference (batch 100) | <200ms |
| NSGA-II optimization | <10s |
| API response time | <12s |
| Frontend initial load | <2s |

## Input Parameters

| Parameter | Range | Unit | Description |
|-----------|-------|------|-------------|
| LOA | 96-192 | inches | Length Overall (nose to tail) |
| Span | 72-216 | inches | Wing span |
| LE Sweep P1 | 0-65 | degrees | Leading edge sweep, inner panel |
| LE Sweep P2 | -20-60 | degrees | Leading edge sweep, outer panel |
| TE Sweep P1 | -60-60 | degrees | Trailing edge sweep, inner panel |
| TE Sweep P2 | -60-60 | degrees | Trailing edge sweep, outer panel |
| Panel Break | 0.1-0.65 | fraction | Span fraction where panels meet |

## Output Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| Range | nm | Maximum flight range (nautical miles) |
| Endurance | hr | Maximum flight endurance (hours) |
| MTOW | lbm | Maximum Take-Off Weight |
| Cost | USD | Estimated manufacturing cost |
| Wingtip Deflection | in | Structural deflection at wingtip |

## References

- Dataset: `backend/data/doe_summary.csv` (9,996 Latin Hypercube samples)
- Original DOE study: nTop parametric model with FEA/CFD analysis

## License

MIT License

Copyright (c) 2024-2025 nTop Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contact

For questions or issues, contact the nTop engineering team.
