# Improving the ML Model: GPU Training & Modern Architectures

## Executive Summary

This document outlines a plan to improve the DOE drone optimizer ML models by:
1. **Adding TabPFN** (state-of-the-art tabular foundation model) to the existing ensemble
2. **Optimizing GPU training** for faster iteration cycles
3. **Benchmarking performance** improvements

**Key Finding**: Your neural network already has GPU support. For your dataset size (~10k samples), GPU provides minimal speedup (2-5 min → <1 min). The bigger opportunity is adding TabPFN, which shows 100% win rate vs XGBoost on datasets <10k samples.

## Current Implementation

### Architecture Overview
- **Framework**: PyTorch MLP + XGBoost ensemble
- **Neural Network**: 3-layer MLP (input→128→64→32→5 outputs)
  - Dropout: 0.2, BatchNorm, ReLU activations
  - Optimizer: Adam (lr=0.001, weight_decay=1e-4)
  - Scheduler: ReduceLROnPlateau
  - Early stopping: patience=30 epochs
- **Training Config**:
  - Epochs: 300 (typically stops at 100-150 with early stopping)
  - Batch size: 32
  - Dataset: 9,996 samples (70% train, 15% val, 15% test)
- **Ensemble**: 40% NN + 60% XGBoost (weights optimized on validation set)

### GPU Support Status
✅ **Already Implemented** in `backend/app/models/neural_model.py:119-124`
```python
if device is None:
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    self.device = torch.device(device)
logger.info(f"Using device: {self.device}")
```

### Performance Metrics
The model predicts 5 drone performance metrics:
1. `range_nm` - Range in nautical miles
2. `endurance_hr` - Endurance in hours
3. `mtow_lbm` - Max Takeoff Weight
4. `material_cost_usd` - Material cost
5. `wingtip_deflection_in` - Wingtip deflection

## GPU Training Analysis

### Expected Speedup for Your Dataset (~10k samples)
**Verdict**: GPU provides marginal speedup (2-5x)

**Reasons**:
1. **Small dataset**: ~7,000 training samples fit entirely in memory
2. **Transfer overhead**: GPU data transfer time offsets computational gains
3. **Batch size**: At batch_size=32, only ~220 batches per epoch
4. **Model size**: Small MLP (17→128→64→32→5) has minimal compute requirements

**Estimated Training Times**:
- CPU: 2-5 minutes (300 epochs with early stopping)
- GPU (CUDA): <1 minute
- **Speedup**: 2-5x (saves 1-4 minutes per training run)

### When GPU Training Helps
- Larger datasets (>100k samples)
- Deeper neural networks
- Larger batch sizes
- Multiple training runs (hyperparameter tuning)

## Modern ML Models for Tabular Data (2025)

### TabPFN (Tabular Prior-data Fitted Network) - RECOMMENDED

**Status**: Published in Nature (January 2025), state-of-the-art for small/medium datasets

**Why TabPFN is Perfect for This Use Case**:
- Designed for datasets with <10,000 samples (your exact scenario)
- No hyperparameter tuning required (foundation model)
- 100% win rate vs default XGBoost on datasets <10k samples
- Fast inference via in-context learning
- Latest version (TabPFN-2.5) handles up to 100k rows, 2k features

**Performance Benchmarks**:
- Outperforms XGBoost by wide margin on <10k samples
- Training time: 2.8 seconds (vs 4 hours for tuned ensemble in literature)
- 92.3% accuracy on tiny datasets (<1,000 samples)
- 87% win rate on datasets up to 100k samples

**Limitations**:
- Memory constraints on very large datasets (>100k rows)
- May need to train separate model per output for multi-output regression

### Other Modern Approaches

**TabNet** (Google's attention-based model):
- Interpretable feature selection
- Slower training than GBDT methods
- May not beat well-tuned XGBoost

**CatBoost/LightGBM** (Alternative GBDT):
- CatBoost: Better categorical feature handling
- LightGBM: Faster on large datasets
- Both often match XGBoost performance

## Implementation Plan

### Phase 1: Add TabPFN Model (Accuracy Improvement)

#### Step 1.1: Create TabPFN Model Wrapper
**File**: Create `backend/app/models/tabpfn_model.py`

**Implementation**:
- Wrap TabPFN API for consistency with existing models
- Provide `fit()`, `predict()`, `evaluate()`, `save()`, `load()` methods
- Handle multi-output regression (5 outputs)
- Add proper logging and error handling

**Key Design Decisions**:
- TabPFN may not support multi-output directly → train 5 separate models (one per output)
- Use TabPFN's default settings (no hyperparameter tuning needed)
- Cache fitted models for fast inference

**Example API**:
```python
from tabpfn import TabPFNRegressor

class TabPFNDroneModel:
    def __init__(self):
        # One model per output
        self.models = [TabPFNRegressor() for _ in range(5)]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train one TabPFN model per output"""
        for i, model in enumerate(self.models):
            model.fit(X_train, y_train[:, i])

    def predict(self, X):
        """Predict all outputs"""
        predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        return predictions

    def evaluate(self, X, y, output_names):
        """Evaluate on test set"""
        # Similar to existing models
        pass
```

#### Step 1.2: Update Training Script
**File**: `backend/scripts/train_models.py`

**Changes**:
1. Import TabPFN model wrapper
2. Add TabPFN training step after XGBoost and Neural Network
3. Train TabPFN on engineered features (same as XGBoost)
4. Evaluate TabPFN performance on test set
5. Update ensemble optimization to include 3 models
6. Save TabPFN models (5 separate models for 5 outputs)

**Updated Training Flow**:
```
[1] Load data
[2] Engineer features
[3] Train XGBoost → save
[4] Train Neural Network → save
[5] Train TabPFN → save (NEW)
[6] Optimize 3-model ensemble weights
[7] Evaluate ensemble on test set
[8] Save results
```

**Code Changes**:
```python
# After line 135 (after NN evaluation)
# ========================================
# 5. TRAIN TABPFN MODEL
# ========================================
logger.info("\n[5/7] Training TabPFN model...")

from app.models.tabpfn_model import train_tabpfn_model

tabpfn_model, tabpfn_val_metrics = train_tabpfn_model(
    X_train_eng, loader.y_train,
    X_val_eng, loader.y_val,
    output_dim=loader.y_train.shape[1]
)

# Evaluate TabPFN on test set
tabpfn_test_metrics = tabpfn_model.evaluate(
    X_test_eng, loader.y_test, loader.get_output_names()
)

logger.info("\nTabPFN Test Results:")
logger.info(f"  Overall R²: {tabpfn_test_metrics['overall']['r2']:.4f}")
for name, metrics in tabpfn_test_metrics['per_output'].items():
    logger.info(f"    {name}: {metrics['r2']:.4f} (MAE: {metrics['mae']:.2f})")
```

#### Step 1.3: Update Ensemble Logic
**File**: `backend/app/models/ensemble.py`

**Changes**:
1. Extend `optimize_ensemble_weights()` to support 3 models
2. Grid search over weight combinations: (w_xgb, w_nn, w_tabpfn) where sum=1
3. Return 3 weights instead of 2
4. Update `EnsembleModel` class to handle 3 models
5. Update `predict()` to combine 3 model predictions

**Weight Optimization Strategy**:
```python
def optimize_ensemble_weights_3models(
    xgb_model, nn_model, tabpfn_model,
    X_val, y_val,
    X_val_scaled=None
):
    """Optimize weights for 3-model ensemble"""
    best_r2 = -np.inf
    best_weights = (0.33, 0.33, 0.34)

    # Grid search over weight space
    # Test 66 combinations (11 x 6 grid where weights sum to 1)
    for w_xgb in np.linspace(0, 1, 11):
        for w_nn in np.linspace(0, 1 - w_xgb, 6):
            w_tabpfn = 1 - w_xgb - w_nn

            # Compute ensemble predictions
            pred_xgb = xgb_model.predict(X_val)
            pred_nn = nn_model.predict(X_val_scaled)
            pred_tabpfn = tabpfn_model.predict(X_val)

            ensemble_pred = (w_xgb * pred_xgb +
                           w_nn * pred_nn +
                           w_tabpfn * pred_tabpfn)

            # Evaluate
            r2 = r2_score(y_val, ensemble_pred)

            if r2 > best_r2:
                best_r2 = r2
                best_weights = (w_xgb, w_nn, w_tabpfn)

    return best_weights
```

#### Step 1.4: Update Model Manager
**File**: `backend/app/core/model_manager.py`

**Changes**:
1. Load TabPFN models on initialization
2. Handle 3-model ensemble predictions
3. Update caching logic for TabPFN

**Code Changes**:
```python
class ModelManager:
    def __init__(self):
        # Load existing models
        self.xgb_model = load_xgboost_model(...)
        self.nn_model = load_neural_model(...)

        # Load TabPFN model (NEW)
        self.tabpfn_model = load_tabpfn_model(...)

        # Load ensemble metadata (now with 3 weights)
        self.ensemble_weights = load_ensemble_weights()  # (w_xgb, w_nn, w_tabpfn)

    def predict(self, X):
        """Ensemble prediction with 3 models"""
        pred_xgb = self.xgb_model.predict(X)
        pred_nn = self.nn_model.predict(X_scaled)
        pred_tabpfn = self.tabpfn_model.predict(X)

        w_xgb, w_nn, w_tabpfn = self.ensemble_weights

        return (w_xgb * pred_xgb +
                w_nn * pred_nn +
                w_tabpfn * pred_tabpfn)
```

### Phase 2: GPU Training Optimization (Speed Improvement)

#### Step 2.1: Run Training on GPU Machine
**Action**: Execute training script on GPU machine

```bash
# SSH to GPU machine
ssh 192.168.20.178  # NVIDIA SPARK

# Navigate to project
cd /path/to/DOE_Optimize

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"

# Run training
cd backend
uv run python scripts/train_models.py
```

#### Step 2.2: Add Training Time Benchmarks
**File**: `backend/scripts/train_models.py`

**Changes**: Add timing for each training phase

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    """Context manager for timing"""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{name} took {elapsed:.2f} seconds")

# Use in training script
with timer("XGBoost training"):
    xgb_model, xgb_val_metrics = train_xgboost_model(...)

with timer("Neural Network training"):
    nn_model, nn_val_metrics = train_neural_network_model(...)

with timer("TabPFN training"):
    tabpfn_model, tabpfn_val_metrics = train_tabpfn_model(...)
```

#### Step 2.3: Log Device Information
**File**: `backend/scripts/train_models.py`

**Add at start of main()**:
```python
# Log device information
logger.info("\nDevice Information:")
logger.info(f"  PyTorch version: {torch.__version__}")
logger.info(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"  CUDA version: {torch.version.cuda}")
    logger.info(f"  GPU device: {torch.cuda.get_device_name(0)}")
    logger.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    logger.info("  Running on CPU")
```

### Phase 3: Testing & Validation

#### Step 3.1: Validate TabPFN Integration
**Tests**:
1. Verify TabPFN trains on dataset
2. Check prediction output shapes (n_samples, 5)
3. Compare R² scores: XGBoost vs Neural Network vs TabPFN
4. Validate ensemble predictions are reasonable
5. Test model saving/loading

#### Step 3.2: Performance Comparison
**Metrics to Track**:
- Overall R² (target: >0.90)
- Per-output R² for each metric (range, endurance, mtow, cost, deflection)
- Training time per model (CPU vs GPU)
- Ensemble weight distribution
- Validation vs test performance gap

**Create Comparison Report**:
```python
# In train_models.py, add summary table
logger.info("\n" + "="*80)
logger.info("MODEL COMPARISON SUMMARY")
logger.info("="*80)

comparison_data = {
    'Model': ['XGBoost', 'Neural Network', 'TabPFN', 'Ensemble (3-model)'],
    'Overall R²': [xgb_r2, nn_r2, tabpfn_r2, ensemble_r2],
    'Training Time': [xgb_time, nn_time, tabpfn_time, total_time],
    'Weight': [w_xgb, w_nn, w_tabpfn, 1.0]
}

# Log as table
# ...
```

#### Step 3.3: Run Full Training Pipeline
**Checklist**:
1. ✓ Install TabPFN: `uv pip install tabpfn`
2. ✓ Create TabPFN model wrapper
3. ✓ Update training script
4. ✓ Update ensemble logic
5. ✓ Run training on CPU (baseline)
6. ✓ Run training on GPU (192.168.20.178)
7. ✓ Compare results
8. ✓ Update model manager for inference
9. ✓ Test API with new ensemble

## Expected Outcomes

### Accuracy Improvements
- **TabPFN standalone**: Expected to outperform XGBoost on this dataset size
- **3-model ensemble**: Combining diverse model types should improve R²
- **Target**: R² > 0.90 (may already be met, but should be more robust)

### Speed Improvements
- **Neural Network on GPU**: 2-5 min → <1 min (2-5x speedup)
- **TabPFN training**: Very fast (2-3 seconds expected)
- **Total training time**: <2 minutes on GPU (down from 3-7 minutes on CPU)

### Production Benefits
- Faster model iteration cycles
- More robust predictions via 3-model ensemble
- Better handling of edge cases (TabPFN excels on small datasets)
- Uncertainty quantification via model disagreement

## Rollback Plan

If TabPFN integration fails or degrades performance:
1. Keep TabPFN model code for future experiments
2. Revert ensemble to XGBoost + Neural Network only
3. Document failure mode in this file
4. Consider alternatives (CatBoost, LightGBM)

## Dependencies

### Python Packages
```bash
# Install TabPFN
uv pip install tabpfn

# May need to upgrade scikit-learn
uv pip install --upgrade scikit-learn>=1.0
```

### Hardware Requirements
- **GPU machine**: ssh 192.168.20.178 (NVIDIA SPARK)
- CUDA toolkit installed
- PyTorch with CUDA support (should already be installed)

### Compatibility Notes
- TabPFN requires scikit-learn API compatibility
- Check TabPFN version for multi-output regression support
- May need to handle each output separately if not supported

## Critical Files to Modify

1. **Create new**: `backend/app/models/tabpfn_model.py` (TabPFN wrapper)
2. **Modify**: `backend/scripts/train_models.py` (add TabPFN training)
3. **Modify**: `backend/app/models/ensemble.py` (3-model ensemble logic)
4. **Modify**: `backend/app/core/model_manager.py` (load/use TabPFN)

## References

### Tabular Foundation Models
- [Accurate predictions on small data with a tabular foundation model | Nature](https://www.nature.com/articles/s41586-024-08328-6)
- [TabPFN: The AI Revolution for Tabular Data](https://www.inwt-statistics.com/blog/tabpfn-the-ai-revolution-for-tabular-data)
- [Benchmarking TabPFN against XGboost and Catboost](https://blog.humblebee.ai/blog/2025/11/23/benchmarking-tabpfn-against-xgboost-and-catboost-on-kaggle-datasets/)
- [TabPFN-2.5: Advancing the State of the Art (arXiv)](https://arxiv.org/abs/2511.08667)

### GPU Training Performance
- [Training speed comparison: GPU approximation](https://www.neuraldesigner.com/blog/training-speed-comparison-gpu-approximation/)
- [PyTorch GPU vs CPU discussion](https://discuss.pytorch.org/t/the-training-speed-of-between-gpu-and-cpu-is-same/1241)
- [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

## Timeline Estimate

**Phase 1** (TabPFN Integration): 2-4 hours
- Create TabPFN wrapper: 1 hour
- Update training script: 1 hour
- Update ensemble logic: 1-2 hours
- Update model manager: 30 minutes

**Phase 2** (GPU Optimization): 30 minutes
- Run on GPU machine: 15 minutes
- Add benchmarks: 15 minutes

**Phase 3** (Testing): 1-2 hours
- Validation tests: 1 hour
- Performance comparison: 30 minutes
- Documentation: 30 minutes

**Total**: 4-7 hours of implementation time

## Next Steps

1. Install TabPFN: `uv pip install tabpfn`
2. Create `backend/app/models/tabpfn_model.py` wrapper
3. Test TabPFN on small sample of data
4. Integrate into full training pipeline
5. Run comparative benchmarks
6. Update production model manager
7. Deploy to API if results are positive

## Questions/Notes

- Does TabPFN support multi-output regression natively, or do we need 5 separate models?
- Should we keep XGBoost in the ensemble if TabPFN outperforms it significantly?
- Consider adding CatBoost/LightGBM as alternative GBDT methods for future experiments
- Monitor memory usage with TabPFN (may be higher than tree methods)
