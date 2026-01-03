# PR Review: Week 2 - NSGA-II Multi-Objective Optimization Engine

**PR:** https://github.com/bradrothenberg/doe-drone-optimizer/pull/2
**Verdict:** Request Changes

---

## Summary

This PR introduces a well-structured multi-objective optimization engine using NSGA-II for drone design. The architecture is clean with good separation of concerns across modules. However, there are critical bugs that need to be fixed before this can be approved.

---

## Critical Issues (Must Fix)

### 1. Bug: Corrupted Pareto Objective Values

**Location:** `nsga.py:113-118, 186-189`

The returned objective values are corrupted by penalties, making them incorrect.

**Problem:**
```python
# In _evaluate() (line 113-118):
objectives = np.column_stack([
    -range_nm + penalty,      # Stored value = -range + penalty
    -endurance_hr + penalty,
    mtow_lbm + penalty,
    cost_usd + penalty
])

# In run_nsga2_optimization() (line 186-189):
range_nm = -pareto_objectives[:, 0]  # = -(-range + penalty) = range - penalty  ❌
```

The code already fetches correct predictions on line 198-199 but doesn't use them:
```python
X_eng = feature_engineer.transform(pareto_designs)
predictions, uncertainty = ensemble_model.predict(X_eng, return_uncertainty=True)
```

**Fix:** Use the fresh `predictions` array instead of the penalty-corrupted `pareto_objectives`:
```python
results = {
    'pareto_designs': pareto_designs,
    'pareto_objectives': {
        'range_nm': predictions[:, 0],       # Use fresh predictions
        'endurance_hr': predictions[:, 1],
        'mtow_lbm': predictions[:, 2],
        'cost_usd': predictions[:, 3]
    },
    ...
}
```

---

### 2. Bug: Warning Treated as Validation Failure

**Location:** `constraints.py:355-358`

```python
# Check for conflicting constraints
if 'min_range_nm' in constraints and 'max_cost_usd' in constraints:
    if constraints['min_range_nm'] > 3000 and constraints['max_cost_usd'] < 20000:
        errors.append("Warning: High range requirement with low cost limit may be infeasible")

return len(errors) == 0, errors  # ❌ Warning causes validation failure
```

**Problem:** Warnings are appended to the `errors` list, causing `is_valid` to return `False` for valid-but-challenging constraints.

**Fix:** Separate warnings from errors:
```python
def validate_constraints(constraints: Dict[str, float]) -> Tuple[bool, List[str], List[str]]:
    errors = []
    warnings = []

    # ... validation logic ...

    if 'min_range_nm' in constraints and 'max_cost_usd' in constraints:
        if constraints['min_range_nm'] > 3000 and constraints['max_cost_usd'] < 20000:
            warnings.append("High range requirement with low cost limit may be infeasible")

    return len(errors) == 0, errors, warnings
```

---

### 3. Security: Pickle Deserialization Risk

**Location:** `neural_model.py:381`

```python
checkpoint = torch.load(input_path, map_location=self.device, weights_only=False)
```

**Problem:** Using `weights_only=False` enables arbitrary code execution via pickle. This becomes a remote code execution vulnerability if model paths ever become user-controlled.

**Fix:** Use `weights_only=True` (PyTorch 2.0+) or validate model file integrity:
```python
checkpoint = torch.load(input_path, map_location=self.device, weights_only=True)
```

If custom objects are needed, use `torch.serialization.add_safe_globals()` to allowlist specific classes.

---

## What's Good

- Clean module structure with clear separation of concerns
- Comprehensive constraint relaxation strategies
- Good test infrastructure foundation
- Performance meets targets (~10s optimization, <1ms per design)
- Documentation and docstrings are thorough

---

## Next Steps

Please fix the three issues above and re-request review.
