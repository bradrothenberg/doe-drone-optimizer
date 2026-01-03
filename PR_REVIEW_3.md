## PR Review: Fix division by zero in find_nearest_feasible

**PR #3** | `week2-nsga-optimization` → `master` | +20 −2 lines

### Summary

This PR adds defensive checks to prevent division by zero in the `find_nearest_feasible` method when constraint values are zero or negative.

### ✅ What's Good

1. **Correctly identifies the bug** - Division by zero produces NaN/inf values that corrupt ranking calculations
2. **Appropriate logging** - Uses `logger.warning()` to flag this edge case
3. **Handles both zero and negative** - The `<= 0` check covers both invalid cases
4. **Non-breaking fallback** - Uses raw violation values instead of crashing

### ⚠️ Concerns / Suggestions

**1. Code duplication**

The same 7-line guard block is copy-pasted for both `'min'` and `'max'` branches. Consider refactoring:

```python
for constraint_name, violation_array in violations.items():
    target = self.constraints[constraint_name]
    if target <= 0:
        logger.warning(
            f"Constraint {constraint_name} has non-positive value ({target}). "
            "Skipping normalization to avoid division by zero."
        )
        violation_scores += violation_array
    else:
        violation_scores += violation_array / target
```

This eliminates the `if 'min'` / `elif 'max'` branching entirely since the normalization logic is identical.

**2. Missing test coverage**

No tests are added for this edge case. Consider adding a test like:

```python
def test_find_nearest_feasible_zero_constraint():
    """Test that zero constraint values don't cause division by zero"""
    handler = ConstraintHandler({'min_range_nm': 0})  # zero constraint
    designs = np.random.rand(10, 7)
    predictions = {'range_nm': np.array([100]*10), ...}

    result = handler.find_nearest_feasible(designs, predictions)

    # Should not produce NaN/inf
    assert not np.any(np.isnan(result['violation_scores']))
    assert not np.any(np.isinf(result['violation_scores']))
```

**3. Root cause not addressed**

The PR description correctly notes that `validate_constraints()` exists but isn't enforced before `ConstraintHandler` is used. This defensive fix is good, but you might also consider:

- Adding validation in `ConstraintHandler.__init__()` (with an option to skip)
- Documenting that callers should validate constraints first

**4. Inconsistent scoring behavior**

When mixing normalized and raw violations (e.g., one constraint at zero, others valid), the raw values could dominate the score unexpectedly depending on their magnitude. This is likely fine for an edge case, but worth noting.

### Verdict

**Approve with minor suggestions** ✅

The fix correctly addresses the immediate bug with appropriate defensive programming. The code duplication and missing tests are minor issues that don't block merging but would improve maintainability if addressed.
