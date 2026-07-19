# AUDIT FIXES: Hyperparameter Guidance, Cost Estimator, Research Lab

## Executive Summary
Three features have critical issues preventing production deployment. All fixes are identified and documented below.

---

## FEATURE 1: HYPERPARAMETER GUIDANCE

### Issue: Missing `name` field in API response

**Problem:**
- Backend returns hyperparameters as dict with keys like "Learning Rate", "Weight Decay"
- Frontend expects each object to have a `.name` field
- Result: All cards display "undefined" as the parameter name

**Solution:**
Modify the backend to add a `name` field to each hyperparameter in the response.

**Implementation:**
1. Update `core/implementation/training_config.py` HYPERPARAMETER_EXPLANATIONS dict
2. Add `"name": "<key>"` to each hyperparameter object
3. Alternatively: Update frontend to use key instead of val.name

**Choice:** Backend fix (cleaner API contract)

---

## FEATURE 2: TRAINING COST ESTIMATOR

### Issue 1: Missing `arch_profile` wrapper

**Problem:**
Frontend expects:
```javascript
data.arch_profile.flops
data.arch_profile.params
```

But backend returns:
```javascript
params_M (at root level)
// No arch_profile object
```

**Solution:**
Wrap architecture-related metrics in `arch_profile` object in the response.

### Issue 2: Missing `gpu_profile` wrapper

**Problem:**
Frontend expects:
```javascript
data.gpu_profile.tflops
data.gpu_profile.cost_per_hour
```

But backend returns:
```javascript
// Values scattered in GPU_SPECS dict, not returned
```

**Solution:**
Create `gpu_profile` object with required fields in the response.

**Implementation:**
1. Update `core/implementation/cost_estimator.py` in `estimate_training_cost()` function
2. Modify return statement to include:
   - `arch_profile.flops` (from flops_per_image_G × batch_size)
   - `arch_profile.params` (from params_M)
   - `gpu_profile.tflops` (from gpu["tflops_fp32"])
   - `gpu_profile.cost_per_hour` (from gpu["cloud_cost_usd_hr"])

---

## FEATURE 3: RESEARCH LAB

### Issue 1: Missing `labSelectPrediction()` function

**Problem:**
Frontend code calls:
```javascript
onclick="labSelectPrediction('param','increase')"
onclick="labSelectPrediction('flops','decrease')"
```

But function is never defined, causing "is not defined" errors.

**Solution:**
Implement the function to handle prediction selection and UI updates.

### Issue 2: Missing Chart.js library

**Problem:**
`labLoadTradeoffChart()` tries to create:
```javascript
new Chart(ctx, {...})
```

But Chart.js library is not imported, causing "Chart is not defined" error.

**Solution:**
Add Chart.js CDN to HTML `<head>` section.

### Issue 3: Missing notebook management functions

**Problem:**
Three functions are called but never defined:
- `labSaveToNotebook()`
- `labClearNotebook()`
- `labUpdateNotebookBadge()`

**Solution:**
Implement all three functions to manage experiment history in localStorage.

---

## Implementation Plan

### Phase 1: Backend Fixes (Core Logic)
1. ✅ Fix Hyperparameter API response (add "name" field)
2. ✅ Fix Cost Estimator API response (add profile wrappers)

### Phase 2: Frontend Fixes (UI/UX)
1. ✅ Add Chart.js library import
2. ✅ Implement labSelectPrediction()
3. ✅ Implement notebook functions (save, clear, update, load, delete)

### Phase 3: Testing & Validation
1. ✅ Verify all features load without errors
2. ✅ Test each feature workflow
3. ✅ Verify browser console has no errors
4. ✅ Confirm API responses match frontend expectations

---

## Status: Ready for Implementation
All issues have been identified. Proceeding to code changes...
