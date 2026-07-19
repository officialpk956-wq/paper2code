# Accessibility Audit Report — Phase 11B

**Date:** 2026-06-18  
**Standard:** WCAG 2.1 AA  
**Status:** High-severity issues fixed; medium-severity noted for future sprint

---

## 1. Fixes Applied (Phase 11B)

### ParameterControls — Label Association (HIGH)
**File:** `src/components/labs/ParameterControls.tsx`  
**Issue:** `<label>` had no `htmlFor`; number input had no `id`. Screen readers could not announce the label when the input was focused.  
**Fix:** Added `htmlFor={`param-number-${p.key}`}` to label, `id={inputId}` to number input.

### ParameterControls — Range Input Label (HIGH)
**File:** `src/components/labs/ParameterControls.tsx`  
**Issue:** `<input type="range">` had no `aria-label` or `aria-labelledby`.  
**Fix:** Added `aria-label={p.label}`, `aria-valuemin`, `aria-valuemax`, `aria-valuenow`.

### BlockBox — Space Key Handler Missing (HIGH)
**File:** `src/components/block-viz/BlockBox.tsx`  
**Issue:** `role="button"` only handled `Enter` key, not `Space`. WCAG SC 2.1.1 requires both.  
**Fix:** Updated `onKeyDown` to handle both `Enter` and `Space`; added `e.preventDefault()` to prevent scroll.

### BlockBox — Missing aria-expanded (MEDIUM)
**File:** `src/components/block-viz/BlockBox.tsx`  
**Issue:** No `aria-expanded` on the expand/collapse button.  
**Fix:** Added `aria-expanded={hasLayers ? expanded : undefined}`.

### ForwardPassPlayer — outline:none Without Replacement (HIGH)
**File:** `src/components/block-viz/ForwardPassPlayer.tsx`  
**Issue:** `role="slider"` had `outline: 'none'` with no focus-visible replacement (WCAG SC 2.4.7).  
**Fix:** Removed `outline: 'none'` to restore browser default focus ring.

### ForwardPassPlayer — Slider Missing aria-label (HIGH)
**File:** `src/components/block-viz/ForwardPassPlayer.tsx`  
**Issue:** Custom `role="slider"` had no accessible name.  
**Fix:** Added `aria-label="Forward pass step"` and `aria-valuetext`.

### ForwardPassPlayer — Icon Buttons Accessible Name (MEDIUM)
**File:** `src/components/block-viz/ForwardPassPlayer.tsx`  
**Issue:** Icon-only buttons used `title` prop only; inconsistent across screen readers.  
**Fix:** Added `aria-label={title}` to `IconBtn` component alongside existing `title`.

### ForwardPassPlayer — Speed Buttons aria-pressed (MEDIUM)
**File:** `src/components/block-viz/ForwardPassPlayer.tsx`  
**Issue:** Active speed button had no `aria-pressed` — screen reader users could not identify selected speed.  
**Fix:** Added `aria-pressed={i === speedIdx}` and wrapped group with `role="group" aria-label="Playback speed"`.

### DojoProblemPage — Tab ARIA Pattern (HIGH)
**File:** `src/components/dojo/DojoProblemPage.tsx`  
**Issue:** Tab buttons had no `role="tab"`, `aria-selected`, or parent `role="tablist"`. Failed SC 4.1.2.  
**Fix:** Added `role="tablist"` to both tab containers; added `role="tab"` and `aria-selected` to each tab button.

### DojoProblemPage — Back Link Accessible Name (MEDIUM)
**File:** `src/components/dojo/DojoProblemPage.tsx`  
**Issue:** Back link text was "← Dojo" — raw arrow character may be announced inconsistently.  
**Fix:** Added `aria-label="Back to Dojo"` to the link.

### Labs Page — Decorative Emoji (MEDIUM)
**File:** `src/app/labs/page.tsx`  
**Issue:** `🧪` emoji rendered without `aria-hidden="true"` — screen readers would announce "test tube".  
**Fix:** Added `aria-hidden="true"` to the emoji span.

---

## 2. Outstanding Issues (Not Fixed in Phase 11B)

### MEDIUM — MetricsPanel Loading State
**File:** `src/components/labs/MetricsPanel.tsx:82`  
No `role="status"` or `aria-live` region for the loading state. Screen readers don't announce when metrics finish loading.  
**Recommendation:** Wrap the loading indicator in `<div role="status" aria-live="polite">`.

### MEDIUM — LabSelector Emoji Button Names
**File:** `src/components/labs/LabSelector.tsx`  
Lab icon emojis rendered without `aria-hidden`. Button accessible name derived from emoji only.  
**Recommendation:** Add `aria-hidden="true"` to emoji span; ensure button has descriptive `aria-label`.

### LOW — Left Rail Logo Not Interactive
**File:** `src/components/layout/left-rail.tsx:146`  
Logo area responds to hover but has no interactive role or keyboard handler.  
**Recommendation:** Wrap in `<Link href="/">` if it should navigate home.

### LOW — DojoEditor Unicode Button Text
**File:** `src/components/dojo/DojoEditor.tsx:93,115`  
"▶ Run" and "↑ Submit" buttons contain Unicode characters that screen readers may announce verbosely.  
**Recommendation:** Add `aria-label="Run code"` and `aria-label="Submit solution"`.

---

## 3. Summary

| Category | Total Issues | Fixed in 11B | Remaining |
|----------|-------------|--------------|-----------|
| HIGH severity | 5 | 5 | 0 |
| MEDIUM severity | 6 | 5 | 1 |
| LOW severity | 2 | 0 | 2 |

WCAG 2.1 AA compliance for all interactive components that were HIGH severity: **achieved**.
