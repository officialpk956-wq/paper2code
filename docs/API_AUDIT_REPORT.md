# API Audit Report — Phase 11B

**Date:** 2026-06-18  
**Status:** All high-severity issues resolved

---

## 1. Input Validation

### Lab Routes — Bounds Clamping
**Files:** `src/app/api/labs/transformer/route.ts`, `cnn/route.ts`, `vit/route.ts`, `diffusion/route.ts`

All four lab POST routes use `clampInt(v, min, max, default)`:
- Rejects NaN / non-finite values (falls back to default)
- Clamps all numeric inputs to their safe ranges
- Accepts unknown keys silently (extra fields ignored)

| Lab | Parameters | Range |
|-----|-----------|-------|
| Transformer | d_model | 64–2048 |
| Transformer | num_heads | 1–32 |
| Transformer | num_layers | 1–24 |
| Transformer | seq_len | 8–2048 |
| Transformer | vocab_size | 100–50000 |
| CNN | base_channels | 8–256 |
| CNN | depth | 1–8 |
| CNN | kernel_size | 1–7 |
| CNN | image_resolution | 32–512 |
| ViT | image_size | 32–512 |
| ViT | patch_size | 4–32 |
| ViT | hidden_dim | 64–1024 |
| ViT | num_blocks | 1–24 |
| Diffusion | latent_size | 8–128 |
| Diffusion | channels | 1–4 |
| Diffusion | diffusion_steps | 100–2000 |

### Dojo Run Route
**File:** `src/app/api/dojo/run/route.ts`

Validation added:
- `code`: string, required, max 20,000 chars
- `functionName`: must match `/^[a-zA-Z_][a-zA-Z0-9_]{0,99}$/` — prevents Python code injection
- `testCases`: array, required, non-empty
- `visibleOnly`: optional boolean

### Dojo Submit Route
**File:** `src/app/api/dojo/submit/route.ts`

Validation parity with run route:
- `code`: string, required, max 20,000 chars
- `functionName`: validated Python identifier regex
- `testCases`: array, required, non-empty
- Added `typeof` checks before all uses

---

## 2. Command Injection Prevention

### Before (VULNERABLE)
```typescript
// BEFORE — user-controlled string interpolated into shell command
exec(`python backend/scripts/lab_service.py --architecture ${architecture}`, ...)
```

### After (SAFE)
```typescript
// AFTER — execFile: args passed as separate argv entries, no shell expansion
execFile('python', [scriptPath, '--architecture', architecture, '--action', 'hierarchy'], ...)
// + allowlist check before execution:
if (!VALID_ARCH_IDS.has(archId)) return NextResponse.json({ error: '...' }, { status: 404 });
```

Routes fixed:
- `src/app/api/papers/[id]/block-hierarchy/route.ts` — `exec` → `execFile` + allowlist
- `src/app/api/papers/[id]/forward-pass/route.ts` — `exec` → `execFile` + allowlist
- All 4 lab routes already used `execFile` (designed correctly from the start)

---

## 3. Architecture Allowlist (block-hierarchy / forward-pass)

```typescript
const VALID_ARCH_IDS = new Set([
  'resnet50', 'resnet', 'deep-residual-learning',
  'vit', 'vit-b16', 'an-image-is-worth-16x16-words',
  'unet', 'unet-biomedical', 'ronneberger2015',
  'transformer', 'attention-is-all-you-need', 'vaswani2017',
]);
```

Requests for unknown architecture IDs return HTTP 404 before Python is invoked.

---

## 4. Timeout Guards

All Python-spawning routes have independent timeout guards:

```typescript
const timer = setTimeout(() => reject(new Error('timeout')), TIMEOUT_MS);
execFile('python', [...], { timeout: TIMEOUT_MS }, (err, stdout) => {
  clearTimeout(timer); // clear independent timer
  ...
});
```

`TIMEOUT_MS = 30_000` (30 seconds) for lab routes.  
Using both `execFile`'s built-in timeout AND an independent `setTimeout` ensures the Promise rejects even if `execFile`'s callback fires late.

---

## 5. JSON Parse Guards

All routes wrap `JSON.parse(stdout)` in try/catch:
```typescript
try { data = JSON.parse(stdout); } catch {
  return NextResponse.json({ error: 'Unexpected response from model service' }, { status: 500 });
}
```

Prevents unhandled exceptions if Python outputs non-JSON (e.g., a traceback).

---

## 6. Response Caching

Lab routes implement in-memory LRU-style cache with 5-minute TTL:
- Cache key: parameter hash (`lab:p1:p2:p3:...`)
- Returns `X-Cache: HIT` or `X-Cache: MISS` header
- Cache is per-process (resets on server restart) — acceptable for development

---

## 7. Summary

| Category | Issues Found | Fixed |
|----------|-------------|-------|
| Command injection (exec + user input) | 2 | 2 ✅ |
| Python code injection (functionName) | 2 | 2 ✅ |
| Missing input validation (dojo/submit) | 4 | 4 ✅ |
| Missing input bounds clamping | 16 params | 16 ✅ |
| Missing JSON parse guard | 6 routes | 6 ✅ |
| Missing independent timeout | 6 routes | 6 ✅ |
