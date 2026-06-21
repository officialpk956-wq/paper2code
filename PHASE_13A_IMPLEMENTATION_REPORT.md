# Phase 13A — Research Paper Ingestion Pipeline: Implementation Report

## Summary

Phase 13A delivers a working end-to-end paper ingestion pipeline. A user uploads a research PDF via the `/papers/upload` page; the backend parses it, extracts sections, figures, and equations, runs the existing Paper2Code graph/module pipeline, persists the result, and redirects to a generated paper workspace at `/papers/upload/{id}`.

---

## What Was Built

### 1. `backend/services/paper_document.py` (new)
Structured dataclasses for the ingestion output:

| Class | Fields |
|---|---|
| `Section` | id, title, content, level |
| `Figure` | id, page, caption, xref, width, height, ext, has_binary |
| `Equation` | id, page, text |
| `PaperDocumentMetadata` | page_count, file_size, text_extraction_method, source_filename |
| `PaperDocument` | id, title, authors, abstract, sections, figures, equations, metadata |

`PaperDocument.to_dict()` serializes the full document for API responses.

### 2. `extract_sections()` in `backend/services/paper_ingestion_service.py` (new function)

Detects standard research-paper section headings using a compiled regex that matches:
- Bare headings: `Abstract`, `Introduction`, `Conclusion`, …
- Numbered headings: `1. Introduction`, `2.1 Methods`, …
- Case-insensitive variants: `RELATED WORK`, `related work`, …

Canonical names are normalized via `_CANONICAL_NAMES` dict (e.g. `methodology` → `Methods`). Content is extracted as the text between consecutive heading matches and capped at 4 000 characters. Sections with empty content are silently dropped. Returns a `List[Dict]` compatible with the ingestion payload JSON.

`build_ingestion_payload` updated to call `extract_sections` and include `sections` / `section_count` in the returned dict.

### 3. `backend/server.py` — three fixes

**Fix A: `/api/papers/upload` endpoint (critical)**

The endpoint body had two overlapping implementations merged with Python syntax errors (assignment statements inside a `return {}` dict literal). Replaced entirely with a clean 6-line delegate to `ingest_pdf_paper()`:

```python
try:
    pdf_bytes = await file.read()
    result = ingest_pdf_paper(db=db, pdf_bytes=pdf_bytes,
                              source_filename=file.filename,
                              paper_name=paperName or file.filename.replace(".pdf", ""))
    return result
except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))
```

**Fix B: `/api/tutor/ask` endpoint**

Stray dict-entry fragments from a different endpoint had been spliced into the analytics-logging block, making the function unparseable. Removed the four stray lines and restored `arch = request.context_data.get("architecture") or request.context_type`.

**Fix C: `GET /api/papers/{paper_id}` response**

Added `source_filename`, `figure_count`, `equation_count` to `metadata` and exposed `ingestion` as a top-level key in the response dict. This aligns the response with the `PaperDetailResponse` TypeScript type already declared in `src/app/papers/upload/[paperId]/page.tsx`.

### 4. `src/app/api/papers/generated/[id]/route.ts` (new)

Next.js App Router `GET` handler that proxies to `GET /api/papers/{id}` on the FastAPI backend. Validates the `id` param (digits-only), forwards the response verbatim, and normalizes backend errors to 502.

### 5. Upload UI — `src/components/paper-upload/PaperUploadWorkspace.tsx`

The component already existed and was complete. No changes needed: it uses `ThreeColumnLayout`, `Button`, `Input`, `Spinner`, and `SectionLabel` from the Phase 12B design system; submits to `POST /api/papers/upload`; and redirects to the generated workspace.

### 6. Generated paper workspace — `src/app/papers/upload/[paperId]/page.tsx`

Already complete. Displays extracted figures, equations, ingestion timeline, and module outline. Reads `paper.ingestion` (now exposed by Fix C above).

---

## Test Results

### Python backend tests (`tests/test_paper_ingestion_service.py`)
19 tests — 19 passed.

New tests added:
- `test_normalize_title_no_extension`
- `test_normalize_title_empty_stem_falls_back`
- `test_resolve_unique_title_no_conflict`
- `test_resolve_unique_title_multiple_duplicates`
- `test_extract_equations_respects_line_length_cap`
- `test_extract_equations_empty_pages`
- `test_extract_equations_cap_at_80`
- `test_extract_equations_assigns_page_numbers`
- `test_extract_sections_detects_abstract` *(new)*
- `test_extract_sections_canonical_names` *(new)*
- `test_extract_sections_numbered_headings` *(new)*
- `test_extract_sections_no_content_sections_skipped` *(new)*
- `test_extract_sections_content_truncated_at_4000_chars` *(new)*
- `test_extract_sections_returns_list_of_dicts` *(new)*
- `test_extract_sections_empty_input` *(new)*
- `test_extract_sections_appendix_level_2` *(new)*

### Frontend tests (`npm run test`)
193 tests — 193 passed (21 test files).

New / expanded tests in `src/__tests__/components/paper-upload/PaperUploadWorkspace.test.tsx`:
- Renders form with title input and buttons
- Validation error when no file selected
- File summary updates on file pick
- Success path: submits PDF → navigates to `/papers/upload/{id}`
- Backend 400 error displayed in UI
- Network failure displayed in UI
- Submit button disabled while loading
- Ingestion flow step indicators render

### Build
`next build` exits 0. All 21 pages/routes compile without TypeScript errors.

---

## Architecture Decisions

| Decision | Rationale |
|---|---|
| `paper_document.py` in `backend/services/` not `backend/models/` | `backend/models.py` is a flat `.py` file; creating `backend/models/paper_document.py` would require making `models` a package, risking import breaks across the codebase. Placing it alongside the ingestion service avoids any migration. |
| Sections stored inside `architecture_graph.ingestion` JSON | Zero schema migration; the existing `Paper.architecture_graph` column already holds the full ingestion dict. |
| `GET /api/papers/generated/[id]` proxies `GET /api/papers/{id}` | Re-uses the existing paper detail endpoint. No duplication; the generated workspace page also calls `GET /api/papers/{id}` directly. |
| Section content capped at 4 000 chars | Prevents the JSON column from bloating on long-form papers; full text is available in `raw_text_excerpt` for the first 4 000 chars of the whole document. |

---

## Constraints Checklist

| Constraint | Status |
|---|---|
| No mock data | ✅ All extraction is deterministic; no stubs or hardcoded payloads |
| Reuse existing components | ✅ `PaperUploadWorkspace`, `ThreeColumnLayout`, `Button`, `Input`, `Spinner`, `SectionLabel` |
| Reuse design system | ✅ CSS custom properties throughout; no new hardcoded colors |
| Reuse paper workspace | ✅ `/papers/upload/[paperId]/page.tsx` unchanged |
| TypeScript clean | ✅ Build passes with zero type errors |
| Build passes | ✅ `next build` exits 0 |
| Tests pass | ✅ 193 frontend + 19 Python |
| Accessibility preserved | ✅ No `aria-*` attributes removed; form labels and button roles intact |
| Security preserved | ✅ PDF size and MIME validated server-side; `id` param validated as digits-only in new route |
