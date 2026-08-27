# Phase 5 Report — Orphan Reconciliation

Completed: 2026-08-27
Status: PASS

## Objective

Give every physical architecture, curriculum, system-design, and paper content
path an explicit canonical status without deleting reusable authoring work.

## Policy

- Register content when it adds a distinct learning outcome.
- Redirect historical slugs when a stronger canonical page covers the same
  concept.
- Copy reusable paper articles to canonical library slugs before generation.
- Retain legacy physical folders as migration evidence until a separate,
  explicitly authorized cleanup.

The complete decision map is stored in `orphan-manifest.json`.

## Changes made

### Architectures

- Registered `ae` (Autoencoder) and `vae` (Variational Autoencoder) as distinct
  foundational architectures.
- Added canonical redirects for the remaining 11 legacy architecture slugs,
  including LeNet, GoogLeNet, Swin, U-Net, VGG16/VGG19, GPT, MoE, and generic
  diffusion routes.
- Result: 227 physical paths = 216 registered + 11 explained aliases.

### Curriculum

- Registered all 12 previously unlisted foundational lessons:
  - 3 AI-agent foundations;
  - 4 LLM-engineering foundations;
  - 5 retrieval/RAG foundations.
- Added metadata for level, prerequisites, study time, purpose, and unlocks.
- Added the standard related-platform links to all 12 lessons.
- Fixed three raw `<1ms` MDX literals in the production-RAG lesson.
- Result: 94 physical lessons = 94 registered lessons.

### System design

- Added canonical redirects for all 12 legacy system-design paths.
- Legacy RAG, agent, recommendation, ChatGPT, GitHub Copilot, and Perplexity
  folders remain as migration sources while their routes resolve to one of the
  12 deep canonical system designs.
- Result: 24 physical paths = 12 registered + 12 explained aliases.

### Papers

- Added four missing landmark papers to the canonical library:
  GPT-1, GPT-2, PaLM, and Switch Transformer.
- Added canonical route mappings for all 23 short-slug paper paths.
- Copied 22 unique reusable articles and metadata files into their full
  canonical paper slugs. `stable-diffusion` and `latent-diffusion-models`
  intentionally merge into the same canonical paper.
- Result: 46 current physical paths, all registered or explicitly aliased;
  23/195 canonical papers now have articles and 172 remain for Phase 6.

## Registry changes

| Content type | Before | After | Unexplained paths |
|---|---:|---:|---:|
| Architectures | 214 registered | 216 registered | 0 |
| Curriculum | 82 registered | 94 registered | 0 |
| System design | 12 registered | 12 registered | 0 |
| Papers | 191 registered | 195 registered | 0 |

## Validation

- `node scripts/audit-content-orphans.mjs`: PASS, 0 unexplained paths and 0
  invalid alias targets across all four content families.
- `npm run audit:curriculum`: PASS, 94/94 MDX lessons compiled.
- `npm run audit:system-design`: PASS, 12/12 canonical articles compiled.
- `node scripts/audit-internal-content-links.mjs`: PASS, 539 links checked and
  0 broken.
- TypeScript (`tsc --noEmit --incremental false`): PASS.
- `node scripts/generate-content-index.mjs`: PASS, 319 indexed items across 8
  content types.

## Deferred

- Generate and validate the remaining 172 canonical paper articles and their
  metadata in Phase 6.
- Physical legacy source folders are intentionally retained. They no longer
  represent routing ambiguity because every route decision is explicit.
