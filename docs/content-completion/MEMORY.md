# Paper2Code Content Completion Memory

Last updated: 2026-08-27
Status: ACTIVE
Current phase: Phase 6 — Paper library completion

## Objective

Complete all seven content-repair phases without losing scope across context
compaction. A phase is complete only after its implementation, audit, and
phase report are finished.

## Rules

1. Read this file before resuming work after context loss.
2. Update `Current phase`, counts, decisions, validation, and `Next action`
   before advancing to another phase.
3. Write `PHASE_0N_REPORT.md` after each phase.
4. Preserve user-authored content unless a canonical migration has been
   verified. Prefer aliases/copies over destructive deletion during repair.
5. Never mark the goal complete while a registered item lacks routed content,
   metadata validation fails, or an internal routed link is broken.

## Baseline audit

- Curriculum: 82/82 canonical topics have MDX.
- System design: 12/12 canonical systems have MDX and meta.json.
- Architectures: 214 registered; 201 registered slugs have matching folders;
  13 registered slugs lack matching folders; all 214 physical folders have
  content.mdx; 122 physical folders lack meta.json.
- Papers: 191 registered; 1 has exact canonical MDX; 5 more have workspace
  fallbacks; 23 short-slug article folders may be reusable; 185 library items
  are unavailable.
- Explicit internal Markdown links: 126 total, 26 broken.
- Curriculum prerequisite/unlock chips: 464 total, 21 currently resolve.
- Architecture lineage tokens: 475 total, 357 resolve by exact name.
- Existing valid metadata entries: 162; broken relationship targets: 0.
- Generated content index is stale (2026-08-15), containing old system-design
  slugs. Normal prebuild fails on 122 missing architecture meta.json files.

## Seven phases

### Phase 1 — Architecture foundation

Scope:
- Generate 122 missing architecture meta.json files.
- Reconcile five known canonical-directory aliases:
  - lenet-5 <- lenet
  - googlenet-inception-v1 <- googlenet
  - swin-transformer <- swin
  - u-net <- unet
  - deeplab-v3-2 <- deeplabv3plus
- Create canonical content for eight genuinely missing registered slugs:
  gemini-1-0, gemini-1-5, svd, dien, dcn, dcn-v2, autoint, chebnet.
- Preserve old directories until Phase 5 orphan reconciliation.

Acceptance:
- 214/214 registered architecture slugs have content.mdx and meta.json.
- Architecture metadata validates with no relationship errors.
- Phase report exists.

### Phase 2 — Broken links

Scope: repair the 26 confirmed paper-to-architecture Markdown links and rerun
a route-aware Markdown link audit.

Acceptance: zero broken explicit internal Markdown links; phase report exists.

### Phase 3 — Curriculum navigation

Scope: implement cross-category resolution for prerequisite/unlock chips and
add useful links inside canonical curriculum lessons.

Acceptance: resolvable architecture/system/paper/math targets become clickable;
all generated links validate; phase report exists.

### Phase 4 — System-design linking

Scope: add canonical curriculum, architecture, paper, and sibling-system links
to all 12 system-design articles.

Acceptance: every canonical article has relevant validated internal links;
phase report exists.

### Phase 5 — Orphan reconciliation

Scope: classify 13 architecture, 12 curriculum, 12 system-design, and 23 paper
orphan paths as migrate/register/merge/retain-with-alias/archive. Avoid deletion
without a verified canonical replacement.

Acceptance: no unexplained orphan remains; route/alias decisions documented;
phase report exists.

### Phase 6 — Paper library completion

Scope: map reusable short-slug articles to canonical papers, then create every
remaining registered canonical paper article and meta.json in reviewable
batches. Use the content-research writing skill for article generation.

Acceptance: 195/195 registered papers have a canonical article or deliberately
documented richer workspace path; all canonical paper metadata validates;
phase report exists.

### Phase 7 — Permanent audit and final validation

Scope: add `npm run audit:content`, regenerate content-index.json, run MDX,
KaTeX, relationship, link, TypeScript, and production-build validation.

Acceptance: all audits and normal production build pass; final report exists.

## Completed work

- Curriculum generation and rendering completed before this seven-phase loop.
- System-design generation and metadata completed before this loop.
- `npm run audit:curriculum` passes.
- `npm run audit:system-design` passes.
- Phase 1 completed; report: `PHASE_01_REPORT.md`.
- Architecture canonical coverage is 214/214 for both content and metadata.
- Five alias articles were copied to canonical routes; eight missing articles
  were created; 227/227 physical architecture MDX files compile.
- Content-index generation now passes with 297 indexed items across 8 types.
- Phase 2 completed; report: `PHASE_02_REPORT.md`.
- All 26 confirmed broken paper-to-architecture links were repaired.
- Route-aware explicit Markdown link audit: 124 checked, 0 broken.
- Phase 3 completed; report: `PHASE_03_REPORT.md`.
- Curriculum chip resolution improved from 21/464 to 133/464 using exact
  cross-category matches and 74 curated aliases.
- All 82 canonical curriculum lessons now contain validated related-content
  links. Platform Markdown-link audit: 370 checked, 0 broken.
- Phase 4 completed; report: `PHASE_04_REPORT.md`.
- All 12 canonical system-design articles now contain four curated links to
  related systems, curriculum lessons, architectures, or papers.
- Added a permanent registry-aware internal Markdown-link checker.
- Repaired 48 additional legacy links found in architecture and short-paper
  content. Platform Markdown-link audit: 418 checked, 0 broken.
- Phase 5 completed; report: `PHASE_05_REPORT.md`.
- The full classification is recorded in `orphan-manifest.json`.
- Registered content expanded to 216 architectures, 94 curriculum lessons,
  12 canonical system designs, and 195 papers.
- All physical paths are now canonical or declared aliases: architectures
  227 = 216 + 11 aliases; curriculum 94 = 94 + 0; system design 24 = 12 + 12;
  papers 46 physical paths with 23 explicit short-slug aliases.
- Four landmark papers absent from the registry were added: GPT-1, GPT-2,
  PaLM, and Switch Transformer.
- Reused short-slug paper content was copied to 22 unique canonical targets.
  Canonical paper coverage is now 23/195; 172 remain for Phase 6.

## Decisions pending

- Paper-generation batches must preserve the 23 richer canonical articles and
  generate only the remaining 172 entries.
- Phase 6 metadata should use canonical registry fields and validated internal
  relationships, not inferred short-slug routes.

## Current validation

- Phase 1: PASS.
- `node scripts/generate-content-index.mjs`: PASS.
- Architecture MDX/GFM/KaTeX compile: 227/227 PASS.
- Phase 2 internal Markdown link audit: PASS (0 broken).
- Phase 3 curriculum audit and TypeScript: PASS.
- Phase 4 system-design audit and TypeScript: PASS.
- Repository-wide explicit Markdown links: 418 checked, 0 broken.
- Phase 5 orphan audit: PASS, 0 unexplained paths.
- Expanded curriculum audit: 94/94 PASS.
- Current repository-wide explicit Markdown links: 539 checked, 0 broken.
- Canonical paper coverage: 23/195; 172 missing.

## Next action

Read the content-research writing skill, design a canonical paper article and
metadata generator that preserves the 23 richer articles, generate the 172
missing papers in reviewable batches, and validate every batch before writing
`PHASE_06_REPORT.md`.
