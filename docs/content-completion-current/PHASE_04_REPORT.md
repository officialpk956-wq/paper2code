# Phase 4 Report — System-Design Cross-Linking

Completed: 2026-08-27
Status: PASS

## Objective

Connect every canonical system-design article to relevant curriculum lessons,
architectures, papers, and sibling system designs without introducing dead
routes, and make those links survive future content regeneration.

## Changes made

- Added a curated `relatedLinks` map to
  `scripts/generate-system-design-content.mjs`.
- Added a generated `Related Platform Content` section to all 12 canonical
  system-design articles.
- Each article now contains four context-specific internal links spanning the
  most useful adjacent content types for that system.
- Added a generator guard that refuses to render an article with fewer than
  four related links.
- Extended `scripts/audit-system-design-content.mjs` so the system-design audit
  requires at least four internal links in every canonical article.
- Added `scripts/audit-internal-content-links.mjs`, a reusable registry-aware
  audit for architecture, curriculum, system-design, paper, and Dojo routes.
- The exhaustive audit exposed 48 legacy links outside the original Phase 2
  scan. Added `scripts/repair-legacy-content-links.mjs` and migrated them from
  retired `/math`, `/problems`, and Transformer subroutes to current curriculum,
  Dojo, and architecture routes.

## Article coverage

| Metric | Before | After |
|---|---:|---:|
| Canonical system-design articles | 12 | 12 |
| Articles with required related links | 0/12 | 12/12 |
| Curated links added | 0 | 48 |
| Platform internal Markdown links | 370 | 418 |
| Broken internal Markdown links after exhaustive audit | 48 | 0 |

The 48 newly discovered broken links were pre-existing links in architecture
and short-slug paper content. They were repaired in this phase because the new
checker covered the entire content tree rather than only the previously known
paper-link set.

## Validation

- `node scripts/generate-system-design-content.mjs`: PASS, 12 articles and
  metadata files regenerated.
- `npm run audit:system-design`: PASS, 12/12 MDX articles compiled and all link
  requirements satisfied.
- `node scripts/audit-internal-content-links.mjs`: PASS, 418 links checked and
  0 broken.
- TypeScript (`tsc --noEmit --incremental false`): PASS.

## Deferred

- Legacy physical content directories and their canonical route decisions are
  handled in Phase 5.
- Paper short-slug routes remain valid physical workspaces until their exact
  canonical mappings are recorded in Phase 5 and canonical articles are
  completed in Phase 6.
