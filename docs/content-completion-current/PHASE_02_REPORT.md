# Phase 2 Report — Broken Internal Links

Completed: 2026-08-27
Status: PASS

## Objective

Repair every explicit internal Markdown link that targeted a route rejected by
the platform's canonical registries.

## Changes made

- Repaired 26 broken links across paper articles.
- Replaced legacy architecture slugs with canonical routes:
  - `lenet` -> `lenet-5`
  - `googlenet` -> `googlenet-inception-v1`
  - `vgg16` / `vgg19` -> `vggnet`
  - generic `gpt` links -> `gpt-1`, `gpt-2`, or `gpt-3` according to context
- Replaced unavailable generic concept routes with meaningful canonical pages:
  - generic VAE links -> the Variational Inference & VAE curriculum lesson
  - generic Diffusion links -> DDPM
  - generic MoE links -> Mixtral 8x7B as a concrete sparse-MoE architecture
- Improved several labels so the destination is explicit rather than silently
  redirecting a generic name to a different concept.

## Validation

- Route-aware Markdown links scanned: 124.
- Broken internal Markdown links: 0.
- Checked route families: architectures, system design, curriculum, papers,
  Dojo problems, and static application paths.

The total explicit-link count changed from 126 to 124 because duplicate VGG16
and VGG19 links were consolidated into one canonical VGGNet family link.

## Deferred

- Curriculum prerequisite/unlock chips are data-driven React links rather than
  Markdown links; they are Phase 3.
- System-design articles need new cross-links rather than repairs; they are
  Phase 4.

