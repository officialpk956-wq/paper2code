# Paper2Code — Enhancement Backlog ("What Else")

Additive ideas **beyond** the approved M1–M6 roadmap. M1 (Code Dojo) is shipped.
Grounded in what already exists: Pyodide (now proven in-browser), `TensorTracker`,
`flops_engine`, `mutator`/`diff_engine`, `codegen`, `adaptive_engine`, Chart.js, Cytoscape, Monaco.

Legend — Impact (🔥 high / ✨ med) · Effort (S / M / L).

---

## A. Code Dojo (just shipped) — go deeper

- 🔥 S **More exercises**: backward passes (ReLU/sigmoid/linear grad), Conv2d w/ stride+padding, max-pool, BatchNorm, Dropout (seeded), full Multi-Head Attention, positional encoding, focal loss, KL, IoU, NMS, beam-search step.
- 🔥 M **Boss challenges**: assemble previously-built primitives into a mini-MLP forward+backward, or one full training step of logistic regression on a toy batch.
- 🔥 M **Gradient-check exercises**: learner writes analytical gradient; harness verifies against numerical gradient (finite differences) — teaches backprop rigorously.
- ✨ M **Visual feedback**: plot the learner's activation/loss curve next to the reference curve (Chart.js) so they *see* the difference, not just numbers.
- ✨ S **Smart bug diagnosis**: pattern-match common failures (overflow → "subtract max"; NaN → "log(0), clip first") and surface the matching hint automatically.
- ✨ S **Vectorization badge**: detect Python `for` loops in a passing solution and nudge "can you vectorize this?" (educational, not blocking).
- ✨ S **Per-category progress rings + XP + streaks**; unlock harder tiers as you solve.
- ✨ M **Share / playback**: shareable link to a solved exercise; optional solution gallery.

## B. Hyperparameters — the killer interactive demo

- 🔥 L **Live in-browser training sandbox**: train a tiny MLP on a toy dataset (moons/spiral) via Pyodide while the user drags LR / batch / weight-decay / momentum sliders → watch the **live loss curve and decision boundary** update. This is the single highest-wow educational artifact we can build, and Pyodide is already proven.
- 🔥 M **Optimizer race**: animate SGD vs Momentum vs Adam vs RMSProp descending the *same* 2-D loss surface (saddle / Rosenbrock) simultaneously — reuses the Dojo optimizer implementations.
- ✨ M **"Diagnose this curve"**: show a pathological loss curve, ask which hyperparameter is wrong (links to Assessments + the fix).
- ✨ S **Recipe cards**: per-architecture known-good full config pulled from `training_config.py`, with paper citations and copy-to-clipboard.
- ✨ M **Schedule composer**: build a custom LR schedule (warmup + cosine, step, one-cycle) and see the curve live.

## C. Assessments — adaptivity & retention

- 🔥 M **Adaptive difficulty (IRT-style)**: adjust item difficulty from rolling performance instead of a fixed picker.
- 🔥 M **Mistake review deck (Anki-style)**: spaced-repetition resurfacing of previously-missed items (schema fields already planned in M3/M5).
- ✨ M **Interactive graph-repair challenges**: drag/wire nodes to fix a broken architecture instead of multiple-choice (reuses Cytoscape + `diff_engine`).
- ✨ M **Rubric-graded free text**: deterministic keyword+structure scoring for design-justification answers ("mobile, <5M params — why MobileNet?").
- ✨ S **Remediation links**: every wrong answer deep-links to the exact Dojo exercise / hyperparameter card / Explorer module that teaches it.
- ✨ S **Daily challenge + streak + leaderboard**.

## D. Research Lab — from metrics to outcomes

- 🔥 L **Micro-train the mutation**: actually train the mutated graph on a toy task in-browser to show *real* accuracy/speed, not just FLOPs — finally connects compute cost to outcomes.
- 🔥 M **Ablation runner**: define a mutation grid, auto-run all, produce a results table + Pareto chart (reuses `tradeoff_analyzer`).
- ✨ M **Export mutated architecture as runnable PyTorch** (reuse `codegen`).
- ✨ M **Natural-language what-if**: "remove all skip connections from ResNet50" → auto-applies the mutation + explains consequences (tutor + `mutator`).
- ✨ M **Gradient-flow visualization**: highlight vanishing/exploding-gradient risk visually along depth.

## E. Cost Estimator — scaling & decisions

- 🔥 M **Scaling-law curves**: interactive line charts — cost vs dataset size, time vs batch size, cost vs GPU.
- ✨ M **Distributed/multi-GPU**: data-parallel scaling efficiency and N-GPU cost.
- ✨ S **Inference cost calculator**: throughput + $/1M inferences (not just training).
- ✨ S **Budget mode**: "given $X, best GPU + config"; spot vs on-demand pricing.
- ✨ S **Carbon footprint** estimate (kgCO₂) per run.

## F. Dashboard — motivation & insight

- 🔥 M **Concept-mastery heatmap** (concepts × architectures) + **streak calendar** (GitHub-style).
- 🔥 S **"Next best action"** card from `recommendation_engine`.
- ✨ M **Skill radar over time** (animate improvement); **badges/certificates** on milestones.
- ✨ S **Goal setting** with progress-to-goal.

## G. Architecture Explorer — connect the dots

- 🔥 M **Evolution timeline**: LeNet → AlexNet → VGG → ResNet → ViT with linked diffs ("what changed and why").
- 🔥 S **Module → teach**: click any module → open the matching Dojo exercise or tutor explanation.
- ✨ L **Animated forward pass**: tensor flowing through the graph with shapes updating live (reuses `TensorTracker`).
- ✨ M **Memory/latency timeline** along the journey.

## H. Cross-cutting net-new features

- 🔥 L **Guided Learning Paths / Courses**: ordered curriculum (Beginner → Researcher) stitching Explorer + Dojo + Hyperparameters + Assessments + Lab into modules with a progress bar and a completion certificate. This is the product-level glue that turns separate tools into a *course*.
- 🔥 L **Real PyTorch verification (backend/CI, sandboxed)**: generated code is actually run (forward pass + 1 training step on tiny input) to verify shapes and that loss decreases — closes the "does the code actually work?" gap and reinforces the determinism ethos.
- 🔥 M **Gamification layer**: XP, levels, badges, streaks, daily challenge, leaderboard — applied uniformly across all features.
- ✨ M **AI Tutor deepening**: Socratic mode, step-by-step math derivations, "explain like I'm 5 / like a researcher" levels.
- ✨ M **Auth + cloud sync**: progress currently lives in localStorage + a learner_id; real accounts enable cross-device sync and the social features below.
- ✨ M **Community**: share experiments, solutions, and user-authored Dojo exercises; upvotes; a content-authoring admin UI (add exercises/items without code).
- ✨ S **Accessibility & polish**: keyboard nav, ARIA, color-blind-safe palette, mobile-responsive pass, PWA/offline.

---

## If we do only 5 next (recommended order)

1. **B — Live training sandbox** (Hyperparameters): highest wow, Pyodide already proven. 🔥L
2. **H — Guided Learning Paths**: turns the toolset into an actual course; multiplies the value of everything else. 🔥L
3. **A — Dojo depth** (backward-pass + gradient-check exercises): cheap, compounding, reinforces the new flagship. 🔥S–M
4. **G — Explorer "module → teach" + evolution timeline**: connects exploration to doing. 🔥S–M
5. **C — Assessments adaptivity + mistake-review deck**: retention engine; uses M3/M5 schema. 🔥M

## Sequencing note
M2–M6 (approved roadmap) remain the committed base. The items above are the "what else" layer:
fold A into M2's wrap-up, B replaces/ą supersizes M2's slider work, C/D extend M3/M4, and H (Learning
Paths + real code verification) are the two biggest net-new bets worth their own milestones (M7/M8).
