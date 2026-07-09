# paper2code — Learning + Content Roadmap

Two goals, one workflow: **learn data science by implementing it**, and **each thing you implement ships as content**. Implement an algorithm → it becomes a dojo problem. Read + build a paper → it fills a workspace. Explain a concept → it's learn/architecture content.

---

## PART A — Learn DS by implementing (a build-driven curriculum)

Every item below = one dojo problem you author in `src/data/problems.ts` (same shape as the existing ones). Doing it teaches you the concept AND grows the dojo.

**Already built (~10 — don't duplicate):** numpy array creation, numpy dot product, sigmoid, relu, mse, softmax, normalize, cross-entropy, gradient-descent step, scaled attention.

### Phase 1 — Stats & Probability (foundations, easiest to implement)
1. Mean / variance / std from scratch
2. Covariance & correlation (Pearson)
3. Z-score standardization
4. Normal distribution PDF
5. Bayes' theorem (discrete)
6. Bootstrap sampling + sample mean CI
7. Two-sample t-test (statistic + interpret)
8. Maximum likelihood estimate for a Gaussian
9. Entropy & KL divergence
10. Sigmoid/softmax numerical stability (log-sum-exp)

### Phase 2 — Classic ML from scratch
11. Train/test split (deterministic seed)
12. Linear regression — normal equation
13. Linear regression — gradient descent
14. Logistic regression — gradient descent
15. Classification metrics: accuracy / precision / recall / F1
16. ROC curve + AUC
17. k-fold cross-validation
18. k-Nearest Neighbours classifier
19. k-means clustering (Lloyd's algorithm)
20. PCA via eigendecomposition
21. Decision tree — best Gini split
22. Gaussian Naive Bayes

### Phase 3 — Deep Learning from scratch
23. Backprop for a single neuron (chain rule)
24. 2-layer MLP — forward + backward
25. Batch normalization (forward pass)
26. Layer normalization
27. Dropout (train vs eval behaviour)
28. Conv2D (naive nested loops)
29. Max pooling / average pooling
30. SGD with momentum
31. Adam optimizer (one step)
32. Embedding lookup + gradient

### Phase 4 — Paper implementations (deepest learning; fills flagship workspaces)
Interleave these from Phase 2 onward for motivation.
33. Scaled dot-product attention — *Attention Is All You Need*
34. Multi-head attention (split/concat) — *Attention Is All You Need*
35. Sinusoidal positional encoding — *Attention Is All You Need*
36. Residual block — *Deep Residual Learning (ResNet)*
37. Transformer encoder block — *BERT*
38. Patch embedding — *ViT*
39. Low-rank adapter (LoRA layer) — *LoRA*
40. Tiled/blocked attention (simplified) — *FlashAttention*

**Result:** dojo grows ~13 → ~45+, you cover a full DS curriculum, and Phase 4 fills the 6 flagship paper workspaces.

### Suggested order
Phase 1 (fast wins, build momentum) → start Phase 2, and drop in Phase 4 #33 (attention) early because it's exciting → finish Phase 2 → Phase 3 → remaining Phase 4. Aim for ~3–5 problems/week; the whole curriculum is ~2–3 months at a relaxed pace.

---

## PART B — Complete the missing content

### Current inventory (audited on disk)
| Type | Real content | In library | Gap |
|---|---|---|---|
| Papers | 24 | 200+ | ~176 |
| Architectures | 31 | 207 | ~176 |
| System design | 12 | 12 | ~complete |
| Dojo problems | ~13 | — | grows via Part A |
| Math / Interview / Roadmaps | 2 / 2 / 1 | — | thin |
| Implementations | 9 | — | — |

### Strategy: finish the spine deeply, not the tail shallowly
A deep 30-item product beats a hollow 400-item one. Priority order:
1. **6 flagship papers — 100% complete** (summary, key concepts, architecture diagram, runnable code, related links). ~half done already.
2. **Top ~20 architectures** (the ones actually searched): Transformer, ResNet, BERT, GPT, ViT, U-Net, CNN, RNN/LSTM, GAN, Diffusion/Stable Diffusion, CLIP, LLaMA, AlexNet, VGG, BatchNorm, Attention, Seq2Seq, Word2Vec, Autoencoder, MoE.
3. **12 learning domains — one strong core topic each.**
4. **Dojo → ~40 problems** (free, from Part A).
5. Everything else: keep as honest "coming soon" placeholders.

### Per-item content workflow (template already exists)
Each item is `meta.json` + `content.mdx` under `src/content/<type>/<slug>/`:
- **meta.json** — title, slug, summary, tags, difficulty, relationships (related papers/archs/problems).
- **content.mdx** — sections: *What it is* → *Key idea / intuition* → *Architecture (diagram)* → *The math* (KaTeX) → *Reference code* → *Why it matters / built upon by*.
- Rebuild index: `node scripts/generate-content-index.mjs` (runs automatically on build).
- Cross-link with the crosslinks helper so papers ↔ architectures ↔ dojo problems connect.

### Production options (decide later)
- **AI-assisted, you edit** — draft each with Groq/Gemini (already wired), then edit + verify. Fast, and editing is where you learn. Best for the ~20 architectures + long tail.
- **Hand-write** — best for the 6 flagship papers where quality matters most.
- Realistic milestones: Month 1 = 6 flagship papers done + 5 architectures. Month 2 = 15 more architectures + 12 domain core topics. Ongoing = long tail as time allows.

---

## How the two parts connect
- Part A Phase 4 (paper implementations) **is** Part B priority 1 (flagship paper code sections).
- Every Part A problem is a real dojo item → Part B dojo target hit for free.
- Writing the "intuition / math" for a concept you just implemented = Part B learn content, written while it's fresh.

Do Part A and a big chunk of Part B happens as a side effect.
