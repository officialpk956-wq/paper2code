# Paper2Code Architecture Coverage Matrix
**Generated:** 2026-06-21 | **Phase:** Content Validation Audit

Evaluates all 21 architectures specified in the audit scope. Scores are based on content that **currently exists** in the codebase.

**Dimensions:**
1. **Overview** — Intro/overview section with purpose and context
2. **History** — Historical background, year, researchers, predecessor context
3. **Diagram** — Interactive SVG diagram in Architecture Explorer (`architecture-catalog.ts`)
4. **Math** — Key mathematical equations and formulas
5. **Implementation** — Dedicated implementation MDX file with code milestones
6. **Training** — Training procedure, loss functions, hyperparameters documented
7. **Evolution** — Follow-on architectures and how it was improved upon
8. **Strengths** — Advantages and where it excels
9. **Weaknesses** — Limitations and failure cases
10. **Related Papers** — Paper MDX link or reference

**Score:** Y = present | P = partially present | N = absent

**Rank:** Complete (9–10) | Mostly Complete (7–8) | Partial (4–6) | Missing (0–3)

---

## TRANSFORMER FAMILY

### Transformer
**Architecture MDX:** `src/content/architectures/transformer/content.mdx` (175 lines, full 15-section)
**Implementation MDX:** `src/content/implementations/attention-is-all-you-need/content.mdx` (510 lines)
**Catalog:** COMPLETE (has interactive diagram, keyFacts, animatedSVG)

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | **10/10 — Complete** |

---

### BERT
**Architecture MDX:** `src/content/architectures/bert/content.mdx` (165 lines, full 15-section)
**Implementation MDX:** `src/content/implementations/bert/content.mdx` (396 lines)
**Catalog:** COMPLETE (has interactive diagram)

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | **10/10 — Complete** |

---

### GPT
**Architecture MDX:** `src/content/architectures/gpt/content.mdx` (170 lines, full 15-section)
**Implementation MDX:** `src/content/implementations/gpt/content.mdx` (300 lines)
**Catalog:** COMPLETE (has interactive diagram)

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | **10/10 — Complete** |

---

### LLaMA
**Architecture MDX:** `src/content/architectures/llama/content.mdx` (167 lines, full 15-section)
**Implementation MDX:** `src/content/implementations/llama/content.mdx` (89 lines, stub)
**Catalog:** `status: "coming-soon"` — BUG (MDX exists and is substantive)

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| Y | Y | N | Y | P | Y | Y | Y | Y | Y | **8/10 — Mostly Complete** |

**Notes:** Catalog bug hides LLaMA in Architecture Explorer. Implementation MDX is 89-line stub. Fix: update catalog `status` to `"complete"`.

---

## VISION TRANSFORMERS

### ViT (Vision Transformer)
**Architecture MDX:** `src/content/architectures/vit/content.mdx` (173 lines, full 15-section)
**Implementation MDX:** `src/content/implementations/vision-transformer/content.mdx` (122 lines, partial — no H2 sections)
**Catalog:** COMPLETE (has interactive diagram)

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| Y | Y | Y | Y | P | Y | Y | Y | Y | Y | **9/10 — Mostly Complete** |

**Note:** Implementation MDX is 122 lines without structured H2 sections — score as partial.

---

### CLIP
**Architecture MDX:** `src/content/architectures/clip/content.mdx` (35 lines, SCAFFOLD — all bodies empty)
**Implementation MDX:** `src/content/implementations/clip/content.mdx` (87 lines, has PyTorch code milestones)
**Catalog:** No complete catalog entry

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| N | N | N | N | P | N | N | N | N | Y | **2/10 — Partial** |

**Notes:** Architecture MDX is a scaffold placeholder — all 17 section headers present but every body is empty. Implementation MDX has real PyTorch (CLIP class, InfoNCE loss, training loop) but no H2 structure. Paper MDX is complete. This is the largest gap in the Vision Transformer family.

---

### Swin Transformer
**Architecture MDX:** `src/content/architectures/swin/content.mdx` (35 lines, SCAFFOLD — all bodies empty)
**Implementation MDX:** None
**Catalog:** No entry (or stub)

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| N | N | N | N | N | N | N | N | N | N | **0/10 — Missing** |

**Notes:** 35-line scaffold with empty section bodies. No implementation MDX. No paper in the 19-paper library. Section headers exist as a placeholder framework only.

---

### DINO
**Architecture MDX:** `src/content/architectures/dino/content.mdx` (35 lines, SCAFFOLD — all bodies empty)
**Implementation MDX:** None
**Catalog:** No entry

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| N | N | N | N | N | N | N | N | N | N | **0/10 — Missing** |

---

## CNN FAMILY

### LeNet
**Architecture MDX:** `src/content/architectures/lenet/content.mdx` (~138 lines, full 15-section)
**Implementation MDX:** None
**Catalog:** No complete catalog entry

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| Y | Y | N | Y | N | Y | Y | Y | Y | N | **7/10 — Mostly Complete** |

**Note:** Pioneering 1989 architecture — MDX covers history and context well. No dedicated paper in the 19-paper library (but paper predates most ML paper libraries).

---

### AlexNet
**Architecture MDX:** `src/content/architectures/alexnet/content.mdx` (~150 lines, full 15-section)
**Implementation MDX:** None
**Catalog:** No complete catalog entry

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| Y | Y | N | Y | N | Y | Y | Y | Y | Y | **8/10 — Mostly Complete** |

**Sources:** Paper: `src/content/papers/alexnet/content.mdx`

---

### VGG16
**Architecture MDX:** `src/content/architectures/vgg16/content.mdx` (~145 lines, full 15-section)
**Implementation MDX:** None
**Catalog:** Catalog has `vgg` stub (different slug — mismatch)

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| Y | Y | N | Y | N | Y | Y | Y | Y | Y | **8/10 — Mostly Complete** |

**Note:** Catalog slug collision — catalog uses `vgg` stub but MDX is at `vgg16`. Paper: `src/content/papers/vgg/content.mdx`

---

### VGG19
**Architecture MDX:** `src/content/architectures/vgg19/content.mdx` (~145 lines, full 15-section)
**Implementation MDX:** None
**Catalog:** Same slug mismatch as VGG16

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| Y | Y | N | Y | N | Y | Y | Y | Y | Y | **8/10 — Mostly Complete** |

---

### GoogLeNet (Inception)
**Architecture MDX:** `src/content/architectures/googlenet/content.mdx` (~148 lines, full 15-section)
**Implementation MDX:** None
**Catalog:** No complete catalog entry

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| Y | Y | N | Y | N | Y | Y | Y | Y | N | **7/10 — Mostly Complete** |

**Note:** No dedicated GoogLeNet/Inception paper in the 19-paper library.

---

### ResNet
**Architecture MDX:** `src/content/architectures/resnet/content.mdx` (177 lines, full 15-section)
**Implementation MDX:** `src/content/implementations/resnet/content.mdx` (279 lines, full)
**Catalog:** COMPLETE (has interactive diagram)

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | **10/10 — Complete** |

---

### DenseNet
**Architecture MDX:** `src/content/architectures/densenet/content.mdx` (~145 lines, full 15-section)
**Implementation MDX:** None
**Catalog:** No complete catalog entry

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| Y | Y | N | Y | N | Y | Y | Y | Y | N | **7/10 — Mostly Complete** |

---

### EfficientNet
**Architecture MDX:** `src/content/architectures/efficientnet/content.mdx` (~150 lines, full 15-section)
**Implementation MDX:** None
**Catalog:** No complete catalog entry

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| Y | Y | N | Y | N | Y | Y | Y | Y | N | **7/10 — Mostly Complete** |

**Note:** Compound scaling coefficient math (α, β, γ) is distinctive and should be in Math section of MDX.

---

## GENERATIVE MODELS

### GAN (Generative Adversarial Network)
**Architecture MDX:** `src/content/architectures/gan/content.mdx` (~160 lines, full 15-section)
**Implementation MDX:** `src/content/implementations/gan/content.mdx` (274 lines, full)
**Catalog:** No complete catalog entry (MDX is full but diagram not wired up)

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| Y | Y | N | Y | Y | Y | Y | Y | Y | Y | **9/10 — Mostly Complete** |

---

### VAE (Variational Autoencoder)
**Architecture MDX:** `src/content/architectures/vae/content.mdx` (~148 lines, full 15-section)
**Implementation MDX:** None
**Catalog:** No complete catalog entry

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| Y | Y | N | Y | N | Y | Y | Y | Y | N | **7/10 — Mostly Complete** |

**Note:** ELBO derivation is complex and valuable. No dedicated VAE paper in the 19-paper library.

---

### Stable Diffusion
**Architecture MDX:** `src/content/architectures/stable-diffusion/content.mdx` (~165 lines, full 15-section)
**Implementation MDX:** `src/content/implementations/stable-diffusion/content.mdx` (433 lines, full)
**Catalog:** No complete catalog entry

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| Y | Y | N | Y | Y | Y | Y | Y | Y | Y | **9/10 — Mostly Complete** |

---

## U-NET FAMILY

### U-Net
**Architecture MDX:** `src/content/architectures/unet/content.mdx` (35 lines, SCAFFOLD — all bodies empty)
**Implementation MDX:** None
**Catalog:** No complete catalog entry

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| N | N | N | N | N | N | N | N | N | N | **0/10 — Missing** |

**Notes:** Critical gap — U-Net is the backbone of Stable Diffusion's denoising process. Architecture MDX is scaffold only. No implementation MDX. No paper in library.

---

## MIXTURE OF EXPERTS

### MoE (Mixture of Experts)
**Architecture MDX:** `src/content/architectures/moe/content.mdx` (~155 lines, full 15-section)
**Implementation MDX:** None
**Catalog:** No complete catalog entry

| Overview | History | Diagram | Math | Implementation | Training | Evolution | Strengths | Weaknesses | Papers | **Score** |
|---------|---------|---------|------|---------------|---------|----------|----------|-----------|--------|-----------|
| Y | Y | N | Y | N | Y | Y | Y | Y | Y | **8/10 — Mostly Complete** |

---

## SUMMARY RANKINGS

### Complete (9–10/10)
| Architecture | Score | Missing |
|-------------|-------|---------|
| Transformer | 10/10 | — |
| BERT | 10/10 | — |
| GPT | 10/10 | — |
| ResNet | 10/10 | — |
| ViT | 9/10 | Implementation MDX is thin |

### Mostly Complete (7–8/10)
| Architecture | Score | Key Gaps |
|-------------|-------|---------|
| LLaMA | 8/10 | No catalog diagram (bug); implementation MDX is stub |
| AlexNet | 8/10 | No implementation MDX; no catalog diagram |
| VGG16 | 8/10 | No implementation MDX; catalog slug mismatch |
| VGG19 | 8/10 | No implementation MDX; catalog slug mismatch |
| MoE | 8/10 | No implementation MDX; no catalog diagram |
| GAN | 9/10 | No catalog diagram |
| Stable Diffusion | 9/10 | No catalog diagram |
| GoogLeNet | 7/10 | No diagram, no implementation, no dedicated paper |
| LeNet | 7/10 | No diagram, no implementation, no dedicated paper |
| DenseNet | 7/10 | No diagram, no implementation, no paper |
| EfficientNet | 7/10 | No diagram, no implementation, no paper |
| VAE | 7/10 | No diagram, no implementation, no paper |

### Partial (4–6/10)
| Architecture | Score | Key Gaps |
|-------------|-------|---------|
| CLIP | 2/10 | Architecture MDX is empty scaffold; implementation has code but no structure |

### Missing (0–3/10)
| Architecture | Score | Key Gaps |
|-------------|-------|---------|
| U-Net | 0/10 | 35-line scaffold with all bodies empty; no implementation; no paper |
| Swin Transformer | 0/10 | 35-line scaffold with all bodies empty; no implementation; no paper |
| DINO | 0/10 | 35-line scaffold with all bodies empty; no implementation; no paper |

---

## DIMENSION-LEVEL ANALYSIS

| Dimension | Complete | Notes |
|-----------|----------|-------|
| Overview | 17/21 (81%) | All full-content MDX has overview sections; 4 scaffold/missing lack them |
| History | 17/21 (81%) | Same — full MDX covers history well |
| Diagram | 5/21 (24%) | Only Transformer, BERT, GPT, ResNet, ViT have Architecture Explorer interactive diagrams |
| Math | 17/21 (81%) | Full MDX has math sections; scaffold/missing lack them |
| Implementation | 6/21 (29%) | Full: Transformer, BERT, GPT, ResNet; Partial: ViT, CLIP |
| Training | 17/21 (81%) | Full MDX covers training procedures |
| Evolution | 17/21 (81%) | Full MDX documents follow-on work |
| Strengths | 17/21 (81%) | Full MDX has strengths/weaknesses analysis |
| Weaknesses | 17/21 (81%) | Same |
| Papers | 13/21 (62%) | 8 architectures lack a corresponding paper in the 19-paper library |

**The single most underrepresented dimension is interactive Diagrams (24%).** The Architecture Explorer shows rich animated SVGs for 5 architectures; the other 16 have no diagram data wired up.
