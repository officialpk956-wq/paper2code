# Architecture Content Harness — All Batches

## HOW TO USE
Copy **MASTER PREAMBLE** + one **BATCH** block into Antigravity per session.
Each batch creates 4–6 MDX files. After Antigravity finishes, commit the new files.

---

## MASTER PREAMBLE (paste at the top of every prompt)

You are writing technical MDX content for paper2code, an educational ML platform.
Each file goes at the exact path listed. Use GitHub-flavored Markdown with KaTeX math (`$inline$`, `$$block$$`).

**Required section structure (copy exactly):**

```
# [Architecture Name]

## 1. Overview
## 2. Historical Context
## 3. Problem It Solves
## 4. Architecture Diagram Data
## 5. Layer-by-Layer Breakdown
## 6. Tensor Flow Walkthrough
## 7. Mathematical Foundations
## 8. Training Procedure
## 9. PyTorch Implementation
## 10. Strengths
## 11. Weaknesses
## 12. Research Evolution
## 13. Interview Questions
## 14. Related Papers
## 15. Further Reading
```

**Content rules:**
- Section 4 must contain a detailed text/ASCII diagram of the full architecture with labeled components
- Section 6 must show tensor shapes at every layer: `Input: (B, C, H, W) → Conv1: (B, 64, H/2, W/2)` etc.
- Section 7 must include the core mathematical equations in KaTeX with explanation of every symbol
- Section 9 must include a working minimal PyTorch class (30–80 lines) with forward() and a shape-verified test at the bottom
- Section 13 must contain 5 interview Q&A pairs with full answers
- Every section must be substantive — minimum 150 words each
- Use `:::note`, `:::tip`, `:::warning` admonitions where helpful

---

## BATCH CNN-1 — Classic CNNs

**Create these files:**
- `src/content/architectures/zfnet/content.mdx`
- `src/content/architectures/vggnet/content.mdx`
- `src/content/architectures/wideresnet/content.mdx`
- `src/content/architectures/resnext/content.mdx`
- `src/content/architectures/senet/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| zfnet | ZFNet | 2013 | Zeiler, Fergus | Deconvnet visualization, larger stride-1 filters in layer 1 | Beginner |
| vggnet | VGGNet | 2014 | Simonyan, Zisserman | 3×3 conv stacking uniformly — depth beats large filters | Beginner |
| wideresnet | WideResNet | 2016 | Zagoruyko, Komodakis | Wider (more filters per layer) rather than deeper — better GPU utilization | Intermediate |
| resnext | ResNeXt | 2017 | Xie et al. | Grouped convolutions — "cardinality" as a third scaling dimension alongside width/depth | Intermediate |
| senet | SENet | 2017 | Hu et al. | Squeeze-and-Excitation blocks — channel attention via global average pool + FC gates | Intermediate |

**Context:** These are all CNN Architectures. Parents: ZFNet←AlexNet, VGGNet←AlexNet, WideResNet←ResNet, ResNeXt←ResNet, SENet←ResNeXt. They live in the ImageNet era (2013–2017). Connect each to its parent in Historical Context.

---

## BATCH CNN-2 — Inception Family

**Create these files:**
- `src/content/architectures/inception-v2-v3/content.mdx`
- `src/content/architectures/inception-v4-inception-resnet/content.mdx`
- `src/content/architectures/xception/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| inception-v2-v3 | Inception v2/v3 | 2015 | Szegedy et al. | Factorized convolutions: replace 5×5 with two 3×3; 7×7 with 1×7+7×1; add batch norm | Intermediate |
| inception-v4-inception-resnet | Inception v4 / Inception-ResNet | 2016 | Szegedy et al. | Residual connections inside Inception modules — combines ResNet skip connections with Inception parallelism | Intermediate |
| xception | Xception | 2016 | Chollet | Extreme Inception: fully factorized into depthwise separable convolutions with pointwise projection | Intermediate |

**Context:** Inception family. Parent chain: GoogLeNet/Inception-v1 → v2/v3 → v4 and Xception. Xception replaces Inception modules with depthwise separable convolutions entirely. All 3 must explain what depthwise separable convolutions are (spatial filter per channel, then 1×1 to mix channels).

---

## BATCH CNN-3 — Mobile/Efficient CNNs

**Create these files:**
- `src/content/architectures/mobilenet-v1/content.mdx`
- `src/content/architectures/mobilenet-v2/content.mdx`
- `src/content/architectures/mobilenet-v3/content.mdx`
- `src/content/architectures/shufflenet/content.mdx`
- `src/content/architectures/shufflenet-v2/content.mdx`
- `src/content/architectures/squeezenet/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| mobilenet-v1 | MobileNet v1 | 2017 | Howard et al. | Depthwise separable conv — 8–9× fewer multiply-adds vs standard conv | Intermediate |
| mobilenet-v2 | MobileNet v2 | 2018 | Sandler et al. | Inverted residual + linear bottleneck — expands then contracts, skip on narrow end | Intermediate |
| mobilenet-v3 | MobileNet v3 | 2019 | Howard et al. | NAS-found cells + SE blocks + hard-swish activation | Intermediate |
| shufflenet | ShuffleNet | 2017 | Zhang et al. | Channel shuffle operation after group conv — allows cross-group information flow | Intermediate |
| shufflenet-v2 | ShuffleNet v2 | 2018 | Ma et al. | Channel split instead of group conv; guidelines based on memory access cost (MAC) | Intermediate |
| squeezenet | SqueezeNet | 2016 | Iandola et al. | Fire modules (squeeze 1×1 + expand 1×1/3×3) — AlexNet accuracy at 50× fewer parameters | Intermediate |

**Context:** All mobile/edge-focused CNN architectures. Section 3 for each must quantify the efficiency: MobileNetV1 reduces ops from $D_K^2 \cdot M \cdot N \cdot D_F^2$ to $D_K^2 \cdot M \cdot D_F^2 + M \cdot N \cdot D_F^2$ where $D_K$ = kernel size, $M/N$ = channels, $D_F$ = feature map size.

---

## BATCH CNN-4 — NAS and Modern CNNs

**Create these files:**
- `src/content/architectures/nasnet/content.mdx`
- `src/content/architectures/darts/content.mdx`
- `src/content/architectures/efficientnetv2/content.mdx`
- `src/content/architectures/regnet/content.mdx`
- `src/content/architectures/convnext/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| nasnet | NASNet | 2018 | Zoph et al. | Neural Architecture Search on proxy task; Normal Cell + Reduction Cell applied repeatedly | Advanced |
| darts | DARTS | 2018 | Liu et al. | Differentiable NAS — relaxes discrete search to continuous via softmax over operations; gradient descent finds architecture | Advanced |
| efficientnetv2 | EfficientNetV2 | 2021 | Tan, Le | Fused-MBConv in early layers + progressive training (smaller→larger images) — 4× faster training than EfficientNet | Advanced |
| regnet | RegNet | 2020 | Radosavovic et al. | Design space analysis — analytical width/depth/group width scaling laws replace search | Advanced |
| convnext | ConvNeXt | 2022 | Liu et al. | Modernized ResNet: inverted bottleneck, 7×7 depthwise conv, GELU, LayerNorm — matches Swin Transformer accuracy | Advanced |

**Context:** NASNet/DARTS explain neural architecture search. DARTS section 7 must explain the continuous relaxation: $\bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o' \in \mathcal{O}} \exp(\alpha_{o'}^{(i,j)})} o(x)$. ConvNeXt section 2 must describe the Swin Transformer competition that motivated it.

---

## BATCH SEQ-1 — Sequence Models

**Create these files:**
- `src/content/architectures/bi-directional-rnn/content.mdx`
- `src/content/architectures/attention-seq2seq/content.mdx`
- `src/content/architectures/wavenet/content.mdx`
- `src/content/architectures/tcn/content.mdx`
- `src/content/architectures/transformer-xl/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| bi-directional-rnn | Bi-directional RNN | 1997 | Schuster, Paliwal | Two RNNs scanning forward and backward; concatenate hidden states — bidirectional context | Beginner |
| attention-seq2seq | Attention Seq2Seq | 2015 | Bahdanau et al. | Additive attention: decoder computes alignment scores over all encoder states at each step — dynamic source context | Intermediate |
| wavenet | WaveNet | 2016 | van den Oord et al. | Dilated causal convolutions — exponentially growing receptive field for raw audio generation | Intermediate |
| tcn | TCN | 2018 | Bai et al. | Temporal Convolutional Network: causal dilated conv + residual connections — outperforms RNNs on many sequence benchmarks | Intermediate |
| transformer-xl | Transformer-XL | 2019 | Dai et al. | Segment-level recurrence with cached hidden states + relative sinusoidal position encoding — context beyond single segment | Intermediate |

**Context:** Parent chain: RNN → Bi-RNN → Seq2Seq → Attention Seq2Seq → Transformer. WaveNet section 4 must diagram the dilated causal convolution stack showing dilation rates 1,2,4,8,16. TCN section 7 must show how receptive field grows: $r = 1 + 2(k-1)(2^L - 1)$ where $k$ = kernel size, $L$ = layers.

---

## BATCH SEQ-2 — State Space Models

**Create these files:**
- `src/content/architectures/s4/content.mdx`
- `src/content/architectures/mamba/content.mdx`
- `src/content/architectures/rwkv/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| s4 | S4 | 2021 | Gu et al. | Structured State Space Sequence model — HiPPO matrix initialization enables long-range memory; computes as convolution during training, recurrence during inference | Advanced |
| mamba | Mamba | 2023 | Gu, Dao | Selective state spaces — $\Delta, B, C$ are input-dependent (unlike S4); hardware-aware parallel scan eliminates attention's O(T²) | Advanced |
| rwkv | RWKV | 2023 | Peng et al. | Token mixing via time-decay weighted key-value: $y_t = \frac{\sum_{i \leq t} e^{-(t-i)w+k_i} v_i}{\sum_{i \leq t} e^{-(t-i)w+k_i}}$ — RNN inference, Transformer training | Advanced |

**Context:** These are the post-Transformer sequence model alternatives. S4 section 7 must derive the state space: $h'(t) = Ah(t) + Bx(t)$, $y(t) = Ch(t) + Dx(t)$ and explain discretization to $\bar{A}, \bar{B}$. Mamba must explain why selectivity matters: S4's $A,B,C$ are fixed so it can't selectively ignore content. RWKV must compare to attention at inference time (O(1) per token vs O(T) for KV cache growth).

---

## BATCH NLP-1 — Word Embeddings

**Create these files:**
- `src/content/architectures/word2vec/content.mdx`
- `src/content/architectures/glove/content.mdx`
- `src/content/architectures/fasttext/content.mdx`
- `src/content/architectures/elmo/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| word2vec | Word2Vec | 2013 | Mikolov et al. | Skip-gram and CBOW: predict context from word or word from context — dense 300d embeddings from cooccurrence | Beginner |
| glove | GloVe | 2014 | Pennington et al. | Global Vectors: factorize word cooccurrence matrix — combines global statistics with local context window | Beginner |
| fasttext | FastText | 2017 | Joulin et al. | Character n-gram subword embeddings: word = sum of n-gram vectors — handles OOV and morphology | Beginner |
| elmo | ELMo | 2018 | Peters et al. | Contextual embeddings from deep Bi-LSTM — same word gets different vectors depending on sentence context | Intermediate |

**Context:** Pre-BERT NLP representations. Section 7 for Word2Vec must show the skip-gram objective: $\mathcal{L} = -\sum_{(w,c) \in D} \log \sigma(v_c \cdot v_w) - \sum_{(w,c') \in D'} \log \sigma(-v_{c'} \cdot v_w)$. GloVe: $J = \sum_{i,j} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$. ELMo must contrast static vs contextual embeddings.

---

## BATCH NLP-2 — BERT Variants

**Create these files:**
- `src/content/architectures/roberta/content.mdx`
- `src/content/architectures/albert/content.mdx`
- `src/content/architectures/distilbert/content.mdx`
- `src/content/architectures/bart/content.mdx`
- `src/content/architectures/electra/content.mdx`
- `src/content/architectures/deberta/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| roberta | RoBERTa | 2019 | Liu et al. | BERT done right: no NSP, dynamic masking, 10× more data, larger batches — significant NLU gains | Intermediate |
| albert | ALBERT | 2019 | Lan et al. | Factorized embedding (vocab→small→hidden) + cross-layer parameter sharing — 18× fewer params than BERT-large | Intermediate |
| distilbert | DistilBERT | 2019 | Sanh et al. | Knowledge distillation from BERT: 6 layers from 12, cosine embedding + attention transfer loss — 40% smaller, 60% faster | Intermediate |
| bart | BART | 2019 | Lewis et al. | Denoising autoencoder for seq2seq: corrupt text in many ways (mask, delete, permute, rotate, infill) then reconstruct — excels at generation | Intermediate |
| electra | ELECTRA | 2020 | Clark et al. | Replaced Token Detection: small generator masks tokens, discriminator detects replacements — 4× more efficient pre-training signal | Advanced |
| deberta | DeBERTa | 2020 | He et al. | Disentangled attention: separate content and position vectors; enhanced mask decoder with absolute position — SuperGLUE SOTA | Advanced |

**Context:** All derived from BERT (2018). ALBERT section 7 must show factorized embedding: vocab $V$ → $E$ (small) → $H$ (hidden), reducing $V \times H$ to $V \times E + E \times H$. ELECTRA section 4 must diagram the generator-discriminator pipeline. DeBERTa must explain why disentangling content and position improves attention.

---

## BATCH NLP-3 — Cross-lingual and XL Models

**Create these files:**
- `src/content/architectures/xlnet/content.mdx`
- `src/content/architectures/xlm/content.mdx`
- `src/content/architectures/xlm-roberta/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| xlnet | XLNet | 2019 | Yang et al. | Permutation Language Modeling: predict token $x_{z_t}$ given all other tokens in permuted order — bidirectional context without corruption | Advanced |
| xlm | XLM | 2019 | Lample, Conneau | Cross-lingual masked LM: shared BPE vocabulary across 100 languages; Translation Language Modeling with aligned pairs | Intermediate |
| xlm-roberta | XLM-RoBERTa | 2019 | Conneau et al. | RoBERTa training recipe applied to 100 languages, 2.5TB of filtered CC data — strong zero-shot cross-lingual transfer | Intermediate |

**Context:** XLNet section 7 must show the permutation objective: $\max_\theta \mathbb{E}_{z \sim \mathcal{Z}_T} \left[\sum_{t=1}^T \log p_\theta(x_{z_t} | x_{z_{<t}})\right]$ and explain why two-stream attention is needed. XLM section 4 must show the TLM objective diagram with language embeddings.

---

## BATCH LLM-1 — GPT Family

**Create these files:**
- `src/content/architectures/gpt-1/content.mdx`
- `src/content/architectures/gpt-2/content.mdx`
- `src/content/architectures/gpt-3/content.mdx`
- `src/content/architectures/codex/content.mdx`
- `src/content/architectures/instructgpt/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| gpt-1 | GPT-1 | 2018 | Radford et al. | Decoder-only transformer + unsupervised pre-training then supervised fine-tuning — first GPT paper | Intermediate |
| gpt-2 | GPT-2 | 2019 | Radford et al. | Scale to 1.5B params; zero-shot task generalization with task-conditioned prompting; no fine-tuning needed | Intermediate |
| gpt-3 | GPT-3 | 2020 | Brown et al. | 175B parameters; few-shot in-context learning — task examples in prompt, no gradient update; emergent capabilities | Intermediate |
| codex | Codex | 2021 | Chen et al. | GPT-3 fine-tuned on 159GB of GitHub code — powers GitHub Copilot; HumanEval benchmark | Intermediate |
| instructgpt | InstructGPT | 2022 | Ouyang et al. | RLHF: SFT on demonstrations → reward model from comparisons → PPO against reward model — aligns GPT-3 to follow instructions | Advanced |

**Context:** The GPT lineage. GPT-1 section 9 PyTorch must implement the decoder-only transformer block. GPT-3 section 7 must explain in-context learning formally: no weight updates, just $p(y|x, \text{examples})$. InstructGPT section 4 must diagram the 3-phase RLHF pipeline clearly with SFT, RM, PPO stages.

---

## BATCH LLM-2 — GPT-4 and Frontier Models

**Create these files:**
- `src/content/architectures/gpt-4/content.mdx`
- `src/content/architectures/gpt-4o/content.mdx`
- `src/content/architectures/palm/content.mdx`
- `src/content/architectures/palm-2/content.mdx`
- `src/content/architectures/chinchilla/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| gpt-4 | GPT-4 | 2023 | OpenAI | Multimodal inputs; 8192→128K context window; significantly improved reasoning; sparse MoE rumored | Advanced |
| gpt-4o | GPT-4o | 2024 | OpenAI | Omni-modal: audio+image+text in single end-to-end model; 2× faster, 50% cheaper than GPT-4 | Advanced |
| palm | PaLM | 2022 | Chowdhery et al. | 540B params via Pathways (8192 TPU chips); multi-task + multilingual at scale; chain-of-thought emergent at 540B | Advanced |
| palm-2 | PaLM 2 | 2023 | Google | Chinchilla-optimal training recipe; compute-multilingual data mix; significantly better at reasoning and code | Advanced |
| chinchilla | Chinchilla | 2022 | Hoffmann et al. | Compute-optimal scaling laws: train for 20 tokens per parameter; 70B model beats 280B Gopher — overturned GPT-3 training wisdom | Advanced |

**Context:** Chinchilla is the most important paper in this batch for ML engineers — must include the scaling law derivation: $L(N,D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$ and the optimal allocation $N_{opt} \propto C^{0.5}$, $D_{opt} \propto C^{0.5}$. GPT-4 section 2 must cover the "too dangerous to release" story of GPT-2 and how safety concerns evolved.

---

## BATCH LLM-3 — Open Source LLMs

**Create these files:**
- `src/content/architectures/llama-2/content.mdx`
- `src/content/architectures/llama-3/content.mdx`
- `src/content/architectures/mistral-7b/content.mdx`
- `src/content/architectures/mixtral-8x7b/content.mdx`
- `src/content/architectures/falcon/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| llama-2 | LLaMA 2 | 2023 | Touvron et al. | Grouped Query Attention (GQA); RLHF alignment (Llama-2-Chat); 4096→100K context with positional interpolation | Intermediate |
| llama-3 | LLaMA 3 | 2024 | Meta | 128K token context; improved tokenizer (128K vocab); GQA; 8B/70B/405B sizes; much stronger code/multilingual | Intermediate |
| mistral-7b | Mistral 7B | 2023 | Mistral AI | Sliding Window Attention (SWA): each token attends only to 4096 prior tokens; GQA; outperforms LLaMA-2 13B at 7B params | Intermediate |
| mixtral-8x7b | Mixtral 8×7B | 2023 | Mistral AI | Sparse Mixture of Experts: 8 expert FFNs per layer, router selects 2 per token — 46.7B total params but 12.9B active | Advanced |
| falcon | Falcon | 2023 | TII Abu Dhabi | Multi-Query Attention (MQA): single KV head for all query heads; trained on RefinedWeb (curated CC); efficient inference | Intermediate |

**Context:** GQA section 7 for LLaMA 2 must compare MHA/GQA/MQA: $h$ query heads, $h$ KV heads (MHA) vs $g$ KV heads where $1 < g < h$ (GQA) vs $g=1$ KV heads (MQA). Mixtral section 7 must derive the MoE throughput: with 8 experts and top-2 routing, active params = 2/8 = 25% of total FFN params.

---

## BATCH LLM-4 — Remaining Open LLMs

**Create these files:**
- `src/content/architectures/gopher/content.mdx`
- `src/content/architectures/opt/content.mdx`
- `src/content/architectures/bloom/content.mdx`
- `src/content/architectures/phi-2/content.mdx`
- `src/content/architectures/qwen/content.mdx`
- `src/content/architectures/deepseek-v2/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| gopher | Gopher | 2021 | Rae et al. | 280B dense transformer; RMSNorm instead of LayerNorm; relative positional encodings; strong on knowledge tasks | Advanced |
| opt | OPT | 2022 | Zhang et al. | Open Pre-trained Transformer: full training reproducibility with public codebase and logbook; 125M–175B | Intermediate |
| bloom | BLOOM | 2022 | BigScience | 176B multilingual (46 languages + 13 programming); ALiBi position bias; trained by 1000+ researchers | Intermediate |
| phi-2 | Phi-2 | 2023 | Microsoft | 2.7B model trained on textbook-quality synthetic data (Textbooks Are All You Need); punches far above weight class | Intermediate |
| qwen | Qwen | 2023 | Alibaba | 7B–72B; strong Chinese+English+code; NTK-aware rotary position for long context; tied embedding | Intermediate |
| deepseek-v2 | DeepSeek-V2 | 2024 | DeepSeek AI | MLA (Multi-head Latent Attention): compress KV cache via low-rank projection; MoE with 236B total / 21B active params | Advanced |

**Context:** DeepSeek-V2 section 7 must explain MLA: instead of caching $(k_t, v_t)$ for each head, compress into a single latent vector $c_t^{KV} = W^{DKV} h_t$ and reconstruct at inference — dramatically reduces KV cache memory. BLOOM section 2 must cover the BigScience initiative (crowdsourced research).

---

## BATCH VIS-1 — Vision Transformers

**Create these files:**
- `src/content/architectures/deit/content.mdx`
- `src/content/architectures/swin-v2/content.mdx`
- `src/content/architectures/beit/content.mdx`
- `src/content/architectures/mae/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| deit | DeiT | 2021 | Touvron et al. | Distillation token alongside class token — trains ViT on ImageNet-1K without JFT-300M; teacher CNN provides hard distillation | Intermediate |
| swin-v2 | Swin v2 | 2022 | Liu et al. | Cosine attention (dot product replaced by $\cos$); log-spaced continuous position bias; scales to 3B params and 1536² resolution | Advanced |
| beit | BEiT | 2021 | Bao et al. | BERT-style masked image modeling: tokenize image via dVAE, mask 40% patches, predict visual tokens — self-supervised ViT pre-training | Advanced |
| mae | MAE | 2021 | He et al. | Masked Autoencoder: mask 75% of patches, encode only visible 25%, lightweight decoder reconstructs pixels — simple and scalable | Advanced |

**Context:** DeiT section 6 tensor flow must show how the distillation token works (3 tokens: class, distillation, patches). MAE section 4 must diagram the asymmetric encoder-decoder (large encoder on 25%, small decoder on 100%). BEiT section 3 must explain why predicting visual tokens is better than pixels.

---

## BATCH VIS-2 — Contrastive and Hybrid Vision

**Create these files:**
- `src/content/architectures/align/content.mdx`
- `src/content/architectures/coca/content.mdx`
- `src/content/architectures/maxvit/content.mdx`
- `src/content/architectures/pvt/content.mdx`
- `src/content/architectures/cvt/content.mdx`
- `src/content/architectures/convnext-v2/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| align | ALIGN | 2021 | Jia et al. | Noisy 1.8B image-text pairs — scale over curation; EfficientNet image + BERT text; matches CLIP with noisier but larger data | Advanced |
| coca | CoCa | 2022 | Yu et al. | Contrastive + captioning loss jointly; unimodal encoder + multimodal decoder; single model for classification, retrieval, captioning | Advanced |
| maxvit | MaxViT | 2022 | Tu et al. | Multi-axis attention: local window attention + dilated global grid attention alternating in each block — O(N) complexity, global receptive field | Advanced |
| pvt | PVT | 2021 | Wang et al. | Pyramid Vision Transformer: spatial reduction attention (downsample KV) at multiple scales — dense prediction backbone | Intermediate |
| cvt | CvT | 2021 | Wu et al. | Convolutional token embedding + convolutional projection in attention — adds locality inductive bias to ViT | Intermediate |
| convnext-v2 | ConvNeXt V2 | 2023 | Woo et al. | Fully Convolutional MAE (FCMAE) self-supervised pre-training + Global Response Normalization (GRN) layer | Advanced |

---

## BATCH SEG-1 — Segmentation Architectures

**Create these files:**
- `src/content/architectures/segnet/content.mdx`
- `src/content/architectures/u-net-2/content.mdx`
- `src/content/architectures/attention-u-net/content.mdx`
- `src/content/architectures/deeplab-v1/content.mdx`
- `src/content/architectures/deeplab-v2/content.mdx`
- `src/content/architectures/deeplab-v3/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| segnet | SegNet | 2015 | Badrinarayanan et al. | Encoder-decoder with max-pooling indices passed to decoder — efficient upsampling without learnable transposed conv | Beginner |
| u-net-2 | U-Net++ | 2018 | Zhou et al. | Nested dense skip connections — intermediate nodes on skip paths aggregate features from all previous scales | Intermediate |
| attention-u-net | Attention U-Net | 2018 | Oktay et al. | Attention gates on skip connections: gate signal from decoder selectively amplifies relevant encoder features | Intermediate |
| deeplab-v1 | DeepLab v1 | 2014 | Chen et al. | Atrous (dilated) convolutions preserve spatial resolution while increasing receptive field; CRF post-processing | Intermediate |
| deeplab-v2 | DeepLab v2 | 2016 | Chen et al. | ASPP: Atrous Spatial Pyramid Pooling — parallel atrous conv at rates 6,12,18,24 captures multi-scale context | Intermediate |
| deeplab-v3 | DeepLab v3 | 2017 | Chen et al. | Improved ASPP with batch norm + image-level global average pooling; removes CRF; encoder-only | Intermediate |

**Context:** DeepLab series must all show atrous convolution formula: output $y[i] = \sum_k x[i+r \cdot k] \cdot w[k]$ where $r$ = dilation rate. ASPP diagram in section 4 must show 4 parallel branches at different rates merging via 1×1 conv. U-Net++ section 4 must diagram the nested dense skip connections clearly.

---

## BATCH SEG-2 — Advanced Segmentation

**Create these files:**
- `src/content/architectures/pspnet/content.mdx`
- `src/content/architectures/mask-r-cnn/content.mdx`
- `src/content/architectures/panoptic-fpn/content.mdx`
- `src/content/architectures/segformer/content.mdx`
- `src/content/architectures/mask2former/content.mdx`
- `src/content/architectures/sam/content.mdx`
- `src/content/architectures/sam-2/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| pspnet | PSPNet | 2016 | Zhao et al. | Pyramid Pooling Module: pool at 1×1, 2×2, 3×3, 6×6 and upsample+concat — global scene understanding | Intermediate |
| mask-r-cnn | Mask R-CNN | 2017 | He et al. | RoIAlign (bilinear interpolation replaces RoIPool quantization) + mask head — instance segmentation | Advanced |
| panoptic-fpn | Panoptic FPN | 2019 | Kirillov et al. | Single FPN backbone for both semantic (stuff) and instance (things) — unified panoptic segmentation | Advanced |
| segformer | SegFormer | 2021 | Xie et al. | Hierarchical mix-transformer encoder + lightweight all-MLP decoder — efficient semantic segmentation | Advanced |
| mask2former | Mask2Former | 2022 | Cheng et al. | Masked attention: restrict cross-attention to predicted foreground regions per query — universal segmentation | Advanced |
| sam | SAM | 2023 | Kirillov et al. | Segment Anything: promptable with point/box/text; 1B+ masks training set; heavy ViT encoder + light prompt encoder + mask decoder | Advanced |
| sam-2 | SAM 2 | 2024 | Ravi et al. | Streaming memory for video: memory bank stores per-object features across frames; real-time video segmentation | Advanced |

**Context:** Mask R-CNN section 7 must derive RoIAlign with bilinear interpolation math. SAM section 4 must diagram the 3-component architecture (image encoder, prompt encoder, mask decoder) clearly. SAM-2 section 2 must explain why video segmentation is fundamentally harder than image segmentation.

---

## BATCH DET-1 — R-CNN Family

**Create these files:**
- `src/content/architectures/r-cnn/content.mdx`
- `src/content/architectures/fast-r-cnn/content.mdx`
- `src/content/architectures/faster-r-cnn/content.mdx`
- `src/content/architectures/fpn/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| r-cnn | R-CNN | 2013 | Girshick et al. | Selective Search proposals → warp each to fixed size → CNN features → SVM classifier — first deep object detection | Beginner |
| fast-r-cnn | Fast R-CNN | 2015 | Girshick | RoI Pooling: run CNN once on full image, project proposals onto feature map — end-to-end training, 9× faster than R-CNN | Intermediate |
| faster-r-cnn | Faster R-CNN | 2015 | Ren et al. | Region Proposal Network shares conv features with detection head — proposals generated inside network, 10ms vs 2s | Intermediate |
| fpn | FPN | 2016 | Lin et al. | Feature Pyramid Network: top-down pathway + lateral connections — multi-scale feature hierarchy from single-scale backbone | Intermediate |

**Context:** Section 2 must tell the R-CNN → Fast → Faster progression as a story of eliminating bottlenecks. FPN section 4 must diagram the bottom-up backbone, top-down pathway, and lateral connections clearly with feature map sizes. Faster R-CNN section 7 must explain RPN anchor boxes: 9 anchors (3 scales × 3 ratios) at each spatial location.

---

## BATCH DET-2 — YOLO Family

**Create these files:**
- `src/content/architectures/yolo-v1/content.mdx`
- `src/content/architectures/yolo-v2/content.mdx`
- `src/content/architectures/yolo-v3/content.mdx`
- `src/content/architectures/yolo-v4/content.mdx`
- `src/content/architectures/yolo-v5/content.mdx`
- `src/content/architectures/yolov8/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| yolo-v1 | YOLO v1 | 2015 | Redmon et al. | Grid-based single-pass detection: divide image into S×S grid, each cell predicts B boxes + C class probs simultaneously | Intermediate |
| yolo-v2 | YOLO v2 | 2016 | Redmon, Farhadi | Anchor boxes (k-means on training boxes), BatchNorm on all layers, Darknet-19 backbone, 9000-class WordTree | Intermediate |
| yolo-v3 | YOLO v3 | 2018 | Redmon, Farhadi | Multi-scale predictions at 3 scales (Darknet-53 backbone); sigmoid class prediction (not softmax) for multilabel | Intermediate |
| yolo-v4 | YOLO v4 | 2020 | Bochkovskiy et al. | CSPDarknet53 + PANet + CIoU loss + mosaic augmentation + self-adversarial training + DropBlock | Intermediate |
| yolo-v5 | YOLO v5 | 2020 | Ultralytics | PyTorch rewrite; AutoAnchor; compound scaling (n/s/m/l/x variants); focus slice layer; aggressive augmentation | Intermediate |
| yolov8 | YOLOv8 | 2023 | Ultralytics | Anchor-free detection; decoupled head; C2f module replacing C3; supports detect/segment/pose/classify tasks | Intermediate |

**Context:** YOLOv1 section 7 must show the loss function: $\lambda_{coord} \sum + \lambda_{noobj} \sum + \sum_{class}$, distinguishing responsible vs non-responsible cells. YOLOv2 section 7 must show how anchor boxes work with k-means clustering. Each version section 2 must connect to the previous — this is a linear evolution story.

---

## BATCH DET-3 — Single-Stage and Transformer Detection

**Create these files:**
- `src/content/architectures/ssd/content.mdx`
- `src/content/architectures/retinanet/content.mdx`
- `src/content/architectures/efficientdet/content.mdx`
- `src/content/architectures/detr/content.mdx`
- `src/content/architectures/deformable-detr/content.mdx`
- `src/content/architectures/cornernet/content.mdx`
- `src/content/architectures/centernet/content.mdx`
- `src/content/architectures/fcos/content.mdx`
- `src/content/architectures/rt-detr/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| ssd | SSD | 2015 | Liu et al. | Default boxes at multiple scales and ratios from multiple feature maps — single shot, no proposals | Intermediate |
| retinanet | RetinaNet | 2017 | Lin et al. | Focal Loss: $(1-p_t)^\gamma \log(p_t)$ — down-weights easy negatives, solves class imbalance in dense detection | Intermediate |
| efficientdet | EfficientDet | 2019 | Tan et al. | BiFPN (weighted bidirectional FPN) + compound scaling of backbone/BiFPN/head simultaneously | Advanced |
| detr | DETR | 2020 | Carion et al. | Transformer encoder-decoder + bipartite matching loss — end-to-end detection, no NMS, no anchors | Advanced |
| deformable-detr | Deformable DETR | 2020 | Zhu et al. | Deformable attention: each query attends to small set of learned key sampling points — 10× faster convergence than DETR | Advanced |
| cornernet | CornerNet | 2018 | Law, Deng | Detect top-left and bottom-right corners as heatmaps; associative embedding pairs corners into boxes | Advanced |
| centernet | CenterNet | 2019 | Zhou et al. | Objects as points: center heatmap + WH regression + offset — simpler than CornerNet | Intermediate |
| fcos | FCOS | 2019 | Tian et al. | Fully convolutional, anchor-free: each location predicts (l,t,r,b) distances to box edges + centerness score | Intermediate |
| rt-detr | RT-DETR | 2023 | Lv et al. | Efficient hybrid encoder + uncertainty-minimal query selection — first real-time end-to-end detector matching YOLO speed | Advanced |

**Context:** RetinaNet section 7 must plot the focal loss curve showing how $\gamma$ affects easy vs hard examples. DETR section 4 must diagram the bipartite matching process clearly. Section 3 for each anchor-free method must explain why removing anchors simplifies the pipeline.

---

## BATCH DIF-1 — Core Diffusion Models

**Create these files:**
- `src/content/architectures/ddpm/content.mdx`
- `src/content/architectures/ddim/content.mdx`
- `src/content/architectures/score-sde/content.mdx`
- `src/content/architectures/ldm/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| ddpm | DDPM | 2020 | Ho et al. | Denoising Diffusion Probabilistic Model: T-step forward Markov chain adds Gaussian noise; reverse chain denoises; simplified $\ell_2$ loss on predicted noise | Intermediate |
| ddim | DDIM | 2020 | Song et al. | Non-Markovian reverse process: deterministic sampling skips steps — 10–50× fewer function evaluations; enables latent space interpolation | Intermediate |
| score-sde | Score SDE | 2020 | Song et al. | Continuous-time stochastic differential equations unify DDPM and score matching; VP-SDE and VE-SDE as special cases | Advanced |
| ldm | LDM | 2021 | Rombach et al. | Run diffusion in VQ-VAE latent space (4–8× compressed) + cross-attention conditioning — enables text-to-image at consumer GPU scale | Advanced |

**Context:** DDPM section 7 is critical — must include: forward process $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$, closed-form $q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$, and the simplified loss $\mathcal{L}_{simple} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t,t)\|^2]$. DDIM section 7 must show the deterministic update rule. LDM section 4 must diagram the perceptual compression stage (train separately) then the diffusion stage.

---

## BATCH DIF-2 — Advanced Diffusion

**Create these files:**
- `src/content/architectures/dall-e-2/content.mdx`
- `src/content/architectures/imagen/content.mdx`
- `src/content/architectures/dit/content.mdx`
- `src/content/architectures/sdxl/content.mdx`
- `src/content/architectures/controlnet/content.mdx`
- `src/content/architectures/consistency-models/content.mdx`
- `src/content/architectures/flux/content.mdx`
- `src/content/architectures/stable-diffusion-3/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| dall-e-2 | DALL-E 2 | 2022 | Ramesh et al. | CLIP image embedding as diffusion prior; unCLIP decoder reconstructs image from CLIP latent | Advanced |
| imagen | Imagen | 2022 | Saharia et al. | Large frozen T5-XXL text encoder (4.6B) + cascaded pixel-space diffusion (64→256→1024) | Advanced |
| dit | DiT | 2022 | Peebles, Xie | Diffusion Transformer: replace U-Net backbone with ViT blocks; adaLN-Zero conditions on timestep+class | Advanced |
| sdxl | SDXL | 2023 | Podell et al. | Larger U-Net (2.6B), dual CLIP encoders (OpenCLIP + CLIP-L), refinement model for high-freq details, 1024px native | Intermediate |
| controlnet | ControlNet | 2023 | Zhang et al. | Trainable copy of SD encoder blocks connected via zero-convolution — conditional control from pose/depth/edge/segmentation | Intermediate |
| consistency-models | Consistency Models | 2023 | Song et al. | Self-consistency along ODE trajectory: $f_\theta(x_t, t) = f_\theta(x_{t'}, t')$ for any $t, t'$ on same trajectory — single-step generation | Advanced |
| flux | Flux | 2024 | Black Forest Labs | Flow matching + rectified flows + MMDiT architecture; 12B params; separate text and image processing streams | Advanced |
| stable-diffusion-3 | SD3 | 2024 | Stability AI | Multimodal Diffusion Transformer (MMDiT): separate weights for text and image streams, joined in attention | Advanced |

**Context:** DiT section 7 must explain adaLN-Zero: shift and scale from timestep/class embedding, initialized to output zero at training start. Consistency Models section 7 must explain why the boundary condition $f(x_\epsilon, \epsilon) = x_\epsilon$ is crucial. ControlNet section 9 PyTorch must show the zero-convolution initialization.

---

## BATCH GAN-1 — GAN Architectures

**Create these files:**
- `src/content/architectures/dcgan/content.mdx`
- `src/content/architectures/conditional-gan/content.mdx`
- `src/content/architectures/pix2pix/content.mdx`
- `src/content/architectures/cyclegan/content.mdx`
- `src/content/architectures/wgan/content.mdx`
- `src/content/architectures/wgan-gp/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| dcgan | DCGAN | 2015 | Radford et al. | Architecture guidelines: strided conv (no pooling), BatchNorm, ReLU/LeakyReLU — first stable deep GAN | Intermediate |
| conditional-gan | Conditional GAN | 2014 | Mirza, Osindero | Condition G and D on class label y — control what is generated | Beginner |
| pix2pix | Pix2Pix | 2016 | Isola et al. | Paired image translation: cGAN loss + L1 loss; PatchGAN discriminator operates on 70×70 patches | Intermediate |
| cyclegan | CycleGAN | 2017 | Zhu et al. | Unpaired translation via cycle consistency: $\mathcal{L}_{cyc} = \|F(G(x)) - x\|_1 + \|G(F(y)) - y\|_1$ | Intermediate |
| wgan | WGAN | 2017 | Arjovsky et al. | Wasserstein-1 distance as training objective — meaningful gradient everywhere; weight clipping for Lipschitz constraint | Advanced |
| wgan-gp | WGAN-GP | 2017 | Gulrajani et al. | Gradient penalty replaces weight clipping: $\lambda \mathbb{E}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$ — better stability | Advanced |

**Context:** WGAN section 7 must derive why JS divergence fails when distributions don't overlap and why Earth Mover distance doesn't. CycleGAN section 7 must show all three loss components: adversarial + cycle + identity. PatchGAN section 4 must explain why patch-level discriminator works better than full-image for texture.

---

## BATCH GAN-2 — High-Quality GANs

**Create these files:**
- `src/content/architectures/progan/content.mdx`
- `src/content/architectures/biggan/content.mdx`
- `src/content/architectures/stylegan/content.mdx`
- `src/content/architectures/stylegan2/content.mdx`
- `src/content/architectures/stylegan3/content.mdx`
- `src/content/architectures/gaugan-spade/content.mdx`
- `src/content/architectures/vqgan/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| progan | ProGAN | 2017 | Karras et al. | Progressive growing: start at 4×4, add layers for higher resolution gradually — stable high-res training | Advanced |
| biggan | BigGAN | 2018 | Brock et al. | Class-conditional at scale (512 channels): truncation trick, orthogonal regularization, class conditioning via projection | Advanced |
| stylegan | StyleGAN | 2018 | Karras et al. | Mapping network $z \to w$; AdaIN style injection at each conv layer; stochastic per-pixel noise; no input z to synthesis | Advanced |
| stylegan2 | StyleGAN2 | 2019 | Karras et al. | Weight demodulation replaces AdaIN; lazy regularization; path length regularization — removes water droplet artifacts | Advanced |
| stylegan3 | StyleGAN3 | 2021 | Karras et al. | Alias-free: continuous signal theory, filtered nonlinearities — translation/rotation equivariance | Expert |
| gaugan-spade | GauGAN/SPADE | 2019 | Park et al. | SPADE: spatially adaptive denormalization from segmentation map; $\gamma$ and $\beta$ are spatial functions of the layout | Advanced |
| vqgan | VQGAN | 2020 | Esser et al. | Vector-quantized GAN: CNN encoder → discrete codebook → CNN decoder + transformer over codes; perceptual + adversarial loss | Advanced |

**Context:** StyleGAN section 4 must diagram the two-network structure (mapping network + synthesis network). StyleGAN2 section 7 must explain weight demodulation: $w'_{ijk} = w_{ijk} \cdot s_i / \sqrt{\sum_{i,k} (w_{ijk} \cdot s_i)^2 + \epsilon}$. VQGAN section 9 must show how the codebook lookup works in PyTorch with straight-through estimator.

---

## BATCH RL-1 — Value-Based RL

**Create these files:**
- `src/content/architectures/dqn/content.mdx`
- `src/content/architectures/double-dqn/content.mdx`
- `src/content/architectures/dueling-dqn/content.mdx`
- `src/content/architectures/rainbow/content.mdx`
- `src/content/architectures/a3c/content.mdx`
- `src/content/architectures/trpo/content.mdx`
- `src/content/architectures/ppo/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| dqn | DQN | 2013 | Mnih et al. | Experience replay + separate target network — stable deep Q-learning; beats human on 49 Atari games | Intermediate |
| double-dqn | Double DQN | 2015 | van Hasselt et al. | Decouple action selection (online net) and evaluation (target net): $Q(s,\arg\max_{a'} Q_{online}(s',a'), \theta^-)$ | Intermediate |
| dueling-dqn | Dueling DQN | 2015 | Wang et al. | Shared encoder → separate V(s) and A(s,a) streams → combine: $Q = V + (A - \bar{A})$ | Intermediate |
| rainbow | Rainbow | 2017 | Hessel et al. | Combines 6 improvements: DDQN + prioritized replay + dueling + n-step + distributional + noisy nets | Advanced |
| a3c | A3C | 2016 | Mnih et al. | Asynchronous actor-critic: N workers update global network asynchronously — no replay buffer needed | Intermediate |
| trpo | TRPO | 2015 | Schulman et al. | Trust region constraint on policy update: $\mathbb{E}[\text{KL}[\pi_{old}, \pi_{new}]] \leq \delta$ — monotonic improvement guarantee | Advanced |
| ppo | PPO | 2017 | Schulman et al. | Clipped surrogate objective: $\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)$ — simpler TRPO with similar guarantees | Intermediate |

**Context:** DQN section 7 must show why target network and experience replay are both needed (and what happens without each). PPO is the most important algorithm to cover deeply — section 7 must derive the clipped objective from first principles and explain why clipping is sufficient instead of a hard KL constraint. PPO section 9 must include a complete minimal PyTorch PPO training loop.

---

## BATCH RL-2 — Continuous Control and Model-Based RL

**Create these files:**
- `src/content/architectures/ddpg/content.mdx`
- `src/content/architectures/td3/content.mdx`
- `src/content/architectures/sac/content.mdx`
- `src/content/architectures/alphago/content.mdx`
- `src/content/architectures/alphazero/content.mdx`
- `src/content/architectures/muzero/content.mdx`
- `src/content/architectures/dreamer/content.mdx`
- `src/content/architectures/decision-transformer/content.mdx`
- `src/content/architectures/gato/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| ddpg | DDPG | 2015 | Lillicrap et al. | Deterministic policy gradient for continuous actions; off-policy actor-critic with replay buffer; Ornstein-Uhlenbeck noise for exploration | Advanced |
| td3 | TD3 | 2018 | Fujimoto et al. | Twin critics (take min Q to reduce overestimation); delayed policy update; target smoothing with noise | Advanced |
| sac | SAC | 2018 | Haarnoja et al. | Maximum entropy RL: $\pi^* = \arg\max \mathbb{E}[\sum r_t + \alpha H(\pi(\cdot|s_t))]$; twin critics; automatic temperature tuning | Advanced |
| alphago | AlphaGo | 2016 | Silver et al. | Policy network + value network trained on human data then self-play; MCTS guided by networks | Advanced |
| alphazero | AlphaZero | 2017 | Silver et al. | Self-play from random initialization — no human data; single network for policy + value; mastered Go/Chess/Shogi | Expert |
| muzero | MuZero | 2019 | Schrittwieser et al. | Learns representation, dynamics, and reward models without knowing rules; planning in learned latent model | Expert |
| dreamer | Dreamer | 2019 | Hafner et al. | RSSM world model in latent space; actor-critic trained on imagined rollouts; no real environment needed for policy update | Expert |
| decision-transformer | Decision Transformer | 2021 | Chen et al. | RL as sequence modeling: GPT on (return, state, action) triples; condition on desired return-to-go | Advanced |
| gato | Gato | 2022 | Reed et al. | Single transformer on 604 tasks: tokenize everything (pixels, text, actions) into one sequence; generalist agent | Expert |

**Context:** SAC section 7 must derive the entropy-regularized Bellman equation. AlphaZero section 4 must diagram the MCTS + neural network interaction clearly (4 phases: selection, expansion, evaluation, backup). Dreamer section 4 must show RSSM: recurrent state + stochastic state. Decision Transformer section 3 must explain why framing RL as sequence prediction is powerful.

---

## BATCH REC-1 — Recommendation Systems

**Create these files:**
- `src/content/architectures/matrix-factorization/content.mdx`
- `src/content/architectures/wide-and-deep/content.mdx`
- `src/content/architectures/deepfm/content.mdx`
- `src/content/architectures/ncf/content.mdx`
- `src/content/architectures/two-tower-model/content.mdx`
- `src/content/architectures/dlrm/content.mdx`
- `src/content/architectures/din/content.mdx`
- `src/content/architectures/sasrec/content.mdx`
- `src/content/architectures/bert4rec/content.mdx`
- `src/content/architectures/pinsage/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| matrix-factorization | Matrix Factorization | 2009 | Koren et al. | Decompose R ≈ PQ^T: latent factor model; SGD or ALS optimization; Netflix Prize winner | Beginner |
| wide-and-deep | Wide & Deep | 2016 | Cheng et al. | Wide (memorization via cross-product features) + Deep (generalization via MLP) jointly trained — Google Play | Intermediate |
| deepfm | DeepFM | 2017 | Guo et al. | FM layer replaces wide part — no manual feature engineering; share embedding layer between FM and Deep | Intermediate |
| ncf | NCF | 2017 | He et al. | Neural Collaborative Filtering: replace dot product with MLP — more expressive user-item interaction | Intermediate |
| two-tower-model | Two-Tower Model | 2019 | Yi et al. | Separate user tower + item tower → dot product; precompute item embeddings for ANN retrieval — scales to billions | Intermediate |
| dlrm | DLRM | 2019 | Naumov et al. | Sparse embeddings for categorical + dense MLP for numerical → dot product interactions → MLP — Facebook production | Advanced |
| din | DIN | 2018 | Zhou et al. | Attention over user behavior history weighted by candidate item similarity — activates relevant historical interests | Advanced |
| sasrec | SASRec | 2018 | Kang, McAuley | Unidirectional transformer on user's item sequence — self-attention over interaction history | Intermediate |
| bert4rec | BERT4Rec | 2019 | Sun et al. | Bidirectional transformer with Cloze task on user behavior sequence — random mask and predict | Intermediate |
| pinsage | PinSage | 2018 | Ying et al. | GraphSAGE on billion-node graph: random walks for neighborhood sampling; importance-based sampling | Advanced |

**Context:** Matrix Factorization section 7 must show SVD, ALS, and SGD formulations. Two-Tower section 4 must diagram the retrieval (ANN) + ranking (cross-features) two-stage pipeline. DIN section 7 must show the attention weight formula: $e_i = \text{Attention}(v_A, v_{b_i})$ where $v_A$ is candidate, $v_{b_i}$ is behavior item.

---

## BATCH GNN-1 — Graph Neural Networks

**Create these files:**
- `src/content/architectures/deepwalk/content.mdx`
- `src/content/architectures/node2vec/content.mdx`
- `src/content/architectures/gcn/content.mdx`
- `src/content/architectures/graphsage/content.mdx`
- `src/content/architectures/gat/content.mdx`
- `src/content/architectures/gatv2/content.mdx`
- `src/content/architectures/gin/content.mdx`
- `src/content/architectures/mpnn/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| deepwalk | DeepWalk | 2014 | Perozzi et al. | Random walks on graph → Word2Vec on walk sequences — learn node embeddings from structural context | Beginner |
| node2vec | Node2Vec | 2016 | Grover, Leskovec | Biased random walks: parameter p controls BFS (homophily) vs DFS (structural equivalence) balance | Beginner |
| gcn | GCN | 2016 | Kipf, Welling | Spectral graph conv simplified to $H^{(l+1)} = \sigma(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2} H^{(l)} W^{(l)})$ — message passing | Intermediate |
| graphsage | GraphSAGE | 2017 | Hamilton et al. | Inductive: sample fixed-size neighborhood, aggregate (mean/LSTM/pool) — generalizes to unseen nodes | Intermediate |
| gat | GAT | 2017 | Veličković et al. | Attention-weighted aggregation: $\alpha_{ij} = \text{softmax}(\text{LeakyReLU}(a^T[Wh_i \| Wh_j]))$ | Intermediate |
| gatv2 | GATv2 | 2021 | Brody et al. | Dynamic attention: apply $W$ before concat, not after — fixes static attention limitation in GAT | Intermediate |
| gin | GIN | 2018 | Xu et al. | Graph Isomorphism Network: $h_v^{(k)} = \text{MLP}^{(k)}((1+\epsilon^{(k)}) h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)})$ — maximally expressive | Advanced |
| mpnn | MPNN | 2017 | Gilmer et al. | Unifying message passing framework: message function $M_t$, update function $U_t$, readout function $R$ — molecular graphs | Intermediate |

**Context:** GCN section 7 must derive the spectral convolution from the graph Laplacian to the simplified formula step by step. GIN section 7 must explain the Weisfeiler-Leman graph isomorphism test and why sum aggregation is more powerful than mean. GAT section 4 must diagram multi-head attention on a small example graph.

---

## BATCH GNN-2 — Advanced Graphs

**Create these files:**
- `src/content/architectures/r-gcn/content.mdx`
- `src/content/architectures/transe/content.mdx`
- `src/content/architectures/diffpool/content.mdx`
- `src/content/architectures/graph-transformer/content.mdx`
- `src/content/architectures/graphormer/content.mdx`
- `src/content/architectures/graphgps/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| r-gcn | R-GCN | 2017 | Schlichtkrull et al. | Relation-specific weight matrices: $h_i^{(l+1)} = \sigma(\sum_r \sum_{j \in \mathcal{N}_r(i)} \frac{1}{c_{i,r}} W_r^{(l)} h_j^{(l)})$ | Intermediate |
| transe | TransE | 2013 | Bordes et al. | Knowledge graph embedding: $h + r \approx t$ in vector space; minimize $\|h+r-t\|$ for true triples | Beginner |
| diffpool | DiffPool | 2018 | Ying et al. | Differentiable pooling: soft cluster assignment matrix $S$ learned jointly with GNN | Advanced |
| graph-transformer | Graph Transformer | 2020 | Dwivedi, Bresson | Full attention on graph with Laplacian PE as positional encoding — generalizes ViT to graphs | Advanced |
| graphormer | Graphormer | 2021 | Ying et al. | Centrality encoding (in+out degree), spatial encoding (shortest path distance), edge encoding in attention bias | Advanced |
| graphgps | GraphGPS | 2022 | Rampášek et al. | General, Powerful, Scalable: interleave local MPNN + global attention + PE at each layer | Expert |

**Context:** TransE section 7 must derive the scoring function and loss (margin-based). Graphormer section 7 must show how the spatial encoding $\phi(v_i, v_j)$ enters the attention: $A_{ij} = \frac{(h_i W_Q)(h_j W_K)^T}{\sqrt{d}} + b_{\phi(v_i,v_j)}$. GraphGPS section 3 must explain why neither pure MPNN nor pure transformer alone is optimal.

---

## BATCH MM-1 — Multimodal Architectures

**Create these files:**
- `src/content/architectures/dall-e/content.mdx`
- `src/content/architectures/flamingo/content.mdx`
- `src/content/architectures/blip/content.mdx`
- `src/content/architectures/blip-2/content.mdx`
- `src/content/architectures/instructblip/content.mdx`
- `src/content/architectures/llava/content.mdx`
- `src/content/architectures/llava-1-5/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| dall-e | DALL-E | 2021 | Ramesh et al. | dVAE tokenizes image into 32×32 discrete tokens; transformer autoregressively models (text, image) token sequence — zero-shot text-to-image | Advanced |
| flamingo | Flamingo | 2022 | Alayrac et al. | Perceiver Resampler compresses N image patches to fixed 64 tokens; cross-attention layers inserted into frozen Chinchilla | Advanced |
| blip | BLIP | 2022 | Li et al. | CapFilt: train captioner and filter on noisy web data; three objectives: ITC + ITM + LM — flexible for understanding and generation | Intermediate |
| blip-2 | BLIP-2 | 2023 | Li et al. | Q-Former: 32 learned query tokens bridge frozen image encoder and frozen LLM via cross-attention — alignment without full finetuning | Advanced |
| instructblip | InstructBLIP | 2023 | Dai et al. | Instruction-aware Q-Former: instruction text also fed to Q-Former, not just image — task-aware visual feature extraction | Advanced |
| llava | LLaVA | 2023 | Liu et al. | Linear projection from CLIP visual features to LLaMA token space; GPT-4 generated visual instruction tuning data | Advanced |
| llava-1-5 | LLaVA 1.5 | 2023 | Liu et al. | MLP connector (2-layer) replaces linear; CLIP-ViT-L-336px; better instruction data; 10× more compute-efficient | Advanced |

**Context:** BLIP-2 section 4 must diagram Q-Former clearly — the 32 query tokens attend to image via cross-attention and to each other via self-attention. Flamingo section 4 must show Perceiver Resampler architecture. LLaVA section 3 must explain why the connector (linear vs MLP) matters for visual understanding.

---

## BATCH MM-2 — Advanced Multimodal

**Create these files:**
- `src/content/architectures/minigpt-4/content.mdx`
- `src/content/architectures/gpt-4v/content.mdx`
- `src/content/architectures/gemini/content.mdx`
- `src/content/architectures/imagebind/content.mdx`
- `src/content/architectures/whisper/content.mdx`
- `src/content/architectures/seamlessm4t/content.mdx`
- `src/content/architectures/dall-e-3/content.mdx`
- `src/content/architectures/cogvlm/content.mdx`

**Architecture metadata:**

| Slug | Name | Year | Authors | Key Innovation | Difficulty |
|------|------|------|---------|----------------|------------|
| minigpt-4 | MiniGPT-4 | 2023 | Zhu et al. | Single linear layer aligning BLIP-2 visual encoder with Vicuna — minimal connector, strong results | Advanced |
| gpt-4v | GPT-4V | 2023 | OpenAI | Native vision inputs in GPT-4; image as additional input tokens via visual encoder; strong spatial reasoning | Expert |
| gemini | Gemini | 2023 | Google DeepMind | Natively multimodal from pre-training on interleaved text+image+audio+video data — not retrofitted | Expert |
| imagebind | ImageBind | 2023 | Girdhar et al. | Single shared embedding for 6 modalities: image, text, audio, depth, thermal, IMU — bind without all pairs | Expert |
| whisper | Whisper | 2022 | Radford et al. | Encoder-decoder transformer on log-mel spectrogram; 680K hours weakly supervised; multitask: transcribe/translate/detect | Intermediate |
| seamlessm4t | SeamlessM4T | 2023 | Barrault et al. | Unified speech+text translation for 100+ languages in single model: S2ST, S2TT, T2ST, T2TT, ASR | Advanced |
| dall-e-3 | DALL-E 3 | 2023 | Betker et al. | GPT-4 as image recaptioner: generate highly detailed synthetic captions for training — dramatically improves prompt following | Advanced |
| cogvlm | CogVLM | 2023 | Wang et al. | Visual expert module: separate trainable QKV+FFN per transformer layer for visual tokens — no visual-linguistic conflict | Advanced |

**Context:** ImageBind section 7 must explain why image is used as the binding modality (image-audio pairs, image-depth pairs available even without audio-depth pairs). Whisper section 6 must show the log-mel spectrogram computation and how it maps to encoder input. Gemini section 3 must contrast with retrofitted multimodal (BLIP-2, LLaVA) to explain the advantage of native multimodal training.

---

## QUALITY CHECKLIST (Antigravity must verify before finishing each file)

- [ ] Section 4 has a real ASCII/text diagram of the architecture (not just text description)
- [ ] Section 6 shows tensor shapes at every major step using `Input → Layer → Output: (B, C, H, W)` format
- [ ] Section 7 has at least 2 equations in KaTeX with all symbols defined
- [ ] Section 9 has a PyTorch class with `forward()` and a `if __name__ == '__main__':` test block
- [ ] Section 13 has exactly 5 Q&A pairs with full multi-sentence answers
- [ ] No section is a stub — every section has substantive content (150+ words)
- [ ] Parent architecture is mentioned in Section 2 with context of what it improved upon
- [ ] Related architectures are mentioned in Section 12 (what this led to)
