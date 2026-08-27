import fs from "node:fs";
import path from "node:path";

const aliases = {
  alexnet: "imagenet-classification-with-deep-convolutional-neural-networks-alexnet",
  "bahdanau-attention": "neural-machine-translation-by-jointly-learning-to-align-and-translate",
  "batch-normalization": "batch-normalization-accelerating-deep-network-training",
  bert: "bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding",
  chinchilla: "training-compute-optimal-large-language-models-chinchilla",
  clip: "learning-transferable-visual-models-from-natural-language-supervision-clip",
  "deep-residual-learning": "deep-residual-learning-for-image-recognition",
  dpo: "direct-preference-optimization-dpo",
  "flash-attention": "flashattention-fast-and-memory-efficient-exact-attention-with-io-awareness",
  gan: "generative-adversarial-nets",
  gpt: "improving-language-understanding-by-generative-pre-training-gpt-1",
  "gpt-2": "language-models-are-unsupervised-multitask-learners-gpt-2",
  "gpt-3": "language-models-are-few-shot-learners-gpt-3",
  instructgpt: "training-language-models-to-follow-instructions-with-human-feedback-instructgpt",
  "latent-diffusion-models": "high-resolution-image-synthesis-with-latent-diffusion-models-stable-diffusion",
  llama: "llama-open-and-efficient-foundation-language-models",
  lora: "lora-low-rank-adaptation-of-large-language-models",
  palm: "palm-scaling-language-modeling-with-pathways",
  "segment-anything": "segment-anything-sam",
  "stable-diffusion": "high-resolution-image-synthesis-with-latent-diffusion-models-stable-diffusion",
  "switch-transformer": "switch-transformers-scaling-to-trillion-parameter-models-with-simple-and-efficient-sparsity",
  vgg: "very-deep-convolutional-networks-for-large-scale-image-recognition-vggnet",
  "vision-transformer": "an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale-vit",
};

const root = path.join(process.cwd(), "src/content/papers");
let copiedArticles = 0;
let copiedMetadata = 0;
let existingTargets = 0;

for (const [sourceSlug, targetSlug] of Object.entries(aliases)) {
  const sourceDirectory = path.join(root, sourceSlug);
  const targetDirectory = path.join(root, targetSlug);
  const sourceArticle = path.join(sourceDirectory, "content.mdx");
  const targetArticle = path.join(targetDirectory, "content.mdx");
  const sourceMeta = path.join(sourceDirectory, "meta.json");
  const targetMeta = path.join(targetDirectory, "meta.json");

  if (!fs.existsSync(sourceArticle)) {
    throw new Error(`Missing reusable paper source: ${sourceSlug}`);
  }

  fs.mkdirSync(targetDirectory, { recursive: true });
  if (!fs.existsSync(targetArticle)) {
    fs.copyFileSync(sourceArticle, targetArticle);
    copiedArticles += 1;
  } else {
    existingTargets += 1;
  }

  if (fs.existsSync(sourceMeta) && !fs.existsSync(targetMeta)) {
    const metadata = JSON.parse(fs.readFileSync(sourceMeta, "utf8"));
    metadata.slug = targetSlug;
    fs.writeFileSync(targetMeta, `${JSON.stringify(metadata, null, 2)}\n`);
    copiedMetadata += 1;
  }
}

console.log(`Canonical paper articles copied: ${copiedArticles}`);
console.log(`Canonical paper metadata copied: ${copiedMetadata}`);
console.log(`Already-existing canonical targets: ${existingTargets}`);
