"""
core/implementation/reproduction_cards.py

Phase 10: Research reproduction summary cards per architecture.

Each card provides:
  - Paper summary
  - Architecture overview
  - Training configuration
  - Expected results
  - Known limitations
  - Common failure modes
"""

from typing import Any

REPRODUCTION_CARDS: dict[str, dict[str, Any]] = {
    "ResNet": {
        "classification": "ResNet",
        "paper": "Deep Residual Learning for Image Recognition",
        "authors": "He, Zhang, Ren, Sun (Microsoft Research)",
        "year": 2016,
        "venue": "CVPR 2016 (Best Paper Award)",
        "paper_summary": (
            "ResNet introduced skip connections (residual shortcuts) that bypass one or more "
            "convolutional layers. This addressed the 'degradation problem' where adding more "
            "layers to a plain network unexpectedly reduced training accuracy. With residual "
            "learning, networks with 152 layers were trained successfully — previously impossible."
        ),
        "architecture": {
            "key_insight": "Learning residual functions F(x) = H(x) - x is easier than learning H(x) directly",
            "building_block": "3×3 Conv → BN → ReLU → 3×3 Conv → BN → (+x) → ReLU",
            "variants": [
                "ResNet-18",
                "ResNet-34",
                "ResNet-50 (bottleneck)",
                "ResNet-101",
                "ResNet-152",
            ],
            "parameter_counts": {"ResNet-18": "11.7M", "ResNet-50": "25.6M", "ResNet-101": "44.5M"},
        },
        "training_config": {
            "dataset": "ImageNet ILSVRC 2012 (1.28M training images)",
            "optimizer": "SGD, momentum=0.9, weight_decay=1e-4",
            "lr_schedule": "Start at 0.1, divide by 10 at epochs 30, 60, 90",
            "batch_size": "256 (across 8 GPUs for ImageNet)",
            "epochs": 90,
            "data_augmentation": "Random crop 224×224, horizontal flip, color jitter",
        },
        "expected_results": {
            "ResNet-50 Top-1 ImageNet": "76.1%",
            "ResNet-50 Top-5 ImageNet": "92.9%",
            "ResNet-101 Top-1 ImageNet": "77.4%",
            "CIFAR-10 ResNet-110": "6.43% error",
        },
        "known_limitations": [
            "Still uses a large number of parameters compared to efficient architectures (MobileNet, EfficientNet)",
            "Fixed receptive field per layer — cannot dynamically attend to long-range dependencies",
            "Requires careful LR scheduling; large LR at start can cause NaN loss",
            "Plain ResNet underperforms ViT on very large datasets (>100M images)",
        ],
        "common_failure_modes": [
            {
                "symptom": "NaN loss in early training",
                "cause": "Learning rate too high (0.1 can be unstable without batch=256)",
                "fix": "Use warmup for first 5 epochs, or reduce initial LR to 0.01",
            },
            {
                "symptom": "Accuracy plateaus at ~60% on ImageNet",
                "cause": "Missing weight decay or incorrect data augmentation pipeline",
                "fix": "Verify weight_decay=1e-4 and ensure RandomResizedCrop is applied",
            },
            {
                "symptom": "Skip connection shape mismatch RuntimeError",
                "cause": "Stride=2 block without downsample projection shortcut",
                "fix": "Add 1×1 Conv projection when stride≠1 or channels change",
            },
        ],
        "reproduction_difficulty": "Easy",
        "reproduction_notes": "Well-reproduced. torchvision provides official pretrained weights.",
    },
    "Transformer": {
        "classification": "Transformer",
        "paper": "Attention Is All You Need",
        "authors": "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (Google Brain)",
        "year": 2017,
        "venue": "NeurIPS 2017",
        "paper_summary": (
            "The Transformer replaced recurrent networks (LSTMs) with pure attention mechanisms "
            "for sequence-to-sequence tasks. The key insight: attention can model global "
            "dependencies between any two positions in one step, unlike RNNs which process "
            "sequentially. This enabled massive parallelism during training and became the "
            "foundation for BERT, GPT, T5, and all modern language models."
        ),
        "architecture": {
            "key_insight": "Multi-head attention can replace sequential processing — no recurrence needed",
            "building_block": "LayerNorm → MHSA → Residual → LayerNorm → FFN → Residual",
            "variants": ["Transformer Base (512d, 6L, 8H)", "Transformer Large (1024d, 6L, 16H)"],
            "parameter_counts": {"Transformer Base": "65M", "Transformer Large": "213M"},
        },
        "training_config": {
            "dataset": "WMT 2014 English-German (4.5M sentence pairs)",
            "optimizer": "Adam, β₁=0.9, β₂=0.98, ε=1e-9",
            "lr_schedule": "Warmup 4000 steps, then inverse square root decay",
            "batch_size": "25000 tokens (not samples — variable length batching)",
            "epochs": "300K steps (≈100 epochs equivalent)",
            "data_augmentation": "Label smoothing ε=0.1, residual dropout=0.1",
        },
        "expected_results": {
            "WMT EN-DE BLEU (Base)": "27.3",
            "WMT EN-FR BLEU (Big)": "41.0",
            "Training time (Base, 8×P100)": "~12 hours",
        },
        "known_limitations": [
            "Quadratic attention complexity O(n²) — slow on very long sequences (>2048 tokens)",
            "Requires large datasets — doesn't generalize well from scratch on small datasets",
            "Position encoding is fixed sinusoidal in original paper — lacks inductive spatial bias",
            "Memory-intensive: attention matrices for long sequences require significant GPU RAM",
        ],
        "common_failure_modes": [
            {
                "symptom": "Loss NaN or divergence in first 100 steps",
                "cause": "Missing LR warmup — Adam with high LR and no warmup is unstable",
                "fix": "Implement the paper's warmup schedule exactly: lrate = d_model^(-0.5) * min(step^(-0.5), step × warmup^(-1.5))",
            },
            {
                "symptom": "BLEU score much lower than paper (~20 vs 27)",
                "cause": "Missing label smoothing or incorrect tokenization (not BPE)",
                "fix": "Use sentencepiece BPE tokenizer, add label_smoothing=0.1 to CrossEntropyLoss",
            },
            {
                "symptom": "OOM on long sequences",
                "cause": "Attention matrix is O(n²·d) — 2048 tokens × 768d saturates 16GB GPU",
                "fix": "Use gradient checkpointing, reduce batch size, or use FlashAttention",
            },
        ],
        "reproduction_difficulty": "Medium",
        "reproduction_notes": "Warmup schedule is critical and often missed. Use HuggingFace transformers for reference.",
    },
    "ViT": {
        "classification": "ViT",
        "paper": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "authors": "Dosovitskiy, Beyer, Kolesnikov et al. (Google Brain)",
        "year": 2020,
        "venue": "ICLR 2021",
        "paper_summary": (
            "ViT applied the pure Transformer architecture directly to image classification "
            "by treating images as sequences of 16×16 patches. Critically, ViT requires "
            "pre-training on very large datasets (JFT-300M) to outperform CNNs — it lacks "
            "the inductive biases (translation equivariance, locality) built into CNNs. "
            "When trained only on ImageNet, ViT-B underperforms ResNet-50."
        ),
        "architecture": {
            "key_insight": "Images as token sequences: split into P×P patches, project to embeddings, apply Transformer",
            "building_block": "PatchEmbed → [CLS] concat → +PosEmbed → N×TransformerBlock → MLP Head",
            "variants": ["ViT-S/16 (22M)", "ViT-B/16 (86M)", "ViT-L/16 (307M)", "ViT-H/14 (632M)"],
            "parameter_counts": {"ViT-B/16": "86.6M", "ViT-L/16": "307M", "ViT-H/14": "632M"},
        },
        "training_config": {
            "dataset": "JFT-300M (Google internal) for pretraining, ImageNet for fine-tuning",
            "optimizer": "Adam, β₁=0.9, β₂=0.999, weight_decay=0.1",
            "lr_schedule": "Linear warmup + cosine decay",
            "batch_size": "4096 (across many TPUs)",
            "epochs": "300 epochs ImageNet / 14 epochs JFT",
            "data_augmentation": "RandAugment, Mixup, CutMix, RandomErasing",
        },
        "expected_results": {
            "ViT-B/16 ImageNet Top-1 (ImageNet only)": "77.9%",
            "ViT-B/16 ImageNet Top-1 (JFT pretrain)": "84.6%",
            "ViT-L/16 ImageNet Top-1 (JFT pretrain)": "87.1%",
        },
        "known_limitations": [
            "Requires very large pretraining data — fails without JFT or large ImageNet-21K",
            "No inductive bias: must learn locality and translation equivariance from data",
            "Computationally expensive: O(n²) attention over 196 patches (14×14 grid for 224px)",
            "Fixed resolution at training time — requires interpolation for different resolutions",
        ],
        "common_failure_modes": [
            {
                "symptom": "ViT trained from scratch on ImageNet achieves only ~70% (vs 77%)",
                "cause": "Insufficient data augmentation or small batch size",
                "fix": "Use DeiT training recipe: Mixup+CutMix+RandAugment, batch=1024, label_smoothing=0.1",
            },
            {
                "symptom": "Patch embedding dimension mismatch",
                "cause": "Input image size not divisible by patch_size",
                "fix": "Ensure img_size % patch_size == 0; use img_size=224, patch_size=16",
            },
            {
                "symptom": "Loss does not decrease after warmup",
                "cause": "Very high weight_decay (0.3) may be too aggressive for your dataset",
                "fix": "Reduce weight_decay to 0.05-0.1, check learning rate scale with batch size",
            },
        ],
        "reproduction_difficulty": "Hard (requires large pretraining or DeiT recipe)",
        "reproduction_notes": "Use DeiT (Data-efficient Image Transformer) for ImageNet-only reproduction.",
    },
    "Encoder-Decoder": {
        "classification": "Encoder-Decoder",
        "paper": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
        "authors": "Ronneberger, Fischer, Brox (University of Freiburg)",
        "year": 2015,
        "venue": "MICCAI 2015",
        "paper_summary": (
            "U-Net introduced a fully convolutional encoder-decoder architecture with "
            "skip connections directly concatenating encoder feature maps to corresponding "
            "decoder layers. Originally designed for biomedical segmentation with limited "
            "labeled data, it became the dominant architecture for semantic segmentation "
            "and is widely used in medical imaging, satellite imagery, and generative models."
        ),
        "architecture": {
            "key_insight": "Skip connections concatenate high-resolution encoder features directly to decoder — preserving spatial detail",
            "building_block": "Encoder: DoubleConv→MaxPool; Decoder: Upsample+Concat(skip)→DoubleConv",
            "variants": ["U-Net (original)", "U-Net++", "Attention U-Net", "3D U-Net"],
            "parameter_counts": {"U-Net original": "31M"},
        },
        "training_config": {
            "dataset": "ISBI 2012 EM segmentation challenge (30 training images — very small!)",
            "optimizer": "SGD, momentum=0.99",
            "lr_schedule": "Fixed LR=0.01 in original; ReduceLROnPlateau used in practice",
            "batch_size": "1 (original uses per-pixel loss over full image — high-res)",
            "epochs": "Not specified in paper; typically 50-200 for medical datasets",
            "data_augmentation": "Elastic deformations, random rotation, random flip — critical for limited data",
        },
        "expected_results": {
            "ISBI EM Cell Segmentation": "IoU > 0.92 (rand error)",
            "ISBI Cell Tracking Challenge": "1st place at time of publication",
            "General Segmentation (PASCAL VOC)": "Not benchmarked — domain-specific architecture",
        },
        "known_limitations": [
            "Batch size of 1 makes batch normalization unstable — use Instance Norm or Group Norm instead",
            "Skip connection feature map sizes must match exactly — padding strategy critical",
            "Memory scales quadratically with input resolution — 512×512 requires ~8GB VRAM",
            "Original architecture has no attention — later variants add attention gates",
        ],
        "common_failure_modes": [
            {
                "symptom": "Skip connection concat fails with shape mismatch",
                "cause": "Encoder output spatial size ≠ decoder upsample output size",
                "fix": "Use CenterCrop or pad to align; or use 'same' padding in all convolutions",
            },
            {
                "symptom": "Model learns to output all zeros (mode collapse for segmentation)",
                "cause": "Severe class imbalance — background >> foreground pixels",
                "fix": "Add Dice Loss or Focal Loss; use pos_weight in BCEWithLogitsLoss",
            },
            {
                "symptom": "Training with batch=1 causes unstable BN",
                "cause": "BatchNorm statistics with B=1 are meaningless",
                "fix": "Replace BatchNorm2d with nn.GroupNorm(8, channels) or nn.InstanceNorm2d",
            },
        ],
        "reproduction_difficulty": "Easy (biomedical) / Medium (natural image segmentation)",
        "reproduction_notes": "For natural images, use DeepLab or SegFormer. U-Net shines in medical imaging.",
    },
    "DenseNet": {
        "classification": "DenseNet",
        "paper": "Densely Connected Convolutional Networks",
        "authors": "Huang, Liu, van der Maaten, Weinberger",
        "year": 2017,
        "venue": "CVPR 2017 (Best Paper Award)",
        "paper_summary": (
            "DenseNet connects every layer to every other layer in a feed-forward fashion. "
            "Each layer receives the feature maps of all preceding layers as input and passes "
            "its own feature maps to all subsequent layers. This dense connectivity enables "
            "maximum feature reuse, reduces the number of parameters, and strengthens gradient "
            "flow throughout the network."
        ),
        "architecture": {
            "key_insight": "Concatenate (not add) feature maps from all previous layers — maximum feature reuse",
            "building_block": "DenseBlock: BN+ReLU+Conv(growth_rate) × L layers; TransitionLayer: BN+1×1Conv+AvgPool",
            "variants": [
                "DenseNet-121 (8M)",
                "DenseNet-169 (14M)",
                "DenseNet-201 (20M)",
                "DenseNet-264 (34M)",
            ],
            "parameter_counts": {"DenseNet-121": "8M", "DenseNet-201": "20M"},
        },
        "training_config": {
            "dataset": "ImageNet ILSVRC 2012 / CIFAR-10/100",
            "optimizer": "SGD, momentum=0.9, weight_decay=1e-4, nesterov=True",
            "lr_schedule": "Cosine annealing over 300 epochs (CIFAR) / Step decay (ImageNet)",
            "batch_size": "64 (CIFAR) / 256 (ImageNet)",
            "epochs": "300 (CIFAR) / 90 (ImageNet)",
            "data_augmentation": "Random crop, horizontal flip, normalization",
        },
        "expected_results": {
            "DenseNet-121 ImageNet Top-1": "74.7%",
            "DenseNet-201 ImageNet Top-1": "77.2%",
            "DenseNet-100 (k=12) CIFAR-10": "4.10% error",
            "DenseNet-100 (k=12) CIFAR-100": "20.20% error",
        },
        "known_limitations": [
            "Memory hungry during training — concatenation increases feature map count quadratically within blocks",
            "Slower than ResNet for same parameter count due to memory copies in concatenation",
            "Growth rate must be tuned — too high causes memory OOM; too low reduces capacity",
            "TransitionLayer compression ratio (θ=0.5) is a critical hyperparameter",
        ],
        "common_failure_modes": [
            {
                "symptom": "CUDA out of memory during forward pass",
                "cause": "Dense connections accumulate all feature maps — memory grows as O(L² × growth_rate)",
                "fix": "Use DenseNet memory-efficient implementation (on-the-fly recomputation); reduce growth_rate",
            },
            {
                "symptom": "Lower accuracy than expected (~72% vs 74.7%)",
                "cause": "Missing transition layer compression (should reduce channels by θ=0.5)",
                "fix": "Ensure transition layers have Conv2d(in, in*0.5) to compress feature count",
            },
        ],
        "reproduction_difficulty": "Medium",
        "reproduction_notes": "Use memory-efficient DenseNet implementation from gpleiss/efficient_densenet_pytorch.",
    },
}


def get_reproduction_card(classification: str) -> dict[str, Any]:
    """
    Return the research reproduction card for a given architecture classification.
    Falls back to ResNet if classification not found.
    """
    card = REPRODUCTION_CARDS.get(classification)
    if not card:
        # Fuzzy match
        classification_lower = classification.lower()
        for key in REPRODUCTION_CARDS:
            if key.lower() in classification_lower or classification_lower in key.lower():
                card = REPRODUCTION_CARDS[key]
                break

    return card or REPRODUCTION_CARDS["ResNet"]
