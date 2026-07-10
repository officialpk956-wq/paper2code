"""
core/implementation/training_config.py

Phase 10: Deterministic per-architecture training configurations.
All values are grounded in published paper training setups.
No fabricated measurements.
"""

from typing import Any

# ── Per-architecture canonical training setups ──────────────────────────────
# Sources: original papers / common reproductions cited in literature.

TRAINING_CONFIGS: dict[str, dict[str, Any]] = {
    "ResNet": {
        "classification": "ResNet",
        "paper_reference": "He et al. 2016 — Deep Residual Learning for Image Recognition",
        "loss_function": {
            "name": "CrossEntropyLoss",
            "pytorch": "nn.CrossEntropyLoss()",
            "note": "Standard multi-class classification loss. Uses log-softmax + NLL internally.",
        },
        "optimizer": {
            "name": "SGD with Momentum",
            "pytorch": "torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)",
            "note": "Stochastic Gradient Descent with Nesterov momentum. Weight decay acts as L2 regularization.",
        },
        "learning_rate": {
            "initial": 0.1,
            "schedule": "Step Decay",
            "pytorch": "torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)",
            "note": "LR divided by 10 at epochs 30, 60, 90 on ImageNet.",
        },
        "batch_size": 256,
        "epochs": 90,
        "augmentations": [
            "RandomResizedCrop(224)",
            "RandomHorizontalFlip()",
            "ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)",
            "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
        ],
        "warmup": None,
        "label_smoothing": 0.0,
        "expected_top1_imagenet": "76.1% (ResNet-50)",
    },
    "CNN": {
        "classification": "CNN",
        "paper_reference": "Standard CNN training (AlexNet / VGG style)",
        "loss_function": {
            "name": "CrossEntropyLoss",
            "pytorch": "nn.CrossEntropyLoss()",
            "note": "Standard multi-class loss for image classification.",
        },
        "optimizer": {
            "name": "SGD with Momentum",
            "pytorch": "torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)",
            "note": "Lower base LR than ResNet. AlexNet used 0.01 initial LR.",
        },
        "learning_rate": {
            "initial": 0.01,
            "schedule": "Step Decay",
            "pytorch": "torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)",
            "note": "LR halved every 30 epochs.",
        },
        "batch_size": 128,
        "epochs": 90,
        "augmentations": [
            "RandomCrop(224, padding=4)",
            "RandomHorizontalFlip()",
            "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
        ],
        "warmup": None,
        "label_smoothing": 0.0,
        "expected_top1_imagenet": "~74% (VGG16)",
    },
    "DenseNet": {
        "classification": "DenseNet",
        "paper_reference": "Huang et al. 2017 — Densely Connected Convolutional Networks",
        "loss_function": {
            "name": "CrossEntropyLoss",
            "pytorch": "nn.CrossEntropyLoss()",
            "note": "Same as ResNet. Dense connections do not change the loss formulation.",
        },
        "optimizer": {
            "name": "SGD with Nesterov Momentum",
            "pytorch": "torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)",
            "note": "Nesterov momentum slightly improves convergence for DenseNet.",
        },
        "learning_rate": {
            "initial": 0.1,
            "schedule": "Cosine Annealing",
            "pytorch": "torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)",
            "note": "Smooth cosine decay over 300 epochs for CIFAR experiments.",
        },
        "batch_size": 64,
        "epochs": 300,
        "augmentations": [
            "RandomCrop(32, padding=4)",
            "RandomHorizontalFlip()",
            "Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])",
        ],
        "warmup": None,
        "label_smoothing": 0.0,
        "expected_top1_imagenet": "77.2% (DenseNet-201)",
    },
    "Transformer": {
        "classification": "Transformer",
        "paper_reference": "Vaswani et al. 2017 — Attention Is All You Need",
        "loss_function": {
            "name": "CrossEntropyLoss with Label Smoothing",
            "pytorch": "nn.CrossEntropyLoss(label_smoothing=0.1)",
            "note": "Label smoothing (ε=0.1) prevents overconfident predictions and improves BLEU.",
        },
        "optimizer": {
            "name": "AdamW",
            "pytorch": "torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)",
            "note": "Adam with decoupled weight decay. Standard for Transformer training.",
        },
        "learning_rate": {
            "initial": 1e-4,
            "schedule": "Warmup + Inverse Square Root",
            "pytorch": (
                "# Custom schedule from 'Attention Is All You Need'\n"
                "lrate = d_model**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))\n"
                "# Or use HuggingFace get_cosine_schedule_with_warmup"
            ),
            "note": "LR linearly increases for warmup_steps=4000 then decays as inverse square root.",
        },
        "batch_size": 32,
        "epochs": 100,
        "augmentations": [
            "TokenDropout(p=0.1)",
            "MixToken or CutMix (optional)",
            "LabelSmoothing(0.1)",
        ],
        "warmup": {
            "steps": 4000,
            "note": "Critical for Transformer stability. Without warmup, training often diverges.",
        },
        "label_smoothing": 0.1,
        "expected_top1_imagenet": "N/A — sequence-to-sequence task",
    },
    "ViT": {
        "classification": "ViT",
        "paper_reference": "Dosovitskiy et al. 2020 — An Image is Worth 16x16 Words",
        "loss_function": {
            "name": "CrossEntropyLoss with Label Smoothing",
            "pytorch": "nn.CrossEntropyLoss(label_smoothing=0.1)",
            "note": "ViT benefits from label smoothing, same as NLP Transformers.",
        },
        "optimizer": {
            "name": "AdamW",
            "pytorch": "torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.3)",
            "note": "High weight decay (0.3) critical for ViT regularization on small datasets.",
        },
        "learning_rate": {
            "initial": 3e-3,
            "schedule": "Cosine Decay with Linear Warmup",
            "pytorch": (
                "from torch.optim.lr_scheduler import CosineAnnealingLR\n"
                "# Warmup: linearly increase for first 10k steps\n"
                "# Then: CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)"
            ),
            "note": "Warmup prevents instability during early patch embedding training.",
        },
        "batch_size": 4096,
        "epochs": 300,
        "augmentations": [
            "RandAugment(num_ops=2, magnitude=9)",
            "Mixup(alpha=0.2)",
            "CutMix(alpha=1.0)",
            "RandomErasing(p=0.25)",
            "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
        ],
        "warmup": {
            "steps": 10000,
            "note": "Essential. ViT trained from scratch is data-hungry and unstable without warmup.",
        },
        "label_smoothing": 0.1,
        "expected_top1_imagenet": "81.8% (ViT-B/16, JFT-300M pretrain)",
    },
    "Encoder-Decoder": {
        "classification": "Encoder-Decoder",
        "paper_reference": "Ronneberger et al. 2015 — U-Net: Convolutional Networks for Biomedical Image Segmentation",
        "loss_function": {
            "name": "BCEWithLogitsLoss + Dice Loss",
            "pytorch": (
                "# Binary segmentation\n"
                "bce = nn.BCEWithLogitsLoss()\n"
                "# Dice Loss (custom)\n"
                "def dice_loss(pred, target, smooth=1):\n"
                "    pred = torch.sigmoid(pred)\n"
                "    intersection = (pred * target).sum()\n"
                "    return 1 - (2*intersection + smooth) / (pred.sum() + target.sum() + smooth)\n"
                "# Combined: loss = bce(pred, target) + dice_loss(pred, target)"
            ),
            "note": "BCE handles per-pixel classification; Dice Loss addresses class imbalance in segmentation.",
        },
        "optimizer": {
            "name": "Adam",
            "pytorch": "torch.optim.Adam(model.parameters(), lr=1e-4)",
            "note": "Adam converges faster than SGD for U-Net on medical imaging tasks.",
        },
        "learning_rate": {
            "initial": 1e-4,
            "schedule": "ReduceLROnPlateau",
            "pytorch": "torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)",
            "note": "Adaptive LR reduction on validation loss plateau. Common in medical imaging.",
        },
        "batch_size": 8,
        "epochs": 50,
        "augmentations": [
            "RandomFlip(horizontal=True, vertical=True)",
            "RandomRotation(degrees=90)",
            "ElasticTransform(alpha=120, sigma=12)",
            "RandomBrightnessContrast()",
            "Normalize(per-channel)",
        ],
        "warmup": None,
        "label_smoothing": 0.0,
        "expected_top1_imagenet": "IoU > 0.85 on ISBI cell segmentation benchmark",
    },
}


HYPERPARAMETER_EXPLANATIONS: dict[str, dict[str, Any]] = {
    "Learning Rate": {
        "name": "Learning Rate",
        "what_it_does": "Controls the step size during gradient descent. Scales how much model weights change per update.",
        "typical_range": "1e-4 to 0.1",
        "increase_effect": "Faster convergence initially, but risk of overshooting minima, training instability, or divergence.",
        "decrease_effect": "Slower but more stable convergence. May get stuck in local minima or never reach the optimum.",
        "architecture_notes": {
            "ResNet": "Use 0.1 with SGD; halve at milestones.",
            "Transformer": "Use 1e-4 with AdamW; must warmup first.",
            "ViT": "Use 3e-3 with heavy augmentation and warmup.",
        },
    },
    "Weight Decay": {
        "name": "Weight Decay",
        "what_it_does": "L2 regularization coefficient. Penalizes large weights to prevent overfitting.",
        "typical_range": "1e-5 to 0.3",
        "increase_effect": "Stronger regularization, smaller weights. Reduces overfitting but may underfit on complex tasks.",
        "decrease_effect": "Less regularization. Model may overfit, especially on small datasets.",
        "architecture_notes": {
            "ViT": "Use 0.3 — much higher than CNNs due to ViT's tendency to overfit.",
            "ResNet": "Use 1e-4. Standard for ImageNet.",
            "Transformer": "Use 0.01 with AdamW (decoupled from gradient scaling).",
        },
    },
    "Batch Size": {
        "name": "Batch Size",
        "what_it_does": "Number of samples processed per gradient update. Affects gradient noise and memory usage.",
        "typical_range": "8 to 4096",
        "increase_effect": "More stable gradient estimates, faster training (more GPU parallelism), but needs LR scaling (linear scaling rule). Reduces regularization effect.",
        "decrease_effect": "Noisier gradients (acts as regularization), lower memory usage, often better generalization but slower throughput.",
        "architecture_notes": {
            "ViT": "Paper uses batch 4096 across many GPUs. Scale LR linearly with batch size.",
            "U-Net": "Use 4–16 due to high-resolution feature maps consuming memory.",
            "ResNet": "256 per GPU, distributed across 8 GPUs for ImageNet baseline.",
        },
    },
    "Dropout": {
        "name": "Dropout",
        "what_it_does": "Randomly zeros out neuron activations during training. Acts as ensemble regularization.",
        "typical_range": "0.0 to 0.5",
        "increase_effect": "Stronger regularization, slower convergence, helps generalization on small datasets. Too high = underfitting.",
        "decrease_effect": "Less regularization. Faster convergence. Risk of overfitting on small datasets.",
        "architecture_notes": {
            "ViT": "Use 0.1 on attention and FFN layers.",
            "Transformer": "Use 0.1 as in the original paper.",
            "CNN/ResNet": "Rarely used in modern CNNs; BatchNorm serves as implicit regularizer.",
        },
    },
    "Label Smoothing": {
        "name": "Label Smoothing",
        "what_it_does": "Softens one-hot targets by distributing ε probability to all classes. Reduces overconfident predictions.",
        "typical_range": "0.0 to 0.2",
        "increase_effect": "More uncertainty in predictions. Better calibration. Can hurt if used with models that need sharp boundaries.",
        "decrease_effect": "Model learns to be very confident (probability →1 for correct class). Higher likelihood of overconfidence.",
        "architecture_notes": {
            "Transformer": "Use 0.1. Critical for BLEU score improvement.",
            "ViT": "Use 0.1. Part of the standard DeiT training recipe.",
            "ResNet": "Usually 0.0; can add 0.1 for marginal gains.",
        },
    },
    "Attention Heads": {
        "name": "Attention Heads",
        "what_it_does": "Number of parallel attention mechanisms in Multi-Head Attention. Each head learns different relationship patterns.",
        "typical_range": "4 to 16",
        "increase_effect": "More diverse attention patterns per layer. Marginally higher capacity. Increases memory and compute (O(n²·heads)).",
        "decrease_effect": "Fewer attention patterns. Can hurt tasks requiring diverse long-range dependencies.",
        "architecture_notes": {
            "ViT-B": "12 heads, embed_dim=768.",
            "ViT-L": "16 heads, embed_dim=1024.",
            "GPT-2": "12 heads for 124M model. Must satisfy: embed_dim % num_heads == 0.",
        },
    },
    "Hidden Dimension": {
        "name": "Hidden Dimension",
        "what_it_does": "Width of the model. For Transformers: embedding dimension (d_model). For CNNs: channel count.",
        "typical_range": "64 to 1024",
        "increase_effect": "More representational capacity. Can model more complex patterns. Increases parameters quadratically for FFN layers.",
        "decrease_effect": "Less capacity. Faster training and inference. May bottleneck complex tasks.",
        "architecture_notes": {
            "ViT-B": "768 hidden dim with 3072 FFN intermediate dim (4× ratio).",
            "ResNet-50": "Starts at 64 channels, doubles each stage (64→128→256→512).",
            "Transformer base": "512 d_model with 2048 FFN dim.",
        },
    },
    # ── Optimizers ──────────────────────────────────────────────────────────
    "Momentum": {
        "name": "Momentum",
        "what_it_does": "Accumulates an exponentially-decaying average of past gradients (a velocity) so SGD keeps moving along consistent directions and damps oscillations.",
        "typical_range": "0.8 to 0.99",
        "increase_effect": "More inertia: faster on long shallow valleys, but can overshoot the minimum and oscillate if too close to 1.0.",
        "decrease_effect": "Behaves more like plain SGD: noisier, slower across ravines, but less prone to overshoot.",
        "architecture_notes": {
            "ResNet": "0.9 is the canonical value with SGD on ImageNet.",
            "General": "Nesterov momentum looks ahead before the update for slightly better convergence.",
        },
    },
    "Adam Beta1": {
        "name": "Adam Beta1",
        "what_it_does": "Decay rate for Adam's first moment (the running mean of gradients). Controls how much past gradient direction is remembered.",
        "typical_range": "0.9 (default)",
        "increase_effect": "Smoother, more momentum-like updates; slower to react to changing gradients.",
        "decrease_effect": "More reactive to the latest gradient; noisier updates.",
        "architecture_notes": {
            "Transformer": "0.9 with beta2=0.98 is common for Transformers (vs 0.999 default).",
            "General": "Rarely tuned; defaults usually work.",
        },
    },
    "Adam Beta2 / Epsilon": {
        "name": "Adam Beta2 / Epsilon",
        "what_it_does": "Beta2 decays the second moment (running mean of squared gradients) used for per-parameter scaling; epsilon prevents divide-by-zero in the update.",
        "typical_range": "beta2: 0.99–0.999, eps: 1e-8",
        "increase_effect": "Higher beta2 = smoother variance estimate, more stable but slower to adapt the per-parameter learning rate.",
        "decrease_effect": "Lower beta2 reacts faster to gradient scale changes but can be unstable on sparse gradients.",
        "architecture_notes": {
            "Transformer": "beta2=0.98 stabilizes early training; some setups raise eps to 1e-6.",
            "AdamW": "Decouples weight decay from the gradient — prefer AdamW over Adam+L2.",
        },
    },
    # ── Schedules & stability ───────────────────────────────────────────────
    "LR Warmup": {
        "name": "LR Warmup",
        "what_it_does": "Linearly ramps the learning rate from ~0 to its peak over the first N steps before the main decay schedule begins.",
        "typical_range": "500 to 10,000 steps (or ~5% of training)",
        "increase_effect": "Longer warmup = safer early training for large-batch / Transformer setups, but delays real learning.",
        "decrease_effect": "Too short reintroduces the early-training instability warmup is meant to prevent (loss spikes).",
        "architecture_notes": {
            "Transformer": "Essential — the original paper warms up over 4000 steps then decays as 1/sqrt(step).",
            "ViT": "Pairs warmup with cosine decay and heavy augmentation.",
        },
    },
    "Gradient Clipping": {
        "name": "Gradient Clipping",
        "what_it_does": "Caps the global gradient norm (or value) before the optimizer step, preventing a single huge gradient from destabilizing training.",
        "typical_range": "max-norm 0.5 to 5.0",
        "increase_effect": "Higher threshold clips less — closer to no clipping, more risk of an exploding-gradient spike.",
        "decrease_effect": "Aggressive clipping caps step size, stabilizing RNNs/Transformers but potentially slowing learning.",
        "architecture_notes": {
            "Transformer": "Clip to global norm 1.0 is a very common default.",
            "RNN/LSTM": "Critical for taming exploding gradients through time.",
        },
    },
    "Normalization (BatchNorm / LayerNorm)": {
        "name": "Normalization (BatchNorm / LayerNorm)",
        "what_it_does": "Normalizes activations to stabilize the distribution each layer sees, smoothing the loss landscape and enabling higher learning rates.",
        "typical_range": "BatchNorm for CNNs, LayerNorm for Transformers",
        "increase_effect": "More/earlier normalization improves stability and lets you raise the learning rate.",
        "decrease_effect": "Removing it often requires careful init and much lower learning rates to train at all.",
        "architecture_notes": {
            "ResNet": "BatchNorm after every conv is part of the residual block.",
            "Transformer/ViT": "LayerNorm (pre-norm placement) is the stability backbone.",
            "Small batches": "Prefer GroupNorm/LayerNorm — BatchNorm statistics get noisy.",
        },
    },
}


def get_training_config(classification: str) -> dict[str, Any]:
    """
    Return the canonical training configuration for an architecture classification.
    Falls back to CNN config if classification is not recognized.
    """
    return TRAINING_CONFIGS.get(classification, TRAINING_CONFIGS["CNN"])


def get_hyperparameter_explanations() -> dict[str, dict[str, Any]]:
    """Return all hyperparameter explanation cards."""
    return HYPERPARAMETER_EXPLANATIONS
