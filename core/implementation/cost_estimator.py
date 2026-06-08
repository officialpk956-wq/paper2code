"""
core/implementation/cost_estimator.py

Phase 10: Deterministic training cost estimation from architecture metadata.

All formulas are documented with derivations.
No fabricated measurements — outputs are estimates with stated assumptions.
"""

from typing import Dict, Any, Optional


# ── GPU Specifications (published specs) ──────────────────────────────────────
# Sources: NVIDIA product pages, official datasheets, and published benchmarks.
# tflops_fp32: peak FP32 TFLOPS (non-tensor-core where applicable)
# vram_gb: total GPU memory
# cloud_cost_usd_hr: approximate on-demand price (AWS/GCP/Azure, varies by provider)
# memory_bandwidth_gb_s: peak HBM/GDDR bandwidth in GB/s
GPU_SPECS: Dict[str, Dict[str, Any]] = {
    "T4": {
        "tflops_fp32": 8.1,              # NVIDIA T4 FP32 datasheet
        "vram_gb": 16,
        "cloud_cost_usd_hr": 0.53,       # Approx. GCP n1-standard per GPU
        "memory_bandwidth_gb_s": 320,
    },
    "L4": {
        "tflops_fp32": 30.3,             # NVIDIA L4 FP32 datasheet
        "vram_gb": 24,
        "cloud_cost_usd_hr": 0.72,       # Approx. GCP g2-standard per GPU
        "memory_bandwidth_gb_s": 300,
    },
    "RTX3090": {
        "tflops_fp32": 35.6,             # NVIDIA RTX 3090 FP32
        "vram_gb": 24,
        "cloud_cost_usd_hr": 1.10,       # Approx. vast.ai pricing
        "memory_bandwidth_gb_s": 936,
    },
    # Legacy key kept for backward compatibility
    "RTX 3090": {
        "tflops_fp32": 35.6,
        "vram_gb": 24,
        "cloud_cost_usd_hr": 1.10,
        "memory_bandwidth_gb_s": 936,
    },
    "RTX4090": {
        "tflops_fp32": 82.6,             # NVIDIA RTX 4090 FP32
        "vram_gb": 24,
        "cloud_cost_usd_hr": 1.89,       # Approx. vast.ai pricing
        "memory_bandwidth_gb_s": 1008,
    },
    "V100": {
        "tflops_fp32": 14.0,             # NVIDIA V100 FP32 datasheet
        "vram_gb": 16,
        "cloud_cost_usd_hr": 2.48,       # Approx. AWS p3.2xlarge per GPU
        "memory_bandwidth_gb_s": 900,
    },
    "A100": {
        "tflops_fp32": 19.5,             # NVIDIA A100 40GB FP32 datasheet
        "vram_gb": 40,
        "cloud_cost_usd_hr": 3.20,       # Approx. AWS p4d.xlarge per GPU
        "memory_bandwidth_gb_s": 1555,
    },
    "A100_80GB": {
        "tflops_fp32": 19.5,             # Same compute as 40GB, double VRAM
        "vram_gb": 80,
        "cloud_cost_usd_hr": 4.10,       # Approx. AWS p4de per GPU
        "memory_bandwidth_gb_s": 2000,
    },
    "H100": {
        "tflops_fp32": 67.0,             # NVIDIA H100 SXM5 FP32 datasheet
        "vram_gb": 80,
        "cloud_cost_usd_hr": 8.00,       # Approx. AWS p5 per GPU
        "memory_bandwidth_gb_s": 3350,
    },
}

# ── Architecture FLOPs/param profiles (published paper values) ────────────────
# params_M:          total trainable parameters in millions
# flops_per_image_G: GFLOPs for a single 224×224 forward pass (unless noted)
# typical_epochs:    standard training epochs from original papers
# base_memory_gb:    baseline activation VRAM at batch=32
ARCH_PROFILES: Dict[str, Dict[str, Any]] = {
    # Classic ConvNets
    "LeNet": {
        "params_M": 0.06,              # LeCun 1998: ~60K params
        "flops_per_image_G": 0.00042,  # ~0.42 MFLOPs for 32×32 input
        "typical_epochs": 20,
        "base_memory_gb": 0.1,
    },
    "AlexNet": {
        "params_M": 61.1,              # Krizhevsky 2012: 61.1M params
        "flops_per_image_G": 0.72,     # ~720 MFLOPs per 224×224 image
        "typical_epochs": 90,
        "base_memory_gb": 1.5,
    },
    "VGG16": {
        "params_M": 138.4,             # Simonyan 2014: 138.4M params
        "flops_per_image_G": 15.5,     # ~15.5 GFLOPs per 224×224 image
        "typical_epochs": 74,
        "base_memory_gb": 3.5,
    },
    "VGG19": {
        "params_M": 143.7,             # Simonyan 2014: 143.7M params
        "flops_per_image_G": 19.6,     # ~19.6 GFLOPs per 224×224 image
        "typical_epochs": 74,
        "base_memory_gb": 4.0,
    },
    "GoogLeNet": {
        "params_M": 6.8,               # Szegedy 2014: 6.8M params (Inception v1)
        "flops_per_image_G": 1.5,      # ~1.5 GFLOPs per 224×224 image
        "typical_epochs": 90,
        "base_memory_gb": 2.0,
    },
    # ResNet family
    "ResNet18": {
        "params_M": 11.7,              # He 2015: 11.7M params
        "flops_per_image_G": 1.82,     # ~1.82 GFLOPs per 224×224 image
        "typical_epochs": 90,
        "base_memory_gb": 2.5,
    },
    "ResNet34": {
        "params_M": 21.8,              # He 2015: 21.8M params
        "flops_per_image_G": 3.67,     # ~3.67 GFLOPs per 224×224 image
        "typical_epochs": 90,
        "base_memory_gb": 3.5,
    },
    "ResNet50": {
        "params_M": 25.6,              # He 2015: 25.6M params
        "flops_per_image_G": 4.1,      # ~4.1 GFLOPs per 224×224 image
        "typical_epochs": 90,
        "base_memory_gb": 4.0,
    },
    # Generic alias — maps to ResNet50 values
    "ResNet": {
        "params_M": 25.6,
        "flops_per_image_G": 4.1,
        "typical_epochs": 90,
        "base_memory_gb": 4.0,
    },
    # DenseNet family
    "DenseNet121": {
        "params_M": 8.0,               # Huang 2017: 8.0M params
        "flops_per_image_G": 2.87,     # ~2.87 GFLOPs per 224×224 image
        "typical_epochs": 90,
        "base_memory_gb": 4.5,         # Dense skip connections increase VRAM
    },
    # Generic alias — maps to DenseNet-201 values
    "DenseNet": {
        "params_M": 20.0,              # DenseNet-201
        "flops_per_image_G": 4.4,
        "typical_epochs": 300,
        "base_memory_gb": 5.5,
    },
    # Efficient / mobile nets
    "MobileNetV2": {
        "params_M": 3.4,               # Sandler 2018: 3.4M params
        "flops_per_image_G": 0.30,     # ~300 MFLOPs per 224×224 image
        "typical_epochs": 300,
        "base_memory_gb": 1.2,
    },
    "EfficientNetB0": {
        "params_M": 5.3,               # Tan & Le 2019: 5.3M params
        "flops_per_image_G": 0.39,     # ~390 MFLOPs per 224×224 image
        "typical_epochs": 350,
        "base_memory_gb": 1.5,
    },
    # Segmentation
    "FCN": {
        "params_M": 134.5,             # Long 2015: FCN-32s ~134.5M params
        "flops_per_image_G": 66.5,     # ~66.5 GFLOPs at 500×500
        "typical_epochs": 50,
        "base_memory_gb": 7.0,
    },
    "UNet": {
        "params_M": 31.0,              # Ronneberger 2015: ~31M params
        "flops_per_image_G": 54.7,     # ~54.7 GFLOPs at 572×572
        "typical_epochs": 50,
        "base_memory_gb": 6.0,
    },
    # Sequence / attention models
    "Transformer": {
        "params_M": 65.0,              # Vaswani 2017 base model: 65M params
        "flops_per_image_G": 12.0,     # Approx. per sequence step (512 tokens)
        "typical_epochs": 100,
        "base_memory_gb": 8.0,
    },
    "ViT": {
        "params_M": 86.6,              # Dosovitskiy 2020: ViT-B/16 86.6M params
        "flops_per_image_G": 17.6,     # ~17.6 GFLOPs per 224×224 image
        "typical_epochs": 300,
        "base_memory_gb": 10.0,        # High due to attention matrices
    },
    # Generic aliases
    "CNN": {
        "params_M": 14.7,              # VGG-16 as representative CNN
        "flops_per_image_G": 15.5,
        "typical_epochs": 90,
        "base_memory_gb": 3.5,
    },
    "Encoder-Decoder": {
        "params_M": 31.0,              # U-Net as representative enc-dec
        "flops_per_image_G": 54.7,
        "typical_epochs": 50,
        "base_memory_gb": 6.0,
    },
}


def estimate_training_cost(
    architecture: str,
    dataset_size: int,
    batch_size: int,
    gpu_type: str,
    epochs: Optional[int] = None,
    mixed_precision: bool = False,
    gradient_checkpointing: bool = False,
) -> Dict[str, Any]:
    """
    Estimate GPU memory, training time, and compute cost for a training run.

    Formula derivations:
    ─────────────────────────────────────────────────────────────
    VRAM estimate:
        vram = base_model_vram + activation_vram + optimizer_state
        base_model_vram   = params_M * 4 bytes (fp32) / 1024 = GB
        activation_vram   = base_memory_gb * (batch_size / 32)
        optimizer_state   = 2× model params for Adam (m, v vectors)
        total             = base_model + activation + optimizer
        mixed_precision   → total_vram /= 1.6  (FP16 weights + FP32 master copy)
        grad_checkpointing→ activation_vram /= 2.0 (trade compute for memory)

    Training time estimate:
        steps_per_epoch   = ceil(dataset_size / batch_size)
        total_steps       = steps_per_epoch * epochs
        flops_per_step    = flops_per_image_G * batch_size (GFLOPs)
        total_gflops      = flops_per_step * total_steps
        total_tflops      = total_gflops / 1000
        gpu_efficiency    = 0.35  (realistic 35% MFU for training)
        wall_time_hours   = total_tflops / (gpu_tflops_fp32 * gpu_efficiency * 3600)
        grad_checkpointing→ wall_time_hours *= 1.25  (recomputation overhead)

    Cost estimate:
        cost_usd          = wall_time_hours * cloud_cost_usd_hr
    ─────────────────────────────────────────────────────────────

    All estimates carry significant uncertainty (±30-50%).
    These are planning estimates, not guarantees.

    Args:
        architecture:          Classification string (ResNet, ViT, etc.)
        dataset_size:          Number of training samples
        batch_size:            Batch size per gradient step
        gpu_type:              One of GPU_SPECS keys
        epochs:                Override default epoch count for architecture
        mixed_precision:       When True, effective VRAM usage divided by 1.6
        gradient_checkpointing:When True, activation VRAM halved but time ×1.25

    Returns:
        Dict with gpu_memory_gb, training_hours, compute_cost_usd,
        steps_total, assumptions, derivation notes, and gpu_comparison.
    """
    import math

    # Normalise frontend key variants to backend dict keys
    _arch_key_map = {
        "EfficientNet-B0": "EfficientNetB0",
        "U-Net": "UNet",
        "Encoder-Decoder": "UNet",
        "DenseNet": "DenseNet121",
        "ResNet": "ResNet50",
        "CNN": "VGG16",
        "RTX 3090": "RTX3090",
        "RTX 4090": "RTX4090",
    }
    architecture = _arch_key_map.get(architecture, architecture)
    gpu_type     = _arch_key_map.get(gpu_type, gpu_type)

    arch_profile = ARCH_PROFILES.get(architecture, ARCH_PROFILES.get("ResNet50", ARCH_PROFILES.get("ResNet", list(ARCH_PROFILES.values())[0])))
    gpu = GPU_SPECS.get(gpu_type, GPU_SPECS["A100"])

    epochs_used = epochs or arch_profile["typical_epochs"]

    # ── VRAM estimate ─────────────────────────────────────────────────────────
    params_M = arch_profile["params_M"]
    model_vram_gb = (params_M * 1e6 * 4) / (1024 ** 3)            # fp32 weights
    optimizer_vram_gb = model_vram_gb * 2.0                        # Adam m + v
    activation_scale = max(1.0, batch_size / 32)
    activation_vram_gb = arch_profile["base_memory_gb"] * activation_scale

    # Apply gradient checkpointing: halve activation VRAM
    if gradient_checkpointing:
        activation_vram_gb /= 2.0

    total_vram_gb = model_vram_gb + optimizer_vram_gb + activation_vram_gb

    # Apply mixed precision: overall VRAM divided by 1.6
    if mixed_precision:
        total_vram_gb /= 1.6

    fits_in_gpu = total_vram_gb <= gpu["vram_gb"]

    # ── Training time estimate ────────────────────────────────────────────────
    steps_per_epoch = math.ceil(dataset_size / batch_size)
    total_steps = steps_per_epoch * epochs_used

    flops_per_step_G = arch_profile["flops_per_image_G"] * batch_size
    total_gflops = flops_per_step_G * total_steps
    total_tflops = total_gflops / 1000.0

    # Backward pass ≈ 2× forward pass (total factor = 3× forward)
    total_tflops_with_bwd = total_tflops * 3.0

    gpu_efficiency = 0.35  # Realistic MFU (model FLOP utilization) for training
    wall_time_seconds = total_tflops_with_bwd / (gpu["tflops_fp32"] * gpu_efficiency)
    wall_time_hours = wall_time_seconds / 3600.0

    # Apply gradient checkpointing time penalty
    if gradient_checkpointing:
        wall_time_hours *= 1.25

    # ── Cost estimate ─────────────────────────────────────────────────────────
    compute_cost_usd = wall_time_hours * gpu["cloud_cost_usd_hr"]

    # ── Per-GPU comparison across all 8 GPUs ─────────────────────────────────
    gpu_comparison = []
    _comparison_gpu_keys = ["T4", "L4", "RTX3090", "RTX4090", "V100", "A100", "A100_80GB", "H100"]
    for gname in _comparison_gpu_keys:
        g = GPU_SPECS[gname]
        g_wall_s = total_tflops_with_bwd / (g["tflops_fp32"] * gpu_efficiency)
        g_wall_h = g_wall_s / 3600.0
        if gradient_checkpointing:
            g_wall_h *= 1.25
        g_cost = round(g_wall_h * g["cloud_cost_usd_hr"], 2)
        # Apply same VRAM logic for comparison
        g_act_vram = activation_vram_gb  # already has grad_checkpointing applied
        g_total_vram = model_vram_gb + optimizer_vram_gb + g_act_vram
        if mixed_precision:
            g_total_vram /= 1.6
        gpu_comparison.append({
            "gpu_name": gname,
            "vram_gb": g["vram_gb"],
            "training_hours": round(g_wall_h, 1),
            "compute_cost_usd": g_cost,
            "fits": g_total_vram <= g["vram_gb"],
        })

    # Calculate FLOPs for arch_profile return object
    arch_flops_per_batch = arch_profile["flops_per_image_G"] * batch_size

    # Build transparency assumptions list
    assumptions = [
        f"Architecture profile: {params_M}M params, {arch_profile['flops_per_image_G']} GFLOPs/image",
        f"GPU: {gpu_type} @ {gpu['tflops_fp32']} TFLOPS FP32, {gpu['vram_gb']}GB VRAM",
        f"Cloud rate: ${gpu['cloud_cost_usd_hr']}/hr (approximate, varies by provider)",
        "GPU utilization assumed at 35% MFU (realistic for single-GPU training)",
        "Backward pass estimated as 2× forward pass FLOPs",
        "Estimates carry ±30–50% uncertainty — use for planning only",
    ]
    if mixed_precision:
        assumptions.append("Mixed precision enabled: effective VRAM reduced by factor 1.6 (FP16 weights + FP32 master copy)")
    if gradient_checkpointing:
        assumptions.append("Gradient checkpointing enabled: activation VRAM halved, training time increased by 25%")

    return {
        "architecture": architecture,
        "gpu_type": gpu_type,
        "dataset_size": dataset_size,
        "batch_size": batch_size,
        "epochs": epochs_used,

        # Results (Frontend expected fields)
        "gpu_memory_gb": round(total_vram_gb, 1),
        "training_hours": round(wall_time_hours, 1),
        "compute_cost_usd": round(compute_cost_usd, 2),

        # Profile objects for frontend UI display
        "arch_profile": {
            "flops": arch_flops_per_batch,
            "params": params_M,
        },
        "gpu_profile": {
            "tflops": gpu["tflops_fp32"],
            "cost_per_hour": gpu["cloud_cost_usd_hr"],
        },

        # Detailed breakdown (for advanced users)
        "gpu_memory_breakdown": {
            "model_weights_gb": round(model_vram_gb, 2),
            "optimizer_state_gb": round(optimizer_vram_gb, 2),
            "activations_gb": round(activation_vram_gb, 2),
        },
        "fits_in_single_gpu": fits_in_gpu,
        "gpu_vram_available_gb": gpu["vram_gb"],

        "steps_total": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "training_days": round(wall_time_hours / 24, 2),

        # Precision / memory flags
        "mixed_precision_enabled": mixed_precision,
        "gradient_checkpointing_enabled": gradient_checkpointing,

        # Cross-GPU comparison
        "gpu_comparison": gpu_comparison,

        # Transparency
        "assumptions": assumptions,
        "label": "Deterministic Estimate — Not a guarantee"
    }



