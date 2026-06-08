"""
core/implementation/cost_estimator.py

Phase 10: Deterministic training cost estimation from architecture metadata.

All formulas are documented with derivations.
No fabricated measurements — outputs are estimates with stated assumptions.
"""

from typing import Dict, Any, Optional


# ── GPU Specifications (published specs) ──────────────────────────────────────
# Sources: NVIDIA product pages and published benchmarks.
GPU_SPECS: Dict[str, Dict[str, Any]] = {
    "A100": {
        "tflops_fp32": 19.5,       # TFLOPS FP32 (non-tensor-core)
        "tflops_bf16": 77.6,       # TFLOPS BF16 tensor core
        "vram_gb": 40,
        "cloud_cost_usd_hr": 3.20, # Approx. AWS p4d.xlarge per GPU
        "memory_bandwidth_gb_s": 1555,
    },
    "V100": {
        "tflops_fp32": 14.0,
        "tflops_bf16": 28.0,
        "vram_gb": 16,
        "cloud_cost_usd_hr": 2.48, # Approx. AWS p3.2xlarge per GPU
        "memory_bandwidth_gb_s": 900,
    },
    "RTX 3090": {
        "tflops_fp32": 35.6,
        "tflops_bf16": 71.2,
        "vram_gb": 24,
        "cloud_cost_usd_hr": 1.10, # Approx. vast.ai pricing
        "memory_bandwidth_gb_s": 936,
    },
    "T4": {
        "tflops_fp32": 8.1,
        "tflops_bf16": 65.0,       # T4 INT8/FP16 optimized
        "vram_gb": 16,
        "cloud_cost_usd_hr": 0.53, # Approx. GCP n1-standard per GPU
        "memory_bandwidth_gb_s": 320,
    },
}

# ── Architecture FLOPs/param profiles (rough published estimates) ─────────────
ARCH_PROFILES: Dict[str, Dict[str, Any]] = {
    "ResNet": {
        "params_M": 25.6,          # ResNet-50 params (~25.6M)
        "flops_per_image_G": 4.1,  # ResNet-50 FLOPs per forward pass (GFLOPs)
        "typical_epochs": 90,
        "base_memory_gb": 4.0,     # Baseline VRAM at batch=32
    },
    "CNN": {
        "params_M": 14.7,          # VGG-16 params
        "flops_per_image_G": 15.5, # VGG-16 GFLOPs
        "typical_epochs": 90,
        "base_memory_gb": 3.5,
    },
    "DenseNet": {
        "params_M": 20.0,          # DenseNet-201 params
        "flops_per_image_G": 4.4,  # DenseNet-201 GFLOPs
        "typical_epochs": 300,
        "base_memory_gb": 5.5,     # Dense connections increase memory
    },
    "Transformer": {
        "params_M": 65.0,          # Base Transformer (WMT) params
        "flops_per_image_G": 12.0, # Approximation per sequence step
        "typical_epochs": 100,
        "base_memory_gb": 8.0,
    },
    "ViT": {
        "params_M": 86.6,          # ViT-B/16 params
        "flops_per_image_G": 17.6, # ViT-B/16 GFLOPs
        "typical_epochs": 300,
        "base_memory_gb": 10.0,    # High due to attention matrices
    },
    "Encoder-Decoder": {
        "params_M": 31.0,          # U-Net params (~31M)
        "flops_per_image_G": 54.7, # U-Net GFLOPs (high-res segmentation)
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

    Training time estimate:
        steps_per_epoch   = ceil(dataset_size / batch_size)
        total_steps       = steps_per_epoch * epochs
        flops_per_step    = flops_per_image_G * batch_size (GFLOPs)
        total_gflops      = flops_per_step * total_steps
        total_tflops      = total_gflops / 1000
        gpu_efficiency    = 0.35  (realistic 35% MFU for training)
        wall_time_hours   = total_tflops / (gpu_tflops_fp32 * gpu_efficiency * 3600)

    Cost estimate:
        cost_usd          = wall_time_hours * cloud_cost_usd_hr
    ─────────────────────────────────────────────────────────────

    All estimates carry significant uncertainty (±30-50%).
    These are planning estimates, not guarantees.

    Args:
        architecture:   Classification string (ResNet, ViT, etc.)
        dataset_size:   Number of training samples
        batch_size:     Batch size per gradient step
        gpu_type:       One of GPU_SPECS keys
        epochs:         Override default epoch count for architecture

    Returns:
        Dict with gpu_memory_gb, training_hours, compute_cost_usd,
        steps_total, assumptions, and derivation notes.
    """
    arch_profile = ARCH_PROFILES.get(architecture, ARCH_PROFILES["ResNet"])
    gpu = GPU_SPECS.get(gpu_type, GPU_SPECS["A100"])

    epochs_used = epochs or arch_profile["typical_epochs"]

    # ── VRAM estimate ─────────────────────────────────────────────────────────
    params_M = arch_profile["params_M"]
    model_vram_gb = (params_M * 1e6 * 4) / (1024 ** 3)            # fp32 weights
    optimizer_vram_gb = model_vram_gb * 2.0                        # Adam m + v
    activation_scale = max(1.0, batch_size / 32)
    activation_vram_gb = arch_profile["base_memory_gb"] * activation_scale
    total_vram_gb = model_vram_gb + optimizer_vram_gb + activation_vram_gb

    fits_in_gpu = total_vram_gb <= gpu["vram_gb"]

    # ── Training time estimate ────────────────────────────────────────────────
    import math
    steps_per_epoch = math.ceil(dataset_size / batch_size)
    total_steps = steps_per_epoch * epochs_used

    flops_per_step_G = arch_profile["flops_per_image_G"] * batch_size
    total_gflops = flops_per_step_G * total_steps
    total_tflops = total_gflops / 1000.0

    # Backward pass ≈ 2× forward pass
    total_tflops_with_bwd = total_tflops * 3.0

    gpu_efficiency = 0.35  # Realistic MFU (model FLOP utilization) for training
    wall_time_seconds = total_tflops_with_bwd / (gpu["tflops_fp32"] * gpu_efficiency)
    wall_time_hours = wall_time_seconds / 3600.0

    # ── Cost estimate ─────────────────────────────────────────────────────────
    compute_cost_usd = wall_time_hours * gpu["cloud_cost_usd_hr"]

    # Calculate FLOPs for arch_profile
    arch_flops_per_batch = arch_profile["flops_per_image_G"] * batch_size

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

        # Transparency
        "assumptions": [
            f"Architecture profile: {params_M}M params, {arch_profile['flops_per_image_G']} GFLOPs/image",
            f"GPU: {gpu_type} @ {gpu['tflops_fp32']} TFLOPS FP32, {gpu['vram_gb']}GB VRAM",
            f"Cloud rate: ${gpu['cloud_cost_usd_hr']}/hr (approximate, varies by provider)",
            "GPU utilization assumed at 35% MFU (realistic for single-GPU training)",
            "Backward pass estimated as 2× forward pass FLOPs",
            "Estimates carry ±30–50% uncertainty — use for planning only",
        ],
        "label": "Deterministic Estimate — Not a guarantee"
    }



