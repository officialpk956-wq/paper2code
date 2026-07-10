from core.architecture_graph import ArchitectureGraph


def classify_architecture(graph: ArchitectureGraph) -> str:
    """
    Deterministically classify the architecture graph into one of 14 families.
    """
    types = [node.type.lower() for node in graph.nodes]
    names = [node.name.lower() for node in graph.nodes]

    all_str = " ".join(types + names)

    has_conv = any("conv" in t for t in types)
    has_attention = any("attention" in t or "transformer" in t or "selfatt" in t for t in types)
    has_upsample = any("upsample" in t or "convtranspose" in t or "upconv" in t for t in types)

    if "generator" in all_str or "discriminator" in all_str or "adversarial" in all_str:
        return "gan"
    elif "noise_scheduler" in all_str and "latent" in all_str and has_conv:
        return "ldm"
    elif (
        ("sinusoidal" in all_str or "denoise" in all_str or "timestep" in all_str)
        and has_upsample
        and has_conv
    ):
        return "diffusion"
    elif "masking" in all_str or "random_mask" in all_str or "mask_ratio" in all_str:
        return "mae"
    elif "patch_embed" in all_str or "patch_embedding" in all_str:
        return "vit"
    elif "window_attention" in all_str or "patch_merging" in all_str:
        return "swin"
    elif "embedding" in all_str and has_attention and not has_conv:
        return "bert_gpt"
    elif has_attention and not has_conv:
        return "transformer"
    elif "detection_head" in all_str or "fpn" in all_str or "anchor" in all_str:
        return "yolo"
    elif ("mbconv" in all_str or "squeeze_excitation" in all_str) and has_conv:
        return "efficientnet"
    elif ("depthwise" in all_str or "inverted_residual" in all_str) and has_conv:
        return "mobilenet"
    elif "dense_block" in all_str or "growth_rate" in all_str or "transition" in all_str:
        return "densenet"
    elif has_conv and has_upsample:
        return "unet"
    elif has_conv and ("residual" in all_str or "bottleneck" in all_str):
        return "resnet"
    elif has_conv:
        return "cnn"
    else:
        return "cnn"  # Fallback if unknown


def infer_family_from_name(paper_name: str) -> str | None:
    """Keyword-based fallback when graph is unavailable."""
    name = paper_name.lower()
    KEYWORDS = {
        "efficientnet": "efficientnet",
        "mbconv": "efficientnet",
        "swin": "swin",
        "window attention": "swin",
        "dcgan": "gan",
        "stylegan": "gan",
        "cyclegan": "gan",
        "generative adversarial": "gan",
        "densenet": "densenet",
        "dense block": "densenet",
        "bert": "bert_gpt",
        "gpt": "bert_gpt",
        "language model": "bert_gpt",
        "mobilenet": "mobilenet",
        "inverted residual": "mobilenet",
        "masked autoencoder": "mae",
        " mae ": "mae",
        "latent diffusion": "ldm",
        "stable diffusion": "ldm",
        "ddpm": "diffusion",
        "denoising diffusion": "diffusion",
        "yolo": "yolo",
        "unet": "unet",
        "u-net": "unet",
        "vit": "vit",
        "vision transformer": "vit",
        "resnet": "resnet",
        "residual network": "resnet",
        "transformer": "transformer",
        "attention is all you need": "transformer",
    }
    for keyword, family in KEYWORDS.items():
        if keyword in name:
            return family
    return None
