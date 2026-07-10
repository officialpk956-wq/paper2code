from typing import Any


def generate_playground_insights(
    arch_type: str,
    original_metrics: dict[str, Any],
    new_metrics: dict[str, Any],
    original_config: dict[str, Any],
    new_config: dict[str, Any],
) -> list[str]:
    """
    Generate rule-based educational insights based on architecture modifications.

    Args:
        arch_type: "ResNet", "Vision Transformer", or "U-Net"
        original_metrics: Metrics from baseline
        new_metrics: Metrics from modified
        original_config: Base configuration
        new_config: Modified configuration

    Returns:
        List of markdown strings with educational insights.
    """
    insights = []

    # Delta calculations
    param_delta = new_metrics["total_params_estimate"] - original_metrics["total_params_estimate"]
    flops_delta = new_metrics["total_flops_score"] - original_metrics["total_flops_score"]
    depth_delta = new_metrics["depth"] - original_metrics["depth"]

    # Generic Parameter Insights
    if param_delta > 0:
        if param_delta > original_metrics["total_params_estimate"] * 1.5:
            insights.append(
                "📈 **Massive Parameter Increase**: You have more than doubled the parameter count. This increases model capacity significantly, but also raises the risk of overfitting if you don't have enough data."
            )
        else:
            insights.append(
                "📈 **Parameter Increase**: Increasing parameters adds capacity to learn more complex features, but requires more memory to store the weights."
            )
    elif param_delta < 0:
        insights.append(
            "📉 **Parameter Reduction**: Reducing parameters makes the model lighter and faster to load. It acts as a form of regularization, potentially reducing overfitting on small datasets."
        )

    # Architecture Specific Insights
    if arch_type == "ResNet":
        base_ch = original_config.get("base_channels", 64)
        new_ch = new_config.get("base_channels", 64)
        if new_ch > base_ch:
            insights.append(
                "🔍 **Wider Convolutions**: Doubling channels significantly increases parameter count because convolutional parameters scale quadratically with channels ($C_{in} \\times C_{out}$)."
            )
        elif new_ch < base_ch:
            insights.append(
                "🔍 **Narrower Convolutions**: Halving channels drastically reduces both FLOPs and parameters. Useful for mobile deployment."
            )

        base_stages = original_config.get("stages", 4)
        new_stages = new_config.get("stages", 4)
        base_blocks = original_config.get("blocks_per_stage", 2)
        new_blocks = new_config.get("blocks_per_stage", 2)

        if new_stages > base_stages or new_blocks > base_blocks:
            insights.append(
                "🧱 **Deeper Architecture**: Additional residual blocks increase depth while preserving gradient flow. This allows the network to learn more abstract hierarchical features without suffering from the vanishing gradient problem."
            )

    elif arch_type == "Vision Transformer":
        base_heads = original_config.get("num_heads", 12)
        new_heads = new_config.get("num_heads", 12)
        if new_heads > base_heads:
            insights.append(
                "🧠 **More Attention Heads**: Increasing attention heads raises compute requirements. It allows the model to attend to information from different representation subspaces at different positions simultaneously."
            )

        base_dim = original_config.get("hidden_size", 768)
        new_dim = new_config.get("hidden_size", 768)
        if new_dim > base_dim:
            insights.append(
                "📏 **Larger Hidden Dimension**: Expanding the embedding dimension increases the representational power per token, but leads to quadratic parameter growth in the linear projections ($O(d^2)$)."
            )

        base_depth = original_config.get("depth", 4)
        new_depth = new_config.get("depth", 4)
        if new_depth > base_depth:
            insights.append(
                "🥞 **Deeper Encoder**: Adding transformer layers increases the path length for information to flow. Attention matrices in deeper layers often become more global and abstract."
            )

    elif arch_type == "U-Net":
        base_ch = original_config.get("base_channels", 64)
        new_ch = new_config.get("base_channels", 64)
        if new_ch > base_ch:
            insights.append(
                "🎨 **Rich Spatial Features**: Increasing base channels gives the U-Net more capacity to represent textures and boundaries at every resolution scale."
            )

        base_stages = original_config.get("stages", 3)
        new_stages = new_config.get("stages", 3)
        if new_stages > base_stages:
            insights.append(
                "🔬 **Larger Receptive Field**: Deeper U-Nets improve receptive field by downsampling further, but increase memory consumption. Skip connections ensure fine-grained details are not lost despite the aggressive downsampling."
            )

    if not insights:
        insights.append("⚖️ **Unchanged**: No significant architectural changes detected.")

    return insights
