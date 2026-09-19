"""Curated definitions for common neural-network operations.

The table is deliberately small. Unknown operations must fall through to the
existing extraction and generation paths rather than receiving guessed facts.
"""

import re

OPERATIONS: dict[str, dict] = {
    "sigmoid": {
        "formula": "sigma(x) = 1 / (1 + exp(-x))",
        "latex": r"\sigma(x) = \frac{1}{1 + e^{-x}}",
        "syntax": "nn.Sigmoid()",
        "functional": "torch.sigmoid(x)",
        "aliases": ["sigmoid", "logistic", "logistic function"],
        "output_range": "(0, 1)",
        "notes": "Elementwise. Do not use as a final layer with BCEWithLogitsLoss.",
    },
    "relu": {
        "formula": "ReLU(x) = max(0, x)",
        "latex": r"\operatorname{ReLU}(x) = \max(0, x)",
        "syntax": "nn.ReLU(inplace=True)",
        "functional": "F.relu(x)",
        "aliases": ["relu", "rectified linear unit"],
        "output_range": "[0, infinity)",
        "notes": "Elementwise; the generator's default module uses inplace=True.",
    },
    "leakyrelu": {
        "formula": "LeakyReLU(x) = max(x, alpha * x)",
        "latex": r"\operatorname{LeakyReLU}(x) = \max(x, \alpha x)",
        "syntax": "nn.LeakyReLU(negative_slope=0.01)",
        "functional": "F.leaky_relu(x, negative_slope=0.01)",
        "aliases": ["leakyrelu"],
        "output_range": "(-infinity, infinity)",
        "notes": "Elementwise; alpha is the negative slope.",
    },
    "gelu": {
        "formula": "GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))",
        "latex": r"\operatorname{GELU}(x) \approx \frac{x}{2}\left[1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right]",
        "syntax": "nn.GELU()",
        "functional": "F.gelu(x, approximate='tanh')",
        "aliases": ["gelu", "gaussian error linear unit"],
        "output_range": "(-infinity, infinity)",
        "notes": 'Formula is the tanh approximation. nn.GELU() defaults to the exact erf form; use nn.GELU(approximate="tanh") for this approximation.',
    },
    "silu": {
        "formula": "SiLU(x) = x * sigmoid(x)",
        "latex": r"\operatorname{SiLU}(x) = x\sigma(x)",
        "syntax": "nn.SiLU()",
        "functional": "F.silu(x)",
        "aliases": ["silu", "sigmoid linear unit"],
        "output_range": "(-infinity, infinity)",
        "notes": "Elementwise smooth activation.",
    },
    "swish": {
        "formula": "Swish(x) = x * sigmoid(x)",
        "latex": r"\operatorname{Swish}(x) = x\sigma(x)",
        "syntax": "nn.SiLU()",
        "functional": "F.silu(x)",
        "aliases": ["swish"],
        "output_range": "(-infinity, infinity)",
        "notes": "Alias of SiLU (beta=1); CANONICAL_TYPES intentionally carries both names.",
    },
    "tanh": {
        "formula": "tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))",
        "latex": r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}",
        "syntax": "nn.Tanh()",
        "functional": "torch.tanh(x)",
        "aliases": ["tanh", "hyperbolic tangent"],
        "output_range": "(-1, 1)",
        "notes": "Elementwise activation centered at zero.",
    },
    "softmax": {
        "formula": "softmax(x_i) = exp(x_i) / sum_j exp(x_j)",
        "latex": r"\operatorname{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}",
        "syntax": "nn.Softmax(dim=-1)",
        "functional": "F.softmax(x, dim=-1)",
        "aliases": ["softmax"],
        "output_range": "(0, 1), sums to 1 along dim",
        "notes": "dim is required in practice; PyTorch warns when it is omitted.",
    },
    "batchnorm2d": {
        "formula": "y = gamma * (x - mean_batch) / sqrt(var_batch + eps) + beta",
        "latex": r"y = \gamma\frac{x - \mu_{\mathrm{batch}}}{\sqrt{\sigma^2_{\mathrm{batch}} + \epsilon}} + \beta",
        "syntax": "nn.BatchNorm2d(ch)",
        "functional": "F.batch_norm(x, running_mean, running_var)",
        "aliases": ["batchnorm2d", "batch norm", "batch normalization"],
        "notes": "Normalizes each channel over batch and spatial dimensions during training.",
    },
    "layernorm": {
        "formula": "y = gamma * (x - mean) / sqrt(var + eps) + beta",
        "latex": r"y = \gamma\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta",
        "syntax": "nn.LayerNorm(in_hs)",
        "functional": "F.layer_norm(x, normalized_shape)",
        "aliases": ["layernorm", "layer normalization"],
        "notes": "Normalizes over the specified trailing dimensions. PyTorch default eps=1e-5.",
    },
    "groupnorm": {
        "formula": "y = gamma * (x - mean_group) / sqrt(var_group + eps) + beta",
        "latex": r"y = \gamma\frac{x - \mu_{\mathrm{group}}}{\sqrt{\sigma^2_{\mathrm{group}} + \epsilon}} + \beta",
        "syntax": "nn.GroupNorm(num_groups, num_channels)",
        "functional": "F.group_norm(x, num_groups)",
        "aliases": ["groupnorm", "group normalization"],
        "notes": "Normalizes channel groups independently of batch size.",
    },
    "rmsnorm": {
        "formula": "y = x / sqrt(mean(x^2) + eps) * gamma",
        "latex": r"y = \frac{x}{\sqrt{\operatorname{mean}(x^2) + \epsilon}}\gamma",
        "syntax": "nn.RMSNorm(normalized_shape)",
        "functional": "F.rms_norm(x, normalized_shape)",
        "aliases": ["rmsnorm", "root mean square normalization"],
        "notes": "No mean subtraction; it is not LayerNorm.",
    },
    "scaled_dot_product_attention": {
        "formula": "Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k), dim=-1) @ V",
        "latex": r"\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}},\, \mathrm{last\ axis}\right)V",
        "syntax": None,
        "functional": "F.scaled_dot_product_attention(q, k, v)",
        "aliases": ["scaled_dot_product_attention"],
        "notes": "Softmax is applied over the last axis of the attention scores.",
    },
    "multiheadattention": {
        "formula": "MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O",
        "latex": r"\operatorname{MultiHead}(Q,K,V) = \operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O",
        "syntax": "nn.MultiheadAttention(embed_dim=in_hs, num_heads=heads, batch_first=True)",
        "functional": None,
        "aliases": ["multiheadattention"],
        "notes": "Each head uses scaled dot-product attention on learned projections.",
    },
    "linear": {
        "formula": "y = x @ W^T + b",
        "latex": r"y = xW^T + b",
        "syntax": "nn.Linear(in_hs, out_hs)",
        "functional": "F.linear(x, weight, bias)",
        "aliases": ["linear", "fully connected", "dense layer"],
        "notes": "Applies an affine transformation to the last input dimension.",
    },
    "conv2d": {
        "formula": "y[n, c_out, h, w] = sum(x * kernel) + bias",
        "latex": r"y_{n,c_{out},h,w} = \sum x \ast K + b_{c_{out}}",
        "syntax": "nn.Conv2d(ch, ch, kernel_size=k, padding=k // 2)",
        "functional": "F.conv2d(x, weight, bias)",
        "aliases": ["conv2d", "2d convolution", "convolution"],
        "notes": "Learned spatial cross-correlation over image-like tensors.",
    },
    "depthwise_conv2d": {
        "formula": "y[n, c, h, w] = sum(x[n, c] * kernel[c]) + bias[c]",
        "latex": r"y_{n,c,h,w} = x_{n,c} \ast K_c + b_c",
        "syntax": "nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels)",
        "functional": "F.conv2d(x, weight, groups=in_channels)",
        "aliases": ["depthwise_conv2d", "depthwise conv", "depthwise convolution"],
        "notes": "One spatial filter per input channel; groups equals in_channels.",
    },
    "dropout": {
        "formula": "y = mask * x / (1 - p) during training",
        "latex": r"y = \frac{m \odot x}{1-p},\quad m \sim \operatorname{Bernoulli}(1-p)",
        "syntax": "nn.Dropout(p=0.1)",
        "functional": "F.dropout(x, p=0.1, training=self.training)",
        "aliases": ["dropout"],
        "notes": "Randomly zeroes activations during training and is identity during evaluation.",
    },
    "residual_add": {
        "formula": "y = f(x) + x",
        "latex": r"y = f(x) + x",
        "syntax": None,
        "functional": "torch.add(x, residual)",
        "aliases": ["residual_add", "skip connection", "skip connections"],
        "notes": "The two tensors must have compatible shapes, often after a projection.",
    },
    "concat": {
        "formula": "y = concat(x_1, x_2, ..., x_n; dim)",
        "latex": r"y = \operatorname{concat}(x_1, x_2, \ldots, x_n; \mathrm{dim})",
        "syntax": None,
        "functional": "torch.cat((x, skip), dim=1)",
        "aliases": ["concat", "concatenate", "concatenation"],
        "notes": "All dimensions except the concatenation dimension must match.",
    },
    "flatten": {
        "formula": "y = reshape(x, (batch, -1))",
        "latex": r"y = \operatorname{reshape}(x, (\mathrm{batch}, -1))",
        "syntax": "nn.Flatten()",
        "functional": "torch.flatten(x, start_dim=1)",
        "aliases": ["flatten", "flattening"],
        "notes": "Typically preserves batch dimension and flattens feature dimensions.",
    },
    "globalavgpool2d": {
        "formula": "y[n, c] = mean_{h,w}(x[n, c, h, w])",
        "latex": r"y_{n,c} = \frac{1}{HW}\sum_{h,w}x_{n,c,h,w}",
        "syntax": "nn.AdaptiveAvgPool2d((1, 1))",
        "functional": "F.adaptive_avg_pool2d(x, (1, 1))",
        "aliases": ["globalavgpool2d", "global average pooling", "global avg pool"],
        "notes": "Reduces each channel's spatial map to one value.",
    },
    "patchembedding": {
        "formula": "tokens = flatten(conv2d(x, kernel=patch_size, stride=patch_size))",
        "latex": r"\mathrm{tokens} = \operatorname{flatten}(\operatorname{Conv2D}(x; k=s=\mathrm{patch\ size}))",
        "syntax": "nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)",
        "functional": "F.conv2d(x, weight, stride=patch_size)",
        "aliases": ["patchembedding", "patch embeddings"],
        "notes": "Convolutional projection that converts non-overlapping image patches into token embeddings.",
    },
}


def _normalise(value: str) -> str:
    return re.sub(r"[\s\-_]+", "", value.lower())


_ALIAS_INDEX: dict[str, str] = {
    _normalise(alias): canonical
    for canonical, entry in OPERATIONS.items()
    for alias in entry["aliases"]
}


def lookup(term: str) -> dict | None:
    """Resolve a free-form term to an entry, or return None for unknown terms."""
    if not isinstance(term, str):
        return None
    canonical = _ALIAS_INDEX.get(_normalise(term))
    return OPERATIONS.get(canonical) if canonical is not None else None


def find_mentioned(text: str) -> list[str]:
    """Return canonicals whose aliases occur as whole words, by first mention."""
    if not isinstance(text, str) or not text:
        return []

    mentions: list[tuple[int, str]] = []
    for canonical, entry in OPERATIONS.items():
        positions = [
            match.start()
            for alias in entry["aliases"]
            for match in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.IGNORECASE)
        ]
        if positions:
            mentions.append((min(positions), canonical))
    return [canonical for _, canonical in sorted(mentions)]
