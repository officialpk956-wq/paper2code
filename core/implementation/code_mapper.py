"""
core/implementation/code_mapper.py

Phase 10: Maps Paper2Code module types → structured PyTorch code views.

All code is generated from graph metadata — never hallucinated.
Every output is labeled with its implementation type:
  - "Educational Implementation": illustrative, not identical to paper
  - "Reference Implementation":  structurally faithful to paper design
  - "Pseudo Implementation":     conceptual scaffold only
"""

from typing import Any

# ── Master code library keyed by module_type ─────────────────────────────────

MODULE_CODE_LIBRARY: dict[str, dict[str, Any]] = {
    "conv2d": {
        "label": "Educational Implementation",
        "component": "nn.Conv2d",
        "concept": "Spatial Feature Extraction",
        "pseudocode": (
            "FOR each output position (h, w):\n"
            "  FOR each output channel c_out:\n"
            "    sum = 0\n"
            "    FOR each input channel c_in:\n"
            "      sum += dot(input[c_in, h:h+K, w:w+K], kernel[c_out, c_in])\n"
            "    output[c_out, h, w] = sum + bias[c_out]"
        ),
        "pytorch_template": (
            "import torch.nn as nn\n\n"
            "# Educational Implementation\n"
            "# Params derived from Paper2Code graph metadata\n"
            "conv = nn.Conv2d(\n"
            "    in_channels={in_channels},\n"
            "    out_channels={out_channels},\n"
            "    kernel_size={kernel_size},\n"
            "    stride={stride},\n"
            "    padding={padding},\n"
            "    bias=False  # Usually False when followed by BatchNorm\n"
            ")\n\n"
            "# Forward pass:\n"
            "# output = conv(x)  # shape: (B, {out_channels}, H', W')"
        ),
        "design_rationale": (
            "Conv2d uses shared kernel weights across spatial positions, enabling "
            "translation-equivariant feature detection. Setting bias=False is standard "
            "when followed by BatchNorm2d, which has its own learnable bias (beta)."
        ),
    },
    "residualblock": {
        "label": "Reference Implementation",
        "component": "Residual Block (He et al. 2016)",
        "concept": "Skip Connections & Gradient Flow",
        "pseudocode": (
            "FUNCTION ResidualBlock(x):\n"
            "  identity = x\n"
            "  out = Conv2d(x)\n"
            "  out = BatchNorm(out)\n"
            "  out = ReLU(out)\n"
            "  out = Conv2d(out)\n"
            "  out = BatchNorm(out)\n"
            "  IF shape(identity) != shape(out):\n"
            "    identity = Downsample(identity)  # 1x1 conv projection\n"
            "  out = out + identity  # <-- skip connection\n"
            "  out = ReLU(out)\n"
            "  RETURN out"
        ),
        "pytorch_template": (
            "import torch\nimport torch.nn as nn\n\n"
            "# Reference Implementation — ResNet Basic Block\n"
            "# He et al. 2016 — Deep Residual Learning for Image Recognition\n"
            "class ResidualBlock(nn.Module):\n"
            "    def __init__(self, in_channels, out_channels, stride=1):\n"
            "        super().__init__()\n"
            "        self.conv1 = nn.Conv2d(in_channels, out_channels,\n"
            "                               kernel_size=3, stride=stride, padding=1, bias=False)\n"
            "        self.bn1   = nn.BatchNorm2d(out_channels)\n"
            "        self.relu  = nn.ReLU(inplace=True)\n"
            "        self.conv2 = nn.Conv2d(out_channels, out_channels,\n"
            "                               kernel_size=3, stride=1, padding=1, bias=False)\n"
            "        self.bn2   = nn.BatchNorm2d(out_channels)\n\n"
            "        # Projection shortcut: needed when spatial/channel dims change\n"
            "        self.downsample = None\n"
            "        if stride != 1 or in_channels != out_channels:\n"
            "            self.downsample = nn.Sequential(\n"
            "                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),\n"
            "                nn.BatchNorm2d(out_channels)\n"
            "            )\n\n"
            "    def forward(self, x):\n"
            "        identity = x\n"
            "        out = self.relu(self.bn1(self.conv1(x)))\n"
            "        out = self.bn2(self.conv2(out))\n"
            "        if self.downsample:\n"
            "            identity = self.downsample(x)\n"
            "        out = out + identity  # Skip connection\n"
            "        return self.relu(out)\n"
        ),
        "design_rationale": (
            "The skip connection (out + identity) allows gradients to flow directly "
            "through the identity path during backprop, solving the vanishing gradient "
            "problem in very deep networks. The projection shortcut (1×1 conv) is only "
            "applied when spatial dimensions or channel counts change between input and output."
        ),
    },
    "bottleneckblock": {
        "label": "Reference Implementation",
        "component": "Bottleneck Block (ResNet-50+)",
        "concept": "Parameter-Efficient Deep Blocks",
        "pseudocode": (
            "FUNCTION Bottleneck(x):\n"
            "  identity = x\n"
            "  out = Conv1x1(x)      # Reduce channels: C → C/4\n"
            "  out = BN + ReLU\n"
            "  out = Conv3x3(out)    # Spatial convolution at reduced width\n"
            "  out = BN + ReLU\n"
            "  out = Conv1x1(out)    # Expand channels: C/4 → C\n"
            "  out = BN\n"
            "  out = out + identity  # Skip connection\n"
            "  RETURN ReLU(out)"
        ),
        "pytorch_template": (
            "# Reference Implementation — ResNet-50 Bottleneck Block\n"
            "class BottleneckBlock(nn.Module):\n"
            "    expansion = 4\n\n"
            "    def __init__(self, in_channels, width, stride=1):\n"
            "        super().__init__()\n"
            "        # 1x1 compress\n"
            "        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)\n"
            "        self.bn1   = nn.BatchNorm2d(width)\n"
            "        # 3x3 spatial\n"
            "        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1, bias=False)\n"
            "        self.bn2   = nn.BatchNorm2d(width)\n"
            "        # 1x1 expand\n"
            "        self.conv3 = nn.Conv2d(width, width * self.expansion, 1, bias=False)\n"
            "        self.bn3   = nn.BatchNorm2d(width * self.expansion)\n"
            "        self.relu  = nn.ReLU(inplace=True)\n"
            "        self.downsample = (\n"
            "            nn.Sequential(nn.Conv2d(in_channels, width * 4, 1, stride, bias=False),\n"
            "                          nn.BatchNorm2d(width * 4))\n"
            "            if stride != 1 or in_channels != width * 4 else None\n"
            "        )\n\n"
            "    def forward(self, x):\n"
            "        out = self.relu(self.bn1(self.conv1(x)))\n"
            "        out = self.relu(self.bn2(self.conv2(out)))\n"
            "        out = self.bn3(self.conv3(out))\n"
            "        identity = self.downsample(x) if self.downsample else x\n"
            "        return self.relu(out + identity)\n"
        ),
        "design_rationale": (
            "The bottleneck design (1×1→3×3→1×1) reduces the computational cost of the "
            "3×3 convolution by first compressing the channel count by 4×. This makes "
            "ResNet-50/101/152 feasible while maintaining accuracy. The expansion=4 ratio "
            "is a design choice from the original paper."
        ),
    },
    "denseblock": {
        "label": "Reference Implementation",
        "component": "Dense Block (Huang et al. 2017)",
        "concept": "Dense Feature Reuse",
        "pseudocode": (
            "FUNCTION DenseBlock(x, num_layers, growth_rate):\n"
            "  features = [x]\n"
            "  FOR i in range(num_layers):\n"
            "    x_i = Concat(features)       # Concatenate ALL previous outputs\n"
            "    new_feature = BN+ReLU+Conv(x_i)  # growth_rate new channels\n"
            "    features.append(new_feature)\n"
            "  RETURN Concat(features)"
        ),
        "pytorch_template": (
            "# Reference Implementation — DenseNet Dense Block\n"
            "# Huang et al. 2017 — Densely Connected Convolutional Networks\n"
            "class DenseLayer(nn.Module):\n"
            "    def __init__(self, in_channels, growth_rate):\n"
            "        super().__init__()\n"
            "        self.bn1   = nn.BatchNorm2d(in_channels)\n"
            "        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, 1, bias=False)  # bottleneck\n"
            "        self.bn2   = nn.BatchNorm2d(4 * growth_rate)\n"
            "        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)\n"
            "        self.relu  = nn.ReLU(inplace=True)\n\n"
            "    def forward(self, x):\n"
            "        out = self.conv1(self.relu(self.bn1(x)))\n"
            "        out = self.conv2(self.relu(self.bn2(out)))\n"
            "        return torch.cat([x, out], dim=1)  # Dense connection: concat, not add\n\n"
            "class DenseBlock(nn.Module):\n"
            "    def __init__(self, num_layers, in_channels, growth_rate):\n"
            "        super().__init__()\n"
            "        self.layers = nn.ModuleList()\n"
            "        for i in range(num_layers):\n"
            "            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))\n\n"
            "    def forward(self, x):\n"
            "        for layer in self.layers:\n"
            "            x = layer(x)  # Each layer receives all previous features\n"
            "        return x\n"
        ),
        "design_rationale": (
            "DenseNet uses torch.cat (concatenation) instead of addition for skip connections. "
            "This means every layer receives ALL preceding feature maps as input, maximizing "
            "feature reuse. The growth_rate controls how many new feature maps each layer adds. "
            "Transition layers (not shown) reduce feature map count between dense blocks."
        ),
    },
    "multiheadattention": {
        "label": "Reference Implementation",
        "component": "Multi-Head Self-Attention (Vaswani et al. 2017)",
        "concept": "Global Dependency Modeling",
        "pseudocode": (
            "FUNCTION MultiHeadAttention(x, num_heads, d_model):\n"
            "  d_k = d_model / num_heads\n"
            "  FOR each head h in range(num_heads):\n"
            "    Q_h = x @ W_Q_h          # (B, N, d_k)\n"
            "    K_h = x @ W_K_h          # (B, N, d_k)\n"
            "    V_h = x @ W_V_h          # (B, N, d_k)\n"
            "    scores = Q_h @ K_h.T / sqrt(d_k)  # Scaled dot-product\n"
            "    attn   = softmax(scores)           # (B, N, N) attention weights\n"
            "    head_h = attn @ V_h                # (B, N, d_k)\n"
            "  output = Concat(head_0...head_H) @ W_O\n"
            "  RETURN output"
        ),
        "pytorch_template": (
            "import torch\nimport torch.nn as nn\n\n"
            "# Reference Implementation — Multi-Head Self-Attention\n"
            "# Vaswani et al. 2017 — Attention Is All You Need\n"
            "class MultiHeadSelfAttention(nn.Module):\n"
            "    def __init__(self, d_model={d_model}, num_heads={num_heads}, dropout=0.1):\n"
            "        super().__init__()\n"
            "        assert d_model % num_heads == 0, 'embed_dim must be divisible by num_heads'\n"
            "        self.d_k    = d_model // num_heads\n"
            "        self.heads  = num_heads\n"
            "        # Fused QKV projection (efficient single matmul)\n"
            "        self.qkv    = nn.Linear(d_model, 3 * d_model, bias=False)\n"
            "        self.proj   = nn.Linear(d_model, d_model)\n"
            "        self.drop   = nn.Dropout(dropout)\n\n"
            "    def forward(self, x):\n"
            "        B, N, C = x.shape\n"
            "        # Split into Q, K, V then reshape for multi-head\n"
            "        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.d_k)\n"
            "        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d_k)\n"
            "        q, k, v = qkv.unbind(0)\n\n"
            "        # Scaled dot-product attention\n"
            "        scale  = self.d_k ** -0.5\n"
            "        attn   = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N)\n"
            "        attn   = self.drop(attn.softmax(dim=-1))\n"
            "        out    = (attn @ v).transpose(1, 2).reshape(B, N, C)\n"
            "        return self.proj(out)\n\n"
            "# Or use PyTorch built-in:\n"
            "# attn = nn.MultiheadAttention(embed_dim={d_model}, num_heads={num_heads}, batch_first=True)\n"
        ),
        "design_rationale": (
            "Multi-head attention splits the embedding into H independent heads, each learning "
            "different relationship patterns (syntactic, semantic, positional). The scaling by "
            "1/√d_k prevents the dot products from entering the saturation region of softmax. "
            "The fused QKV projection is an optimization over 3 separate linear layers."
        ),
    },
    "patchembedding": {
        "label": "Reference Implementation",
        "component": "Patch Embedding (Dosovitskiy et al. 2020)",
        "concept": "Image → Token Sequence Conversion",
        "pseudocode": (
            "FUNCTION PatchEmbed(image, patch_size, embed_dim):\n"
            "  # Divide image into non-overlapping patches\n"
            "  patches = split_into_patches(image, patch_size)  # (B, N, P*P*C)\n"
            "  N = (H / patch_size) * (W / patch_size)         # Number of patches\n"
            "  tokens = Linear(patches, embed_dim)              # (B, N, embed_dim)\n"
            "  cls_token = learnable_parameter()               # (1, 1, embed_dim)\n"
            "  tokens = Concat([cls_token, tokens], dim=1)     # (B, N+1, embed_dim)\n"
            "  pos_embed = learnable_parameter(N+1, embed_dim)\n"
            "  tokens = tokens + pos_embed                      # Add position info\n"
            "  RETURN tokens"
        ),
        "pytorch_template": (
            "import torch\nimport torch.nn as nn\n\n"
            "# Reference Implementation — ViT Patch Embedding\n"
            "# Dosovitskiy et al. 2020 — An Image is Worth 16x16 Words\n"
            "class PatchEmbedding(nn.Module):\n"
            "    def __init__(self, img_size=224, patch_size={patch_size},\n"
            "                 in_channels=3, embed_dim={embed_dim}):\n"
            "        super().__init__()\n"
            "        self.n_patches = (img_size // patch_size) ** 2\n"
            "        # Conv2d with kernel=patch_size, stride=patch_size = non-overlapping patch split\n"
            "        self.proj = nn.Conv2d(in_channels, embed_dim,\n"
            "                              kernel_size=patch_size, stride=patch_size)\n"
            "        # Learnable [CLS] token prepended to sequence\n"
            "        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))\n"
            "        # Learnable position embeddings (1D, one per patch + CLS)\n"
            "        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))\n\n"
            "    def forward(self, x):\n"
            "        B = x.shape[0]\n"
            "        # Project patches: (B, C, H, W) → (B, embed_dim, n, n) → (B, N, embed_dim)\n"
            "        x = self.proj(x).flatten(2).transpose(1, 2)\n"
            "        # Prepend CLS token\n"
            "        cls = self.cls_token.expand(B, -1, -1)\n"
            "        x = torch.cat([cls, x], dim=1)\n"
            "        # Add positional embedding\n"
            "        return x + self.pos_embed\n"
        ),
        "design_rationale": (
            "Using Conv2d(kernel_size=patch_size, stride=patch_size) is mathematically equivalent "
            "to splitting the image into patches and applying a linear projection, but is more "
            "efficient as a single CUDA kernel call. The [CLS] token aggregates global information "
            "for classification. 1D positional embeddings are learned, not fixed sinusoidal."
        ),
    },
    "transformerblock": {
        "label": "Reference Implementation",
        "component": "Transformer Encoder Layer",
        "concept": "Pre-Norm Self-Attention + FFN",
        "pseudocode": (
            "FUNCTION TransformerBlock(x):\n"
            "  # Pre-norm variant (ViT / modern Transformers)\n"
            "  x = x + MHSA(LayerNorm(x))       # Self-attention sublayer\n"
            "  x = x + FFN(LayerNorm(x))         # Feed-forward sublayer\n"
            "  RETURN x\n\n"
            "FUNCTION FFN(x, d_model, d_ff=4*d_model):\n"
            "  x = Linear(x, d_ff)\n"
            "  x = GELU(x)\n"
            "  x = Linear(x, d_model)\n"
            "  RETURN x"
        ),
        "pytorch_template": (
            "import torch.nn as nn\n\n"
            "# Reference Implementation — Transformer Encoder Block (Pre-Norm)\n"
            "class TransformerBlock(nn.Module):\n"
            "    def __init__(self, d_model={d_model}, num_heads={num_heads},\n"
            "                 mlp_ratio=4.0, dropout=0.1):\n"
            "        super().__init__()\n"
            "        self.norm1 = nn.LayerNorm(d_model)\n"
            "        self.attn  = nn.MultiheadAttention(\n"
            "            embed_dim=d_model, num_heads=num_heads,\n"
            "            dropout=dropout, batch_first=True\n"
            "        )\n"
            "        self.norm2 = nn.LayerNorm(d_model)\n"
            "        d_ff = int(d_model * mlp_ratio)\n"
            "        self.ffn = nn.Sequential(\n"
            "            nn.Linear(d_model, d_ff),\n"
            "            nn.GELU(),           # GELU instead of ReLU in modern Transformers\n"
            "            nn.Dropout(dropout),\n"
            "            nn.Linear(d_ff, d_model),\n"
            "            nn.Dropout(dropout),\n"
            "        )\n\n"
            "    def forward(self, x):\n"
            "        # Pre-norm: normalize BEFORE attention (more stable than post-norm)\n"
            "        normed = self.norm1(x)\n"
            "        x = x + self.attn(normed, normed, normed)[0]  # Self-attention\n"
            "        x = x + self.ffn(self.norm2(x))               # Feed-forward\n"
            "        return x\n"
        ),
        "design_rationale": (
            "Pre-LayerNorm (normalize before sublayer) is more training-stable than the "
            "original post-norm formulation and is used in ViT, GPT, and most modern "
            "Transformers. GELU activation outperforms ReLU for language/vision tasks. "
            "The FFN dimension is typically 4× the embedding dimension (mlp_ratio=4)."
        ),
    },
    "upsample": {
        "label": "Educational Implementation",
        "component": "Bilinear Upsample + Conv (U-Net Decoder)",
        "concept": "Spatial Resolution Recovery",
        "pseudocode": (
            "FUNCTION DecoderBlock(x, skip_features):\n"
            "  x = Upsample(x, scale=2)              # 2× spatial resolution\n"
            "  x = Concat([x, skip_features], dim=1) # Append encoder skip features\n"
            "  x = Conv3x3(x)                        # Fuse features\n"
            "  x = BN + ReLU\n"
            "  x = Conv3x3(x)\n"
            "  x = BN + ReLU\n"
            "  RETURN x"
        ),
        "pytorch_template": (
            "import torch\nimport torch.nn as nn\n\n"
            "# Educational Implementation — U-Net Decoder Block\n"
            "class UNetDecoderBlock(nn.Module):\n"
            "    def __init__(self, in_channels, skip_channels, out_channels):\n"
            "        super().__init__()\n"
            "        # Bilinear upsample (no learnable params, avoids checkerboard artifacts)\n"
            "        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)\n"
            "        # After concat with skip: channels = in_channels + skip_channels\n"
            "        self.conv = nn.Sequential(\n"
            "            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),\n"
            "            nn.BatchNorm2d(out_channels),\n"
            "            nn.ReLU(inplace=True),\n"
            "            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),\n"
            "            nn.BatchNorm2d(out_channels),\n"
            "            nn.ReLU(inplace=True),\n"
            "        )\n\n"
            "    def forward(self, x, skip):\n"
            "        x = self.up(x)               # Double spatial resolution\n"
            "        x = torch.cat([x, skip], 1)  # Concatenate encoder skip features\n"
            "        return self.conv(x)\n"
        ),
        "design_rationale": (
            "Bilinear upsampling followed by convolution (the 'resize-convolution' pattern) "
            "avoids the checkerboard artifacts caused by transposed convolutions. The skip "
            "connection concatenation is what distinguishes U-Net from a plain encoder-decoder — "
            "it passes fine-grained spatial detail from the encoder directly to the decoder."
        ),
    },
    "linear": {
        "label": "Educational Implementation",
        "component": "nn.Linear (Fully Connected Layer)",
        "concept": "Global Feature Aggregation / Classification Head",
        "pseudocode": (
            "FUNCTION Linear(x, W, b):\n"
            "  # x: (B, in_features)\n"
            "  # W: (out_features, in_features)\n"
            "  output = x @ W.T + b   # Matrix multiply + bias\n"
            "  RETURN output           # (B, out_features)"
        ),
        "pytorch_template": (
            "import torch.nn as nn\n\n"
            "# Educational Implementation — Classification Head\n"
            "class ClassificationHead(nn.Module):\n"
            "    def __init__(self, in_features={in_features}, num_classes={num_classes}, dropout=0.0):\n"
            "        super().__init__()\n"
            "        self.head = nn.Sequential(\n"
            "            nn.AdaptiveAvgPool2d((1, 1)) if in_features > 2048 else nn.Identity(),\n"
            "            nn.Flatten(),\n"
            "            nn.Dropout(dropout),\n"
            "            nn.Linear(in_features, num_classes)\n"
            "        )\n\n"
            "    def forward(self, x):\n"
            "        return self.head(x)\n\n"
            "# Note: No softmax here — CrossEntropyLoss applies log-softmax internally\n"
        ),
        "design_rationale": (
            "The classification head does not include softmax because nn.CrossEntropyLoss "
            "applies log-softmax+NLL internally for numerical stability. AdaptiveAvgPool2d "
            "is used before the linear layer for CNN heads to handle variable input sizes."
        ),
    },
    "batchnorm2d": {
        "label": "Educational Implementation",
        "component": "nn.BatchNorm2d",
        "concept": "Training Stabilization",
        "pseudocode": (
            "FUNCTION BatchNorm(x, gamma, beta, eps=1e-5):\n"
            "  # Per-channel statistics over (B, H, W)\n"
            "  mu    = mean(x, dims=[B, H, W])\n"
            "  sigma = std(x,  dims=[B, H, W])\n"
            "  x_hat = (x - mu) / sqrt(sigma^2 + eps)\n"
            "  RETURN gamma * x_hat + beta  # Learnable scale + shift"
        ),
        "pytorch_template": (
            "# Educational Implementation — BatchNorm2d\n"
            "# Ioffe & Szegedy 2015 — Batch Normalization\n"
            "bn = nn.BatchNorm2d(\n"
            "    num_features={channels},  # Must match conv output channels\n"
            "    eps=1e-5,                 # Numerical stability\n"
            "    momentum=0.1,             # Running mean/var update rate\n"
            "    affine=True,              # Learn gamma (scale) and beta (shift)\n"
            "    track_running_stats=True  # Maintain running stats for inference\n"
            ")\n\n"
            "# During training: normalize over (B, H, W) per channel\n"
            "# During eval:     use running mean/var accumulated during training\n"
            "# Always call model.train() / model.eval() correctly!"
        ),
        "design_rationale": (
            "BatchNorm normalizes activations to zero mean and unit variance per channel, "
            "enabling higher learning rates and reducing sensitivity to weight initialization. "
            "The learnable gamma/beta allow the network to recover any normalization it needs. "
            "At inference, running statistics (not batch statistics) are used."
        ),
    },
    "layernorm": {
        "label": "Educational Implementation",
        "component": "nn.LayerNorm",
        "concept": "Transformer Normalization",
        "pseudocode": (
            "FUNCTION LayerNorm(x, gamma, beta, eps=1e-5):\n"
            "  # Normalize over the LAST dimension (feature/channel)\n"
            "  # Unlike BatchNorm which normalizes over batch\n"
            "  mu    = mean(x, dim=-1, keepdim=True)\n"
            "  sigma = std(x,  dim=-1, keepdim=True)\n"
            "  x_hat = (x - mu) / sqrt(sigma^2 + eps)\n"
            "  RETURN gamma * x_hat + beta"
        ),
        "pytorch_template": (
            "# Educational Implementation — LayerNorm\n"
            "# Used in Transformers (not CNNs)\n"
            "ln = nn.LayerNorm(\n"
            "    normalized_shape={embed_dim},  # Normalize over last dim\n"
            "    eps=1e-6,                      # ViT uses 1e-6 for stability\n"
            "    elementwise_affine=True        # Learnable gamma and beta\n"
            ")\n\n"
            "# Key difference from BatchNorm:\n"
            "# BatchNorm: normalize over (B, H, W) — batch-dependent\n"
            "# LayerNorm: normalize over (features) — batch-independent\n"
            "# → LayerNorm works with batch_size=1; BatchNorm needs B > 1\n"
        ),
        "design_rationale": (
            "LayerNorm is preferred over BatchNorm in Transformers because it normalizes "
            "over the feature dimension (not batch), making it batch-size independent. "
            "This is critical for autoregressive generation where batch_size=1 is common. "
            "Pre-LayerNorm (before sublayer) is more stable than post-LayerNorm."
        ),
    },
}


def get_module_implementation(
    module_type: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return a structured implementation view for a given module type.

    Args:
        module_type: Canonical type string (e.g. "conv2d", "residualblock")
        params:      Optional parameter dict from graph node to fill templates

    Returns:
        Dict with label, component, concept, pseudocode, pytorch_code, design_rationale
    """
    p = params or {}
    key = (module_type or "").lower().replace(" ", "").replace("_", "").replace("-", "")

    # Try exact key first, then fuzzy matching
    entry = MODULE_CODE_LIBRARY.get(key)
    if not entry:
        for k, v in MODULE_CODE_LIBRARY.items():
            if k in key or key in k:
                entry = v
                break

    if not entry:
        return {
            "label": "Pseudo Implementation",
            "component": f"{module_type} Block",
            "concept": "Custom Module",
            "pseudocode": f"# {module_type} — custom implementation\n# No standard template available",
            "pytorch_code": (
                f"# Pseudo Implementation — {module_type}\n"
                f"# This module type does not have a standard template.\n"
                f"# Refer to the Paper2Code graph metadata for structural details."
            ),
            "design_rationale": "This module type does not have a standard educational implementation template.",
        }

    # Fill template parameters
    template = entry["pytorch_template"]
    fills = {
        "in_channels": p.get("in_channels", p.get("channels", 64)),
        "out_channels": p.get("out_channels", p.get("channels", 64)),
        "kernel_size": p.get("kernel_size", 3),
        "stride": p.get("stride", 1),
        "padding": p.get("padding", 1),
        "d_model": p.get("hidden_size", p.get("embed_dim", 768)),
        "num_heads": p.get("num_heads", p.get("heads", 12)),
        "patch_size": p.get("patch_size", 16),
        "embed_dim": p.get("embed_dim", p.get("hidden_size", 768)),
        "in_features": p.get("in_features", 2048),
        "num_classes": p.get("num_classes", 1000),
        "channels": p.get("channels", 64),
    }
    try:
        pytorch_code = template.format(**fills)
    except (KeyError, ValueError):
        pytorch_code = template  # Fall back to unformatted if params mismatch

    return {
        "label": entry["label"],
        "component": entry["component"],
        "concept": entry["concept"],
        "pseudocode": entry["pseudocode"],
        "pytorch_code": pytorch_code,
        "design_rationale": entry["design_rationale"],
    }


def get_architecture_implementation(
    paper_title: str,
    classification: str,
    modules: list,
) -> dict[str, Any]:
    """
    Build full architecture implementation view for a paper.

    Args:
        paper_title:    Paper title string
        classification: Architecture classification (ResNet, ViT, etc.)
        modules:        List of module dicts from /api/papers/{id}

    Returns:
        Dict with per-module implementation views + architecture summary
    """
    module_implementations = []
    for m in modules:
        impl = get_module_implementation(
            m.get("module_type", ""),
            m.get("graph_nodes", [{}])[0].get("params", {}) if m.get("graph_nodes") else {},
        )
        module_implementations.append(
            {
                "module_id": m.get("id"),
                "layer_name": m.get("layer_name"),
                "module_type": m.get("module_type"),
                "implementation": impl,
            }
        )

    return {
        "paper_title": paper_title,
        "classification": classification,
        "label": "Educational Implementation — Paper2Code generated from graph metadata",
        "modules": module_implementations,
        "safety_notice": (
            "All code shown is an educational reference generated from Paper2Code's "
            "architecture graph. It illustrates the structural design described in the paper "
            "but is not identical to the authors' original implementation."
        ),
    }
