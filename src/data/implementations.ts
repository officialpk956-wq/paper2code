import type { loader } from '@monaco-editor/react';

export type ImplementStep = {
  id: string;
  number: number;
  title: string;
  description: string;
  concepts: string[];
  starterCode: string;
  testCode: string;
  hints: string[];
};

export type PaperImpl = {
  paperId: string;
  paperTitle: string;
  totalSteps: number;
  steps: ImplementStep[];
};

export const IMPLEMENTATIONS: PaperImpl[] = [
  {
    paperId: "attention-is-all-you-need",
    paperTitle: "Attention Is All You Need",
    totalSteps: 5,
    steps: [
      {
        id: "attn-s1",
        number: 1,
        title: "Scaled Dot-Product Attention",
        description: "Implement the core attention formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V. Scale by sqrt(d_k) to prevent vanishing gradients when d_k is large.",
        concepts: ["Scaled dot-product", "Softmax temperature", "Query-Key-Value"],
        starterCode: "import torch\n\nimport torch.nn.functional as F\n\n\n\ndef scaled_dot_product_attention(Q, K, V):\n\n    \"\"\"\n\n    Args:\n\n        Q: (batch, seq_q, d_k)\n\n        K: (batch, seq_k, d_k)\n\n        V: (batch, seq_k, d_v)\n\n    Returns:\n\n        output: (batch, seq_q, d_v)\n\n    \"\"\"\n\n    d_k = Q.size(-1)\n\n    # TODO: compute scores = Q @ K.transpose(-2, -1)\n\n    # TODO: scale by sqrt(d_k)\n\n    # TODO: apply softmax over last dim\n\n    # TODO: return weights @ V\n\n    pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    torch.manual_seed(0)\n\n    Q = torch.randn(1, 4, 8)\n\n    K = torch.randn(1, 4, 8)\n\n    V = torch.randn(1, 4, 8)\n\n    out = scaled_dot_product_attention(Q, K, V)\n\n    assert out is not None, \"Function returned None\"\n\n    assert out.shape == (1, 4, 8), f\"Expected shape (1,4,8), got {out.shape}\"\n\n    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)\n\n    assert torch.allclose(out, ref, atol=1e-5), \"Output does not match reference\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["Transpose K with .transpose(-2, -1)", "Scale before softmax: divide by d_k ** 0.5", "F.softmax(scores, dim=-1)"]
      },
      {
        id: "attn-s2",
        number: 2,
        title: "Multi-Head Attention",
        description: "Project Q, K, V into h parallel heads using linear layers, run scaled dot-product attention on each head, then concatenate and project the results.",
        concepts: ["Parallel attention heads", "Linear projection", "Head split/concat"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\nimport torch.nn.functional as F\n\n\n\ndef scaled_dot_product_attention(Q, K, V):\n\n    d_k = Q.size(-1)\n\n    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)\n\n    weights = F.softmax(scores, dim=-1)\n\n    return weights @ V\n\n\n\nclass MultiHeadAttention(nn.Module):\n\n    def __init__(self, d_model: int, num_heads: int):\n\n        super().__init__()\n\n        assert d_model % num_heads == 0\n\n        self.num_heads = num_heads\n\n        self.d_k = d_model // num_heads\n\n        # TODO: define W_q, W_k, W_v, W_o as nn.Linear(d_model, d_model)\n\n\n\n    def forward(self, x):\n\n        # x: (batch, seq, d_model)\n\n        B, S, _ = x.shape\n\n        # TODO: project Q, K, V\n\n        # TODO: reshape to (B, num_heads, S, d_k)\n\n        # TODO: run scaled_dot_product_attention\n\n        # TODO: reshape back to (B, S, d_model) and apply W_o\n\n        pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    torch.manual_seed(0)\n\n    mha = MultiHeadAttention(d_model=32, num_heads=4)\n\n    x = torch.randn(1, 4, 32)\n\n    out = mha(x)\n\n    assert out is not None, \"forward returned None\"\n\n    assert out.shape == (1, 4, 32), f\"Expected (1,4,32), got {out.shape}\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["Reshape with .view(B, S, self.num_heads, self.d_k).transpose(1,2) to get (B,heads,S,d_k)", "After attention, transpose back and .contiguous().view(B, S, d_model)", "Apply W_o to the concatenated result"]
      },
      {
        id: "attn-s3",
        number: 3,
        title: "Positional Encoding",
        description: "Add sinusoidal positional encodings so the model knows token order. PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(pos/10000^(2i/d)).",
        concepts: ["Sinusoidal encoding", "Position-aware representation"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\nimport math\n\n\n\nclass PositionalEncoding(nn.Module):\n\n    def __init__(self, d_model: int, max_len: int = 5000):\n\n        super().__init__()\n\n        # TODO: create pe tensor of shape (1, max_len, d_model)\n\n        # TODO: fill even dims with sin, odd dims with cos\n\n        # self.register_buffer('pe', pe)\n\n\n\n    def forward(self, x):\n\n        # x: (batch, seq, d_model)\n\n        # TODO: add self.pe[:, :x.size(1)] to x and return\n\n        pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    pe = PositionalEncoding(d_model=16)\n\n    x = torch.zeros(1, 10, 16)\n\n    out = pe(x)\n\n    assert out is not None, \"forward returned None\"\n\n    assert out.shape == (1, 10, 16), f\"Expected (1,10,16), got {out.shape}\"\n\n    assert not torch.allclose(out, x), \"Positional encoding had no effect\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: []
      },
      {
        id: "attn-s4",
        number: 4,
        title: "Encoder Block",
        description: "Stack MHA \u2192 Add+Norm \u2192 Feed-Forward (linear\u2192ReLU\u2192linear) \u2192 Add+Norm. This is one Transformer encoder layer.",
        concepts: ["Residual connections", "Layer normalization", "Position-wise FFN"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\nimport torch.nn.functional as F\n\n\n\n# Paste your MultiHeadAttention here or just import it\n\nclass MultiHeadAttention(nn.Module):\n\n    def __init__(self, d_model, num_heads):\n\n        super().__init__()\n\n        self.num_heads = num_heads\n\n        self.d_k = d_model // num_heads\n\n        self.W_q = nn.Linear(d_model, d_model)\n\n        self.W_k = nn.Linear(d_model, d_model)\n\n        self.W_v = nn.Linear(d_model, d_model)\n\n        self.W_o = nn.Linear(d_model, d_model)\n\n    def forward(self, x):\n\n        B, S, d = x.shape\n\n        Q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1,2)\n\n        K = self.W_k(x).view(B, S, self.num_heads, self.d_k).transpose(1,2)\n\n        V = self.W_v(x).view(B, S, self.num_heads, self.d_k).transpose(1,2)\n\n        scores = F.softmax(Q @ K.transpose(-2,-1) / (self.d_k**0.5), dim=-1)\n\n        out = (scores @ V).transpose(1,2).contiguous().view(B, S, d)\n\n        return self.W_o(out)\n\n\n\nclass EncoderBlock(nn.Module):\n\n    def __init__(self, d_model: int = 32, num_heads: int = 4, d_ff: int = 128):\n\n        super().__init__()\n\n        self.attn = MultiHeadAttention(d_model, num_heads)\n\n        self.norm1 = nn.LayerNorm(d_model)\n\n        self.ff1 = nn.Linear(d_model, d_ff)\n\n        self.ff2 = nn.Linear(d_ff, d_model)\n\n        self.norm2 = nn.LayerNorm(d_model)\n\n\n\n    def forward(self, x):\n\n        # TODO: x = self.norm1(x + self.attn(x))\n\n        # TODO: x = self.norm2(x + self.ff2(F.relu(self.ff1(x))))\n\n        # TODO: return x\n\n        pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    enc = EncoderBlock(d_model=32, num_heads=4, d_ff=128)\n\n    x = torch.randn(1, 4, 32)\n\n    out = enc(x)\n\n    assert out is not None, \"forward returned None\"\n\n    assert out.shape == (1, 4, 32), f\"Expected (1,4,32), got {out.shape}\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["Apply attention first: attn_out = self.attn(x), then x = self.norm1(x + attn_out)", "FFN: ff_out = self.ff2(F.relu(self.ff1(x))), then x = self.norm2(x + ff_out)", "Return x at the end"]
      },
      {
        id: "attn-s5",
        number: 5,
        title: "Transformer Encoder Stack",
        description: "Stack N EncoderBlocks. The full Transformer encoder is just these blocks in sequence with a final linear layer for classification.",
        concepts: ["Stacking layers", "nn.ModuleList", "Sequence transduction"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\nimport torch.nn.functional as F\n\n\n\n# (paste EncoderBlock from step 4 above)\n\nclass EncoderBlock(nn.Module):\n\n    def __init__(self, d_model=32, num_heads=4, d_ff=128):\n\n        super().__init__()\n\n        import torch.nn as nn\n\n        self.attn_q = nn.Linear(d_model, d_model)\n\n        self.attn_k = nn.Linear(d_model, d_model)\n\n        self.attn_v = nn.Linear(d_model, d_model)\n\n        self.attn_o = nn.Linear(d_model, d_model)\n\n        self.norm1 = nn.LayerNorm(d_model)\n\n        self.ff1 = nn.Linear(d_model, d_ff)\n\n        self.ff2 = nn.Linear(d_ff, d_model)\n\n        self.norm2 = nn.LayerNorm(d_model)\n\n        self.num_heads = num_heads\n\n        self.d_k = d_model // num_heads\n\n    def forward(self, x):\n\n        B, S, d = x.shape\n\n        Q = self.attn_q(x).view(B,S,self.num_heads,self.d_k).transpose(1,2)\n\n        K = self.attn_k(x).view(B,S,self.num_heads,self.d_k).transpose(1,2)\n\n        V = self.attn_v(x).view(B,S,self.num_heads,self.d_k).transpose(1,2)\n\n        w = F.softmax(Q@K.transpose(-2,-1)/self.d_k**0.5,dim=-1)\n\n        a = (w@V).transpose(1,2).contiguous().view(B,S,d)\n\n        x = self.norm1(x + self.attn_o(a))\n\n        x = self.norm2(x + self.ff2(F.relu(self.ff1(x))))\n\n        return x\n\n\n\nclass TransformerEncoder(nn.Module):\n\n    def __init__(self, d_model: int = 32, num_heads: int = 4, num_layers: int = 2, num_classes: int = 10):\n\n        super().__init__()\n\n        # TODO: define self.layers as nn.ModuleList of num_layers EncoderBlocks\n\n        # TODO: define self.fc as nn.Linear(d_model, num_classes)\n\n\n\n    def forward(self, x):\n\n        # x: (batch, seq, d_model)\n\n        # TODO: pass x through each layer in self.layers\n\n        # TODO: mean-pool over seq dim: x = x.mean(dim=1)\n\n        # TODO: return self.fc(x)\n\n        pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    model = TransformerEncoder(d_model=32, num_heads=4, num_layers=2, num_classes=10)\n\n    x = torch.randn(1, 4, 32)\n\n    out = model(x)\n\n    assert out is not None, \"forward returned None\"\n\n    assert out.shape == (1, 10), f\"Expected (1,10), got {out.shape}\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: []
      },
    ]
  },
  {
    paperId: "resnet",
    paperTitle: "ResNet",
    totalSteps: 4,
    steps: [
      {
        id: "resnet-s1",
        number: 1,
        title: "Residual Block",
        description: "Implement the residual block: F(x) + x. Two conv3\u00d73 layers with BatchNorm and ReLU, plus the identity shortcut that enables gradient flow through 100+ layers.",
        concepts: ["Skip connections", "Vanishing gradient solution", "Identity shortcut"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\nimport torch.nn.functional as F\n\n\n\nclass ResidualBlock(nn.Module):\n\n    def __init__(self, channels: int):\n\n        super().__init__()\n\n        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)\n\n        self.bn1 = nn.BatchNorm2d(channels)\n\n        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)\n\n        self.bn2 = nn.BatchNorm2d(channels)\n\n\n\n    def forward(self, x):\n\n        # TODO: shortcut = x\n\n        # TODO: out = relu(bn1(conv1(x)))\n\n        # TODO: out = bn2(conv2(out))\n\n        # TODO: out = relu(out + shortcut)\n\n        # TODO: return out\n\n        pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    block = ResidualBlock(channels=64)\n\n    x = torch.randn(1, 64, 56, 56)\n\n    out = block(x)\n\n    assert out is not None, \"forward returned None\"\n\n    assert out.shape == (1, 64, 56, 56), f\"Expected (1,64,56,56), got {out.shape}\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["Save shortcut = x before any transformation", "Apply conv\u2192bn\u2192relu on the first pair, conv\u2192bn on the second", "Add shortcut BEFORE the final relu: F.relu(out + shortcut)"]
      },
      {
        id: "resnet-s2",
        number: 2,
        title: "Residual Block with Downsampling",
        description: "When stride=2 or channels change, the shortcut dimensions won't match. Use a 1\u00d71 conv (the 'projection shortcut') to resize the identity path.",
        concepts: ["Projection shortcut", "1\u00d71 convolution", "Dimensionality matching"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\nimport torch.nn.functional as F\n\n\n\nclass ResidualBlockDown(nn.Module):\n\n    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):\n\n        super().__init__()\n\n        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)\n\n        self.bn1 = nn.BatchNorm2d(out_channels)\n\n        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)\n\n        self.bn2 = nn.BatchNorm2d(out_channels)\n\n        # TODO: define self.downsample as nn.Sequential(Conv2d 1\u00d71, BN)\n\n        # only when stride != 1 or in_channels != out_channels\n\n\n\n    def forward(self, x):\n\n        shortcut = x\n\n        out = F.relu(self.bn1(self.conv1(x)))\n\n        out = self.bn2(self.conv2(out))\n\n        # TODO: apply self.downsample to shortcut if it exists\n\n        return F.relu(out + shortcut)",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    block = ResidualBlockDown(in_channels=64, out_channels=128, stride=2)\n\n    x = torch.randn(1, 64, 56, 56)\n\n    out = block(x)\n\n    assert out is not None\n\n    assert out.shape == (1, 128, 28, 28), f\"Expected (1,128,28,28), got {out.shape}\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["In __init__: if stride != 1 or in_channels != out_channels: self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))", "In forward: if self.downsample is not None: shortcut = self.downsample(x)", "Always initialize self.downsample = None so the check is safe"]
      },
      {
        id: "resnet-s3",
        number: 3,
        title: "ResNet Layer",
        description: "A ResNet layer is a sequence of blocks: the first block may downsample, then N-1 identical blocks at the new resolution.",
        concepts: ["Layer composition", "Sequential blocks"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\nimport torch.nn.functional as F\n\n\n\n# Paste ResidualBlockDown from step 2\n\nclass ResidualBlockDown(nn.Module):\n\n    def __init__(self, in_channels, out_channels, stride=1):\n\n        super().__init__()\n\n        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)\n\n        self.bn1 = nn.BatchNorm2d(out_channels)\n\n        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)\n\n        self.bn2 = nn.BatchNorm2d(out_channels)\n\n        self.downsample = None\n\n        if stride != 1 or in_channels != out_channels:\n\n            self.downsample = nn.Sequential(\n\n                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),\n\n                nn.BatchNorm2d(out_channels),\n\n            )\n\n    def forward(self, x):\n\n        shortcut = x\n\n        out = F.relu(self.bn1(self.conv1(x)))\n\n        out = self.bn2(self.conv2(out))\n\n        if self.downsample is not None:\n\n            shortcut = self.downsample(x)\n\n        return F.relu(out + shortcut)\n\n\n\ndef make_layer(in_channels: int, out_channels: int, num_blocks: int, stride: int = 1):\n\n    # TODO: create first block with stride (handles downsampling)\n\n    # TODO: create remaining (num_blocks - 1) blocks with stride=1, in_channels=out_channels\n\n    # TODO: return nn.Sequential(*blocks)\n\n    pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    layer = make_layer(64, 128, num_blocks=2, stride=2)\n\n    assert layer is not None\n\n    x = torch.randn(1, 64, 56, 56)\n\n    out = layer(x)\n\n    assert out.shape == (1, 128, 28, 28), f\"Expected (1,128,28,28), got {out.shape}\"\n\n    assert len(list(layer.children())) == 2, \"Should have exactly 2 blocks\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: []
      },
      {
        id: "resnet-s4",
        number: 4,
        title: "ResNet-18",
        description: "Assemble the full ResNet-18: conv1(7\u00d77,64,stride=2)\u2192MaxPool\u21924 layers(64,128,256,512)\u2192AdaptiveAvgPool\u2192FC(1000).",
        concepts: ["Full architecture assembly", "Global average pooling"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\nimport torch.nn.functional as F\n\n\n\n# (make_layer + ResidualBlockDown from step 3 assumed available)\n\nclass ResidualBlockDown(nn.Module):\n\n    def __init__(self, in_channels, out_channels, stride=1):\n\n        super().__init__()\n\n        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)\n\n        self.bn1 = nn.BatchNorm2d(out_channels)\n\n        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)\n\n        self.bn2 = nn.BatchNorm2d(out_channels)\n\n        self.downsample = None\n\n        if stride != 1 or in_channels != out_channels:\n\n            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))\n\n    def forward(self, x):\n\n        s = x\n\n        out = F.relu(self.bn1(self.conv1(x)))\n\n        out = self.bn2(self.conv2(out))\n\n        if self.downsample: s = self.downsample(x)\n\n        return F.relu(out + s)\n\n\n\ndef make_layer(ic, oc, n, stride=1):\n\n    b = [ResidualBlockDown(ic, oc, stride)]\n\n    for _ in range(n-1): b.append(ResidualBlockDown(oc, oc))\n\n    return nn.Sequential(*b)\n\n\n\nclass ResNet18(nn.Module):\n\n    def __init__(self, num_classes: int = 1000):\n\n        super().__init__()\n\n        # TODO: conv1 = Conv2d(3, 64, 7, stride=2, padding=3, bias=False)\n\n        # TODO: bn1, relu, maxpool\n\n        # TODO: layer1 = make_layer(64, 64, 2, stride=1)\n\n        # TODO: layer2 = make_layer(64, 128, 2, stride=2)\n\n        # TODO: layer3 = make_layer(128, 256, 2, stride=2)\n\n        # TODO: layer4 = make_layer(256, 512, 2, stride=2)\n\n        # TODO: avgpool = AdaptiveAvgPool2d((1,1))\n\n        # TODO: fc = Linear(512, num_classes)\n\n\n\n    def forward(self, x):\n\n        # TODO: pass through conv1\u2192bn1\u2192relu\u2192maxpool\n\n        # TODO: layer1\u2192layer2\u2192layer3\u2192layer4\n\n        # TODO: avgpool, flatten, fc\n\n        pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    model = ResNet18(num_classes=1000)\n\n    x = torch.randn(1, 3, 224, 224)\n\n    out = model(x)\n\n    assert out is not None\n\n    assert out.shape == (1, 1000), f\"Expected (1,1000), got {out.shape}\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["Conv1: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)", "MaxPool: nn.MaxPool2d(kernel_size=3, stride=2, padding=1)", "In forward: x = self.avgpool(x); x = torch.flatten(x, 1); return self.fc(x)"]
      },
    ]
  },
  {
    paperId: "vit",
    paperTitle: "ViT",
    totalSteps: 4,
    steps: [
      {
        id: "vit-s1",
        number: 1,
        title: "Patch Embedding",
        description: "Split an image into non-overlapping 16\u00d716 patches. Use a Conv2d with kernel_size=patch_size, stride=patch_size \u2014 each patch becomes a d_model-dimensional token.",
        concepts: ["Patch tokenization", "Non-overlapping convolution"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\n\n\nclass PatchEmbedding(nn.Module):\n\n    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, d_model: int = 768):\n\n        super().__init__()\n\n        self.num_patches = (img_size // patch_size) ** 2\n\n        # TODO: define self.proj as Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)\n\n\n\n    def forward(self, x):\n\n        # x: (B, C, H, W)\n\n        # TODO: apply self.proj \u2192 (B, d_model, H/P, W/P)\n\n        # TODO: flatten spatial dims \u2192 (B, d_model, num_patches)\n\n        # TODO: transpose \u2192 (B, num_patches, d_model)\n\n        pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    pe = PatchEmbedding(img_size=224, patch_size=16, in_channels=3, d_model=768)\n\n    x = torch.randn(1, 3, 224, 224)\n\n    out = pe(x)\n\n    assert out is not None\n\n    assert out.shape == (1, 196, 768), f\"Expected (1,196,768), got {out.shape}\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["After proj: shape is (B, d_model, 14, 14)", "Flatten with .flatten(2) \u2192 (B, d_model, 196)", "Transpose with .transpose(1, 2) \u2192 (B, 196, d_model)"]
      },
      {
        id: "vit-s2",
        number: 2,
        title: "CLS Token + Position Embedding",
        description: "Prepend a learnable [CLS] token to the patch sequence. Add learned positional embeddings to the full sequence (CLS + patches).",
        concepts: ["CLS classification token", "Learned positional embeddings", "nn.Parameter"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\n\n\nclass PatchEmbedding(nn.Module):\n\n    def __init__(self, img_size=224, patch_size=16, in_channels=3, d_model=768):\n\n        super().__init__()\n\n        self.num_patches = (img_size // patch_size) ** 2\n\n        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)\n\n    def forward(self, x):\n\n        x = self.proj(x).flatten(2).transpose(1, 2)\n\n        return x\n\n\n\nclass ViTEmbedding(nn.Module):\n\n    def __init__(self, img_size: int = 224, patch_size: int = 16, d_model: int = 768):\n\n        super().__init__()\n\n        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, d_model)\n\n        num_patches = (img_size // patch_size) ** 2\n\n        # TODO: define self.cls_token as nn.Parameter of shape (1, 1, d_model), zeros\n\n        # TODO: define self.pos_embed as nn.Parameter of shape (1, num_patches+1, d_model), zeros\n\n\n\n    def forward(self, x):\n\n        B = x.shape[0]\n\n        # TODO: x = self.patch_embed(x)\n\n        # TODO: expand cls_token to batch: cls = self.cls_token.expand(B, -1, -1)\n\n        # TODO: x = torch.cat([cls, x], dim=1)\n\n        # TODO: x = x + self.pos_embed\n\n        # TODO: return x\n\n        pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    emb = ViTEmbedding(img_size=224, patch_size=16, d_model=768)\n\n    x = torch.randn(2, 3, 224, 224)\n\n    out = emb(x)\n\n    assert out is not None\n\n    assert out.shape == (2, 197, 768), f\"Expected (2,197,768), got {out.shape}\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["CLS token: nn.Parameter(torch.zeros(1, 1, d_model))", "Expand for batch: self.cls_token.expand(B, -1, -1)", "Concatenate along dim=1 (sequence dim)"]
      },
      {
        id: "vit-s3",
        number: 3,
        title: "ViT Encoder Block (Pre-LN)",
        description: "ViT uses pre-norm: LayerNorm BEFORE each sublayer, not after. This improves training stability compared to the original Transformer's post-norm.",
        concepts: ["Pre-layer normalization", "Training stability"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\nimport torch.nn.functional as F\n\n\n\nclass ViTEncoderBlock(nn.Module):\n\n    def __init__(self, d_model: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0):\n\n        super().__init__()\n\n        d_ff = int(d_model * mlp_ratio)\n\n        self.norm1 = nn.LayerNorm(d_model)\n\n        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)\n\n        self.norm2 = nn.LayerNorm(d_model)\n\n        self.mlp = nn.Sequential(\n\n            nn.Linear(d_model, d_ff),\n\n            nn.GELU(),\n\n            nn.Linear(d_ff, d_model),\n\n        )\n\n\n\n    def forward(self, x):\n\n        # Pre-LN attention: normalize first, then attention, then residual\n\n        # TODO: attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))\n\n        # TODO: x = x + attn_out\n\n        # TODO: x = x + self.mlp(self.norm2(x))\n\n        # TODO: return x\n\n        pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    block = ViTEncoderBlock(d_model=768, num_heads=12)\n\n    x = torch.randn(1, 197, 768)\n\n    out = block(x)\n\n    assert out is not None\n\n    assert out.shape == (1, 197, 768), f\"Expected (1,197,768), got {out.shape}\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["Use nn.MultiheadAttention with batch_first=True \u2014 it accepts (B,S,D) directly", "Self-attention: query=key=value=self.norm1(x)", "Add attn_out to x (residual), then apply MLP with norm2"]
      },
      {
        id: "vit-s4",
        number: 4,
        title: "Full ViT Classifier",
        description: "Stack N encoder blocks, then take only the CLS token (index 0) for classification. This is ViT's pooling strategy.",
        concepts: ["CLS pooling", "Depth scaling"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\nimport torch.nn.functional as F\n\n\n\n# (ViTEmbedding and ViTEncoderBlock assumed defined above)\n\n\n\nclass ViT(nn.Module):\n\n    def __init__(\n\n        self,\n\n        img_size: int = 224,\n\n        patch_size: int = 16,\n\n        d_model: int = 768,\n\n        num_heads: int = 12,\n\n        num_layers: int = 2,\n\n        num_classes: int = 10,\n\n    ):\n\n        super().__init__()\n\n        # For testing, use smaller d_model\n\n        # TODO: define self.embedding as ViTEmbedding\n\n        # TODO: define self.blocks as nn.ModuleList of num_layers ViTEncoderBlocks\n\n        # TODO: define self.norm as nn.LayerNorm(d_model)\n\n        # TODO: define self.head as nn.Linear(d_model, num_classes)\n\n\n\n    def forward(self, x):\n\n        # TODO: x = self.embedding(x)         \u2192 (B, 197, d_model)\n\n        # TODO: for block in self.blocks: x = block(x)\n\n        # TODO: x = self.norm(x)\n\n        # TODO: cls = x[:, 0]                 \u2192 (B, d_model) \u2014 CLS token only\n\n        # TODO: return self.head(cls)\n\n        pass\n\nFor testing use d_model=64, num_heads=4:",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n\n\n    class PatchEmbedding(torch.nn.Module):\n\n        def __init__(self, img_size=224, patch_size=16, in_channels=3, d_model=64):\n\n            super().__init__()\n\n            self.num_patches = (img_size // patch_size) ** 2\n\n            self.proj = torch.nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)\n\n        def forward(self, x):\n\n            return self.proj(x).flatten(2).transpose(1, 2)\n\n\n\n    class ViTEmbedding(torch.nn.Module):\n\n        def __init__(self, img_size=224, patch_size=16, d_model=64):\n\n            super().__init__()\n\n            self.patch_embed = PatchEmbedding(img_size, patch_size, 3, d_model)\n\n            n = (img_size // patch_size) ** 2\n\n            self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, d_model))\n\n            self.pos_embed = torch.nn.Parameter(torch.zeros(1, n+1, d_model))\n\n        def forward(self, x):\n\n            B = x.shape[0]\n\n            x = self.patch_embed(x)\n\n            cls = self.cls_token.expand(B, -1, -1)\n\n            x = torch.cat([cls, x], dim=1)\n\n            return x + self.pos_embed\n\n\n\n    class ViTEncoderBlock(torch.nn.Module):\n\n        def __init__(self, d_model=64, num_heads=4, mlp_ratio=4.0):\n\n            super().__init__()\n\n            d_ff = int(d_model * mlp_ratio)\n\n            self.norm1 = torch.nn.LayerNorm(d_model)\n\n            self.attn = torch.nn.MultiheadAttention(d_model, num_heads, batch_first=True)\n\n            self.norm2 = torch.nn.LayerNorm(d_model)\n\n            self.mlp = torch.nn.Sequential(torch.nn.Linear(d_model, d_ff), torch.nn.GELU(), torch.nn.Linear(d_ff, d_model))\n\n        def forward(self, x):\n\n            a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))\n\n            x = x + a\n\n            return x + self.mlp(self.norm2(x))\n\n\n\n    model = ViT(img_size=224, patch_size=16, d_model=64, num_heads=4, num_layers=2, num_classes=10)\n\n    x = torch.randn(1, 3, 224, 224)\n\n    out = model(x)\n\n    assert out is not None\n\n    assert out.shape == (1, 10), f\"Expected (1,10), got {out.shape}\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: []
      },
    ]
  },
  {
    paperId: "lora",
    paperTitle: "LoRA",
    totalSteps: 3,
    steps: [
      {
        id: "lora-s1",
        number: 1,
        title: "Low-Rank Weight Decomposition",
        description: "LoRA parameterizes \u0394W = B\u00b7A where A\u2208\u211d^(r\u00d7d_in), B\u2208\u211d^(d_out\u00d7r). Key: initialize B=0 so \u0394W=0 at the start \u2014 the base model is unchanged.",
        concepts: ["Low-rank factorization", "Zero initialization", "Parameter efficiency"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\n\n\nclass LoRADelta(nn.Module):\n\n    \"\"\"Represents a low-rank weight update \u0394W = B @ A\"\"\"\n\n    def __init__(self, d_in: int, d_out: int, rank: int = 4):\n\n        super().__init__()\n\n        # TODO: self.A = nn.Parameter(torch.randn(rank, d_in) * 0.01)\n\n        # TODO: self.B = nn.Parameter(torch.zeros(d_out, rank))  \u2190 zeros!\n\n\n\n    def get_delta_weight(self):\n\n        # TODO: return self.B @ self.A   (shape: d_out \u00d7 d_in)\n\n        pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    delta = LoRADelta(d_in=64, d_out=64, rank=4)\n\n    dw = delta.get_delta_weight()\n\n    assert dw is not None, \"get_delta_weight returned None\"\n\n    assert dw.shape == (64, 64), f\"Expected (64,64), got {dw.shape}\"\n\n    assert torch.allclose(dw, torch.zeros(64, 64)), \"\u0394W must be zero at init (B is zeros)\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["A is (rank, d_in), B is (d_out, rank) \u2014 B @ A gives (d_out, d_in)", "B=zeros means \u0394W=zeros at init, so the pretrained model is unchanged", "torch.zeros(...) for B, small random (e.g. * 0.01) for A"]
      },
      {
        id: "lora-s2",
        number: 2,
        title: "LoRA Linear Layer",
        description: "Wrap nn.Linear with LoRA. Output = x @ W.T + (x @ A.T @ B.T) * (alpha/r). The base weight W is frozen; only A and B are trained.",
        concepts: ["Adapter pattern", "Alpha scaling", "Frozen base weights"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\n\n\nclass LoRALinear(nn.Module):\n\n    def __init__(self, d_in: int, d_out: int, rank: int = 4, alpha: float = 1.0, bias: bool = True):\n\n        super().__init__()\n\n        self.base = nn.Linear(d_in, d_out, bias=bias)\n\n        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.01)\n\n        self.B = nn.Parameter(torch.zeros(d_out, rank))\n\n        self.scale = alpha / rank\n\n        # Freeze base weight\n\n        self.base.weight.requires_grad = False\n\n        if bias and self.base.bias is not None:\n\n            self.base.bias.requires_grad = False\n\n\n\n    def forward(self, x):\n\n        # TODO: base_out = self.base(x)\n\n        # TODO: lora_out = (x @ self.A.T @ self.B.T) * self.scale\n\n        # TODO: return base_out + lora_out\n\n        pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    torch.manual_seed(0)\n\n    layer = LoRALinear(d_in=32, d_out=64, rank=4, alpha=1.0)\n\n    x = torch.randn(2, 32)\n\n    out = layer(x)\n\n    assert out is not None\n\n    assert out.shape == (2, 64), f\"Expected (2,64), got {out.shape}\"\n\n    # With B=zeros, LoRA output should equal base linear output\n\n    base_out = layer.base(x)\n\n    assert torch.allclose(out, base_out, atol=1e-6), \"At init, LoRA output should match base (B=0)\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["Matrix multiply order: x @ self.A.T gives (batch, rank), then @ self.B.T gives (batch, d_out)", "Multiply by self.scale (= alpha/r) before adding", "At init B=0, so lora_out=0 and output equals base exactly"]
      },
      {
        id: "lora-s3",
        number: 3,
        title: "Freeze Base + Train LoRA Only",
        description: "Apply LoRA to a model: freeze all existing parameters, then replace target nn.Linear layers with LoRALinear. Only A and B will update during fine-tuning.",
        concepts: ["Selective parameter training", "Model surgery", "Fine-tuning efficiency"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\n\n\nclass LoRALinear(nn.Module):\n\n    def __init__(self, d_in, d_out, rank=4, alpha=1.0, bias=True):\n\n        super().__init__()\n\n        self.base = nn.Linear(d_in, d_out, bias=bias)\n\n        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.01)\n\n        self.B = nn.Parameter(torch.zeros(d_out, rank))\n\n        self.scale = alpha / rank\n\n        self.base.weight.requires_grad = False\n\n        if bias and self.base.bias is not None:\n\n            self.base.bias.requires_grad = False\n\n    def forward(self, x):\n\n        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale\n\n\n\ndef apply_lora(model: nn.Module, rank: int = 4, alpha: float = 1.0) -> nn.Module:\n\n    \"\"\"\n\n    Replace all nn.Linear layers in model with LoRALinear.\n\n    Freeze all non-LoRA parameters.\n\n    \"\"\"\n\n    # TODO: first freeze all parameters: for p in model.parameters(): p.requires_grad = False\n\n    # TODO: iterate named_modules, replace nn.Linear with LoRALinear\n\n    # TODO: return model\n\n    pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch, torch.nn as nn\n\n\n\n    class LoRALinear(nn.Module):\n\n        def __init__(self, d_in, d_out, rank=4, alpha=1.0, bias=True):\n\n            super().__init__()\n\n            self.base = nn.Linear(d_in, d_out, bias=bias)\n\n            self.A = nn.Parameter(torch.randn(rank, d_in) * 0.01)\n\n            self.B = nn.Parameter(torch.zeros(d_out, rank))\n\n            self.scale = alpha / rank\n\n            self.base.weight.requires_grad = False\n\n            if bias and self.base.bias is not None: self.base.bias.requires_grad = False\n\n        def forward(self, x):\n\n            return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale\n\n\n\n    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))\n\n    model = apply_lora(model, rank=4, alpha=1.0)\n\n    trainable = [n for n, p in model.named_parameters() if p.requires_grad]\n\n    assert len(trainable) > 0, \"No trainable parameters after apply_lora\"\n\n    assert all(\"lora\" in n.lower() or \".A\" in n or \".B\" in n for n in trainable), \\\n\n        f\"Non-LoRA params trainable: {trainable}\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["Freeze first: for p in model.parameters(): p.requires_grad = False", "Replace with setattr(parent, child_name, LoRALinear(m.in_features, m.out_features, rank, alpha))", "Iterate with list(model.named_modules()) and getattr to find the parent"]
      },
    ]
  },
  {
    paperId: "bert",
    paperTitle: "BERT",
    totalSteps: 4,
    steps: [
      {
        id: "bert-s1",
        number: 1,
        title: "BERT Embeddings",
        description: "BERT embeddings = token_embedding + position_embedding + token_type_embedding, all summed and normalized.",
        concepts: ["Token type IDs", "Positional embeddings", "Embedding sum"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\n\n\nclass BertEmbeddings(nn.Module):\n\n    def __init__(self, vocab_size: int = 30522, d_model: int = 768, max_len: int = 512, num_token_types: int = 2):\n\n        super().__init__()\n\n        # TODO: self.token_embeddings = nn.Embedding(vocab_size, d_model)\n\n        # TODO: self.position_embeddings = nn.Embedding(max_len, d_model)\n\n        # TODO: self.token_type_embeddings = nn.Embedding(num_token_types, d_model)\n\n        # TODO: self.norm = nn.LayerNorm(d_model)\n\n\n\n    def forward(self, input_ids, token_type_ids=None):\n\n        # input_ids: (B, seq)\n\n        B, S = input_ids.shape\n\n        # TODO: position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)\n\n        # TODO: if token_type_ids is None: token_type_ids = torch.zeros_like(input_ids)\n\n        # TODO: embeddings = token_emb + position_emb + token_type_emb\n\n        # TODO: return self.norm(embeddings)\n\n        pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    emb = BertEmbeddings(vocab_size=100, d_model=64, max_len=512)\n\n    ids = torch.randint(0, 100, (1, 8))\n\n    out = emb(ids)\n\n    assert out is not None\n\n    assert out.shape == (1, 8, 64), f\"Expected (1,8,64), got {out.shape}\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)", "Sum all three embeddings before LayerNorm", "token_type_ids defaults to zeros (single-sentence input)"]
      },
      {
        id: "bert-s2",
        number: 2,
        title: "BERT Self-Attention with Mask",
        description: "Extend dot-product attention with an additive mask: add a large negative value (-10000) to masked positions before softmax, making their attention weight \u22480.",
        concepts: ["Additive attention mask", "Padding mask", "Masked positions"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\nimport torch.nn.functional as F\n\n\n\nclass BertSelfAttention(nn.Module):\n\n    def __init__(self, d_model: int = 768, num_heads: int = 12):\n\n        super().__init__()\n\n        self.num_heads = num_heads\n\n        self.d_k = d_model // num_heads\n\n        self.query = nn.Linear(d_model, d_model)\n\n        self.key = nn.Linear(d_model, d_model)\n\n        self.value = nn.Linear(d_model, d_model)\n\n\n\n    def forward(self, hidden_states, attention_mask=None):\n\n        B, S, d = hidden_states.shape\n\n        # TODO: project Q, K, V\n\n        # TODO: reshape to (B, num_heads, S, d_k)\n\n        # TODO: scores = Q @ K.transpose(-2,-1) / sqrt(d_k)\n\n        # TODO: if attention_mask is not None: scores = scores + attention_mask\n\n        # TODO: weights = softmax(scores, dim=-1)\n\n        # TODO: return (weights @ V reshaped back to B,S,d)\n\n        pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    attn = BertSelfAttention(d_model=64, num_heads=4)\n\n    x = torch.randn(1, 8, 64)\n\n    out = attn(x)\n\n    assert out is not None\n\n    assert out.shape == (1, 8, 64), f\"Expected (1,8,64), got {out.shape}\"\n\n    # Test mask: last 2 positions masked\n\n    mask = torch.zeros(1, 1, 1, 8)\n\n    mask[:, :, :, -2:] = -10000.0\n\n    out_masked = attn(x, attention_mask=mask)\n\n    assert out_masked.shape == (1, 8, 64)\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["The mask shape is (B, 1, 1, S) to broadcast across heads and query positions", "Add mask to scores: scores = scores + attention_mask (not multiply)", "Reshape back: .transpose(1,2).contiguous().view(B, S, d)"]
      },
      {
        id: "bert-s3",
        number: 3,
        title: "BERT Encoder Block",
        description: "BERT's encoder block: SelfAttention \u2192 Output(linear+LayerNorm) \u2192 Intermediate(linear+GELU) \u2192 Output2(linear+LayerNorm).",
        concepts: ["BERT-specific FFN", "GELU activation", "Post-norm"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\nimport torch.nn.functional as F\n\n\n\n# BertSelfAttention assumed defined above\n\nclass BertSelfAttention(nn.Module):\n\n    def __init__(self, d_model=64, num_heads=4):\n\n        super().__init__()\n\n        self.num_heads = num_heads\n\n        self.d_k = d_model // num_heads\n\n        self.query = nn.Linear(d_model, d_model)\n\n        self.key = nn.Linear(d_model, d_model)\n\n        self.value = nn.Linear(d_model, d_model)\n\n    def forward(self, x, mask=None):\n\n        B, S, d = x.shape\n\n        def proj(m, t): return m(t).view(B, S, self.num_heads, self.d_k).transpose(1,2)\n\n        Q, K, V = proj(self.query,x), proj(self.key,x), proj(self.value,x)\n\n        s = Q @ K.transpose(-2,-1) / self.d_k**0.5\n\n        if mask is not None: s = s + mask\n\n        w = F.softmax(s, dim=-1)\n\n        return (w @ V).transpose(1,2).contiguous().view(B, S, d)\n\n\n\nclass BertEncoderBlock(nn.Module):\n\n    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ff: int = 256):\n\n        super().__init__()\n\n        self.attn = BertSelfAttention(d_model, num_heads)\n\n        self.attn_out = nn.Linear(d_model, d_model)\n\n        self.norm1 = nn.LayerNorm(d_model)\n\n        self.intermediate = nn.Linear(d_model, d_ff)\n\n        self.output = nn.Linear(d_ff, d_model)\n\n        self.norm2 = nn.LayerNorm(d_model)\n\n\n\n    def forward(self, x, attention_mask=None):\n\n        # TODO: attn_out = self.attn_out(self.attn(x, attention_mask))\n\n        # TODO: x = self.norm1(x + attn_out)\n\n        # TODO: ffn_out = self.output(F.gelu(self.intermediate(x)))\n\n        # TODO: x = self.norm2(x + ffn_out)\n\n        # TODO: return x\n\n        pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    block = BertEncoderBlock(d_model=64, num_heads=4, d_ff=256)\n\n    x = torch.randn(1, 8, 64)\n\n    out = block(x)\n\n    assert out is not None\n\n    assert out.shape == (1, 8, 64), f\"Expected (1,8,64), got {out.shape}\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["GELU: F.gelu(self.intermediate(x))", "Post-norm: add residual THEN normalize (opposite of ViT)", "Attention output goes through a linear layer before the residual add"]
      },
      {
        id: "bert-s4",
        number: 4,
        title: "BERT Classifier",
        description: "Stack N blocks, pool by taking the CLS token ([0] position), apply dropout, then a linear classifier.",
        concepts: ["CLS pooling", "Fine-tuning head", "Dropout regularization"],
        starterCode: "import torch\n\nimport torch.nn as nn\n\nimport torch.nn.functional as F\n\n\n\n# BertEncoderBlock assumed defined above (with BertSelfAttention inline)\n\n\n\nclass BertClassifier(nn.Module):\n\n    def __init__(self, d_model: int = 64, num_heads: int = 4, num_layers: int = 2, num_classes: int = 2, vocab_size: int = 100):\n\n        super().__init__()\n\n        self.embeddings = nn.Embedding(vocab_size, d_model)\n\n        # TODO: self.layers = nn.ModuleList of num_layers BertEncoderBlocks\n\n        # TODO: self.dropout = nn.Dropout(0.1)\n\n        # TODO: self.classifier = nn.Linear(d_model, num_classes)\n\n\n\n    def forward(self, input_ids):\n\n        # TODO: x = self.embeddings(input_ids)\n\n        # TODO: for layer in self.layers: x = layer(x)\n\n        # TODO: cls = x[:, 0]           \u2190 CLS token\n\n        # TODO: return self.classifier(self.dropout(cls))\n\n        pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n\n\n    # minimal BertEncoderBlock for test\n\n    import torch.nn as nn, torch.nn.functional as F\n\n    class BertEncoderBlock(nn.Module):\n\n        def __init__(self, d_model=64, num_heads=4, d_ff=256):\n\n            super().__init__()\n\n            self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)\n\n            self.norm1 = nn.LayerNorm(d_model)\n\n            self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))\n\n            self.norm2 = nn.LayerNorm(d_model)\n\n        def forward(self, x, attention_mask=None):\n\n            a, _ = self.attn(x, x, x)\n\n            x = self.norm1(x + a)\n\n            return self.norm2(x + self.ff(x))\n\n\n\n    model = BertClassifier(d_model=64, num_heads=4, num_layers=2, num_classes=2, vocab_size=100)\n\n    ids = torch.randint(0, 100, (1, 8))\n\n    out = model(ids)\n\n    assert out is not None\n\n    assert out.shape == (1, 2), f\"Expected (1,2), got {out.shape}\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: []
      },
    ]
  },
  {
    paperId: "flash-attention",
    paperTitle: "FlashAttention",
    totalSteps: 4,
    steps: [
      {
        id: "flash-s1",
        number: 1,
        title: "Standard Attention (Baseline)",
        description: "Implement standard O(N\u00b2) attention. This is the baseline we'll optimize. Materializes the full N\u00d7N attention matrix in memory.",
        concepts: ["Full attention matrix", "O(N\u00b2) memory", "Baseline for comparison"],
        starterCode: "import torch\n\nimport torch.nn.functional as F\n\n\n\ndef standard_attention(Q, K, V):\n\n    \"\"\"\n\n    Standard scaled dot-product attention.\n\n    Q, K, V: (batch, seq, d)\n\n    Returns: (batch, seq, d)\n\n    \"\"\"\n\n    # TODO: d_k = Q.size(-1)\n\n    # TODO: scores = Q @ K.transpose(-2, -1) / sqrt(d_k)\n\n    # TODO: weights = F.softmax(scores, dim=-1)\n\n    # TODO: return weights @ V\n\n    pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch, torch.nn.functional as F\n\n    torch.manual_seed(42)\n\n    Q = torch.randn(1, 8, 16)\n\n    K = torch.randn(1, 8, 16)\n\n    V = torch.randn(1, 8, 16)\n\n    out = standard_attention(Q, K, V)\n\n    assert out is not None\n\n    assert out.shape == (1, 8, 16), f\"Expected (1,8,16), got {out.shape}\"\n\n    ref = F.scaled_dot_product_attention(Q, K, V)\n\n    assert torch.allclose(out, ref, atol=1e-5), \"Output doesn't match reference\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["Scale: / Q.size(-1) ** 0.5", "Softmax over last dim: F.softmax(scores, dim=-1)", "Final: weights @ V"]
      },
      {
        id: "flash-s2",
        number: 2,
        title: "Numerically Stable Softmax",
        description: "Prevent overflow by subtracting the row maximum before computing exp. This is the key numerical trick in FlashAttention's online softmax.",
        concepts: ["Numerical stability", "Row-max subtraction", "Overflow prevention"],
        starterCode: "import torch\n\n\n\ndef stable_softmax(x):\n\n    \"\"\"\n\n    Numerically stable softmax along last dimension.\n\n    Subtract row max before exp to prevent overflow.\n\n    x: (..., N)\n\n    \"\"\"\n\n    # TODO: m = x.max(dim=-1, keepdim=True).values\n\n    # TODO: x_shifted = x - m\n\n    # TODO: exp_x = x_shifted.exp()\n\n    # TODO: return exp_x / exp_x.sum(dim=-1, keepdim=True)\n\n    pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch\n\n    # Normal case\n\n    x = torch.tensor([[1.0, 2.0, 3.0]])\n\n    out = stable_softmax(x)\n\n    ref = torch.softmax(x, dim=-1)\n\n    assert torch.allclose(out, ref, at\n<truncated 24505 bytes>\n\nNOTE: The output was truncated because it was too long. Use a more targeted query or a smaller range to get the information you needtestCode:\n# --- tests (do not modify) ---\ntry:\n    import torch\n    # Normal case\n    x = torch.tensor([[1.0, 2.0, 3.0]])\n    out = stable_softmax(x)\n    ref = torch.softmax(x, dim=-1)\n    assert torch.allclose(out, ref, atol=1e-6), f\"Normal case failed: {out} vs {ref}\"\n    # Large values that would overflow naive exp\n    x_large = torch.tensor([[1000.0, 1001.0, 1002.0]])\n    out_large = stable_softmax(x_large)\n    assert not torch.any(torch.isnan(out_large)), \"NaN with large values \u2014 not stable\"\n    assert torch.allclose(out_large.sum(), torch.tensor(1.0), atol=1e-5)\n    print(\"ALL_TESTS_PASSED\")\nexcept AssertionError as e:\n    print(f\"ASSERTION_FAILED: {e}\")\nexcept Exception as e:\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["Subtract max BEFORE exp, not after", "keepdim=True to broadcast correctly", "Sum along the same dim as max: dim=-1"]
      },
      {
        id: "flash-s3",
        number: 3,
        title: "Chunked Attention",
        description: "Process queries in blocks instead of all at once. Compute attention for one chunk of Q at a time \u2014 same output as standard attention, lower peak memory.",
        concepts: ["Block computation", "Memory efficiency", "Chunked processing"],
        starterCode: "import torch\n\nimport torch.nn.functional as F\n\n\n\ndef chunked_attention(Q, K, V, block_size: int = 2):\n\n    \"\"\"\n\n    Process Q in blocks of block_size rows.\n\n    Q, K, V: (batch, seq, d)\n\n    \"\"\"\n\n    B, S, d = Q.shape\n\n    output = torch.zeros_like(Q)\n\n    for start in range(0, S, block_size):\n\n        end = min(start + block_size, S)\n\n        Q_chunk = Q[:, start:end, :]     # (B, block, d)\n\n        # TODO: scores = Q_chunk @ K.transpose(-2,-1) / sqrt(d)\n\n        # TODO: weights = F.softmax(scores, dim=-1)\n\n        # TODO: output[:, start:end, :] = weights @ V\n\n    return output",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch, torch.nn.functional as F\n\n    torch.manual_seed(0)\n\n    Q = torch.randn(1, 8, 16)\n\n    K = torch.randn(1, 8, 16)\n\n    V = torch.randn(1, 8, 16)\n\n    ref = F.scaled_dot_product_attention(Q, K, V)\n\n    for bs in [1, 2, 4, 8]:\n\n        out = chunked_attention(Q, K, V, block_size=bs)\n\n        assert torch.allclose(out, ref, atol=1e-4), f\"block_size={bs}: mismatch\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: []
      },
      {
        id: "flash-s4",
        number: 4,
        title: "Online Softmax (FlashAttention Core)",
        description: "Maintain running statistics (max m, normalizer l, output o) so you never materialize the N\u00d7N matrix. Process K/V in blocks and update statistics on the fly.",
        concepts: ["Online algorithm", "Running max and sum", "IO-awareness"],
        starterCode: "import torch\n\nimport math\n\n\n\ndef flash_attention(Q, K, V, block_size: int = 2):\n\n    \"\"\"\n\n    Online softmax attention \u2014 never stores full N\u00d7N matrix.\n\n    Q, K, V: (batch, seq, d)\n\n    \"\"\"\n\n    B, S, d = Q.shape\n\n    scale = d ** -0.5\n\n    O = torch.zeros_like(Q)\n\n    l = torch.zeros(B, S, 1)   # running normalizer\n\n    m = torch.full((B, S, 1), float('-inf'))  # running max\n\n\n\n    for j in range(0, S, block_size):\n\n        K_j = K[:, j:j+block_size, :]\n\n        V_j = V[:, j:j+block_size, :]\n\n        # TODO: s_j = Q @ K_j.transpose(-2,-1) * scale   # (B, S, block)\n\n        # TODO: m_new = torch.maximum(m, s_j.max(dim=-1, keepdim=True).values)\n\n        # TODO: exp_s = torch.exp(s_j - m_new)\n\n        # TODO: l_new = torch.exp(m - m_new) * l + exp_s.sum(dim=-1, keepdim=True)\n\n        # TODO: O = (torch.exp(m - m_new) * O + exp_s @ V_j)\n\n        # TODO: m, l = m_new, l_new\n\n\n\n    # TODO: return O / l\n\n    pass",
        testCode: "# --- tests (do not modify) ---\n\ntry:\n\n    import torch, torch.nn.functional as F\n\n    torch.manual_seed(1)\n\n    Q = torch.randn(1, 8, 16)\n\n    K = torch.randn(1, 8, 16)\n\n    V = torch.randn(1, 8, 16)\n\n    ref = F.scaled_dot_product_attention(Q, K, V)\n\n    for bs in [1, 2, 4]:\n\n        out = flash_attention(Q, K, V, block_size=bs)\n\n        assert out is not None\n\n        assert torch.allclose(out, ref, atol=1e-4), f\"block_size={bs}: max diff={(out-ref).abs().max()}\"\n\n    print(\"ALL_TESTS_PASSED\")\n\nexcept AssertionError as e:\n\n    print(f\"ASSERTION_FAILED: {e}\")\n\nexcept Exception as e:\n\n    print(f\"ERROR: {type(e).__name__}: {e}\")",
        hints: ["The running max m and normalizer l let you correct previous output when a larger value is seen", "torch.exp(m - m_new) is the correction factor \u2014 it rescales what you already accumulated", "Final: O / l gives the normalized output"]
      },
    ]
  },
];

export const getImpl = (paperId: string): PaperImpl | undefined =>
  IMPLEMENTATIONS.find(i => i.paperId === paperId);
