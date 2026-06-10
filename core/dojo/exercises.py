"""
core/dojo/exercises.py — Code Dojo exercise catalog (Phase 12, M1)

Each exercise is a deterministic, self-contained spec. The learner implements
`fn_name` in NumPy; the browser validates their output against expected outputs
precomputed from `reference_solution`.

Exercise schema:
    id                : stable unique slug
    category          : Activation | Loss | Optimizer | Layer | Metric
    title             : display name
    difficulty        : 1..5
    fn_name           : the function the learner must define
    concept           : one-line summary
    math              : formula (unicode plain text, rendered in a <pre>)
    intuition         : why it works / why it matters (markdown-ish)
    starter_code      : signature + TODO body
    reference_solution: canonical NumPy implementation (server-side only)
    test_inputs       : list of {arg: value} (lists -> np.array, scalars passthrough)
    tolerance         : np.allclose atol
    hints             : progressive hints (revealed one at a time)
    common_mistakes   : list of frequent bugs
    related_hyperparameter : key into HYPERPARAMETER_EXPLANATIONS (or None)
    related_arch      : architecture that prominently uses this (or None)
"""

from typing import Any, Dict, List, Optional
from core.dojo.validator import run_reference


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

EXERCISES: List[Dict[str, Any]] = [

    # ===================== ACTIVATIONS =====================
    {
        "id": "relu",
        "category": "Activation",
        "title": "ReLU",
        "difficulty": 1,
        "fn_name": "relu",
        "concept": "Rectified Linear Unit — zero out negatives.",
        "math": "relu(x) = max(0, x)",
        "intuition": (
            "ReLU is the default activation for CNNs. It is cheap (a single max), "
            "non-saturating for positive inputs (gradients flow), and induces sparsity "
            "(dead-zone for negatives). Its gradient is 1 for x>0 and 0 for x<0."
        ),
        "starter_code": "def relu(x):\n    # x is a NumPy array. Return max(0, x) element-wise.\n    pass\n",
        "reference_solution": "def relu(x):\n    return np.maximum(0, x)\n",
        "test_inputs": [
            {"x": [-2.0, -0.5, 0.0, 1.5, 3.0]},
            {"x": [[-1.0, 2.0], [3.0, -4.0]]},
        ],
        "tolerance": 1e-6,
        "hints": [
            "There is a single NumPy function that takes the element-wise maximum of two arrays.",
            "np.maximum(0, x) broadcasts the scalar 0 against every element of x.",
        ],
        "common_mistakes": [
            "Using max(0, x) (Python builtin) instead of np.maximum — fails on arrays.",
            "Using np.max(x) which collapses to a single value instead of element-wise.",
        ],
        "related_hyperparameter": None,
        "related_arch": "ResNet",
    },
    {
        "id": "leaky_relu",
        "category": "Activation",
        "title": "Leaky ReLU",
        "difficulty": 1,
        "fn_name": "leaky_relu",
        "concept": "ReLU with a small negative slope to avoid dead neurons.",
        "math": "leaky_relu(x) = x if x > 0 else alpha * x",
        "intuition": (
            "Plain ReLU can 'die': once a neuron outputs 0 it may never recover because "
            "its gradient is 0. Leaky ReLU leaks a small slope (alpha, e.g. 0.01) for "
            "negative inputs so gradients never fully vanish."
        ),
        "starter_code": "def leaky_relu(x, alpha):\n    # Return x where x>0, else alpha*x (element-wise).\n    pass\n",
        "reference_solution": "def leaky_relu(x, alpha):\n    return np.where(x > 0, x, alpha * x)\n",
        "test_inputs": [
            {"x": [-2.0, -0.5, 0.0, 1.5, 3.0], "alpha": 0.01},
            {"x": [[-1.0, 2.0], [3.0, -4.0]], "alpha": 0.1},
        ],
        "tolerance": 1e-6,
        "hints": [
            "np.where(condition, a, b) selects a where condition is true, else b.",
            "The condition is x > 0; the two branches are x and alpha*x.",
        ],
        "common_mistakes": [
            "Forgetting that alpha multiplies x (not a constant).",
            "Treating x==0 as negative — convention here is x>0 keeps x.",
        ],
        "related_hyperparameter": None,
        "related_arch": None,
    },
    {
        "id": "sigmoid",
        "category": "Activation",
        "title": "Sigmoid",
        "difficulty": 2,
        "fn_name": "sigmoid",
        "concept": "Squash any real number into (0, 1).",
        "math": "sigmoid(x) = 1 / (1 + exp(-x))",
        "intuition": (
            "Sigmoid maps logits to probabilities for binary classification. It saturates "
            "for large |x| (gradients vanish), which is why deep nets prefer ReLU internally "
            "but still use sigmoid at the output for binary problems."
        ),
        "starter_code": "def sigmoid(x):\n    # Return the logistic sigmoid element-wise.\n    pass\n",
        "reference_solution": "def sigmoid(x):\n    return 1.0 / (1.0 + np.exp(-x))\n",
        "test_inputs": [
            {"x": [-2.0, -1.0, 0.0, 1.0, 2.0]},
            {"x": [[0.0, 4.0], [-4.0, 0.5]]},
        ],
        "tolerance": 1e-6,
        "hints": [
            "Use np.exp for the exponential.",
            "sigmoid(0) must equal exactly 0.5 — check your sign on the exponent.",
        ],
        "common_mistakes": [
            "Sign error: exp(x) instead of exp(-x).",
            "Integer division in other languages — in NumPy use float literals.",
        ],
        "related_hyperparameter": None,
        "related_arch": None,
    },
    {
        "id": "tanh",
        "category": "Activation",
        "title": "Tanh",
        "difficulty": 1,
        "fn_name": "tanh",
        "concept": "Zero-centered squashing into (-1, 1).",
        "math": "tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))",
        "intuition": (
            "Tanh is a zero-centered sigmoid — outputs in (-1, 1). Zero-centering helps "
            "optimization vs sigmoid, which is why it was the RNN default before gating."
        ),
        "starter_code": "def tanh(x):\n    # Implement tanh. You may use np.exp (not np.tanh).\n    pass\n",
        "reference_solution": "def tanh(x):\n    return np.tanh(x)\n",
        "test_inputs": [
            {"x": [-2.0, -1.0, 0.0, 1.0, 2.0]},
            {"x": [[0.0, 3.0], [-3.0, 0.25]]},
        ],
        "tolerance": 1e-6,
        "hints": [
            "np.tanh exists, but try the exp form for understanding.",
            "tanh(x) = 2*sigmoid(2x) - 1 is an equivalent stable formulation.",
        ],
        "common_mistakes": [
            "Overflow with the raw exp form for large |x| — np.tanh is stable.",
        ],
        "related_hyperparameter": None,
        "related_arch": None,
    },
    {
        "id": "softmax",
        "category": "Activation",
        "title": "Softmax (numerically stable)",
        "difficulty": 3,
        "fn_name": "softmax",
        "concept": "Turn a vector of logits into a probability distribution over the last axis.",
        "math": "softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))",
        "intuition": (
            "Softmax normalizes logits into probabilities that sum to 1. The KEY trick is "
            "subtracting max(x) before exp — mathematically identical, but prevents exp() "
            "from overflowing to inf for large logits. Operate over the LAST axis so it works "
            "for both a single vector and a batch."
        ),
        "starter_code": (
            "def softmax(x):\n"
            "    # Stable softmax over the last axis.\n"
            "    # 1) subtract the per-row max  2) exp  3) divide by the per-row sum\n"
            "    pass\n"
        ),
        "reference_solution": (
            "def softmax(x):\n"
            "    z = x - np.max(x, axis=-1, keepdims=True)\n"
            "    e = np.exp(z)\n"
            "    return e / np.sum(e, axis=-1, keepdims=True)\n"
        ),
        "test_inputs": [
            {"x": [1.0, 2.0, 3.0]},
            {"x": [[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]},
            {"x": [1000.0, 1001.0, 1002.0]},
        ],
        "tolerance": 1e-6,
        "hints": [
            "Subtract np.max(x, axis=-1, keepdims=True) before exponentiating.",
            "keepdims=True keeps the axis so broadcasting in the division works.",
            "Divide by np.sum(e, axis=-1, keepdims=True), not the global sum.",
        ],
        "common_mistakes": [
            "Forgetting to subtract the max → overflow on the [1000,1001,1002] case.",
            "Using keepdims=False → broadcasting error or wrong normalization.",
            "Normalizing over the wrong axis (axis=0 instead of axis=-1).",
        ],
        "related_hyperparameter": "Attention Heads",
        "related_arch": "Transformer",
    },
    {
        "id": "gelu",
        "category": "Activation",
        "title": "GELU (tanh approximation)",
        "difficulty": 4,
        "fn_name": "gelu",
        "concept": "Gaussian Error Linear Unit — the Transformer-era activation.",
        "math": "gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))",
        "intuition": (
            "GELU weights inputs by their value under a Gaussian CDF — a smooth blend "
            "between identity and zero. It is the default in BERT/GPT/ViT. Use the tanh "
            "approximation here (the exact form uses erf)."
        ),
        "starter_code": (
            "def gelu(x):\n"
            "    # tanh approximation of GELU.\n"
            "    # constant c = sqrt(2/pi) ~= 0.7978845608\n"
            "    pass\n"
        ),
        "reference_solution": (
            "def gelu(x):\n"
            "    c = np.sqrt(2.0 / np.pi)\n"
            "    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))\n"
        ),
        "test_inputs": [
            {"x": [-2.0, -0.5, 0.0, 0.5, 2.0]},
            {"x": [[1.0, -1.0], [0.0, 3.0]]},
        ],
        "tolerance": 1e-5,
        "hints": [
            "c = np.sqrt(2/np.pi). Watch operator precedence on x**3.",
            "The whole thing is 0.5 * x * (1 + tanh(...)).",
        ],
        "common_mistakes": [
            "Cubing wrong: use x**3, not 3*x.",
            "Dropping the leading 0.5 * x factor.",
        ],
        "related_hyperparameter": None,
        "related_arch": "ViT",
    },

    # ===================== LOSSES =====================
    {
        "id": "mse_loss",
        "category": "Loss",
        "title": "Mean Squared Error",
        "difficulty": 1,
        "fn_name": "mse_loss",
        "concept": "Average squared difference between predictions and targets.",
        "math": "MSE = mean( (y_pred - y_true)^2 )",
        "intuition": (
            "MSE is the default regression loss. Squaring penalizes large errors heavily "
            "(quadratically) and yields a smooth, convex objective for linear models."
        ),
        "starter_code": "def mse_loss(y_pred, y_true):\n    # Return the scalar mean squared error.\n    pass\n",
        "reference_solution": "def mse_loss(y_pred, y_true):\n    return np.mean((y_pred - y_true) ** 2)\n",
        "test_inputs": [
            {"y_pred": [2.0, 0.0, 1.0], "y_true": [1.0, 0.0, 2.0]},
            {"y_pred": [[1.0, 2.0], [3.0, 4.0]], "y_true": [[1.0, 1.0], [1.0, 1.0]]},
        ],
        "tolerance": 1e-6,
        "hints": [
            "Subtract, square with **2, then np.mean over everything.",
        ],
        "common_mistakes": [
            "Using np.sum instead of np.mean.",
            "Forgetting to square (that would be mean error, not MSE).",
        ],
        "related_hyperparameter": None,
        "related_arch": None,
    },
    {
        "id": "mae_loss",
        "category": "Loss",
        "title": "Mean Absolute Error",
        "difficulty": 1,
        "fn_name": "mae_loss",
        "concept": "Average absolute difference — robust to outliers.",
        "math": "MAE = mean( |y_pred - y_true| )",
        "intuition": (
            "MAE grows linearly with error, so it is far less sensitive to outliers than MSE. "
            "Its gradient is constant in magnitude, which can slow convergence near the minimum."
        ),
        "starter_code": "def mae_loss(y_pred, y_true):\n    # Return the scalar mean absolute error.\n    pass\n",
        "reference_solution": "def mae_loss(y_pred, y_true):\n    return np.mean(np.abs(y_pred - y_true))\n",
        "test_inputs": [
            {"y_pred": [2.0, 0.0, 1.0], "y_true": [1.0, 0.0, 2.0]},
            {"y_pred": [[1.0, -2.0]], "y_true": [[1.0, 2.0]]},
        ],
        "tolerance": 1e-6,
        "hints": ["np.abs gives element-wise absolute value; then np.mean."],
        "common_mistakes": ["Squaring instead of np.abs (that is MSE)."],
        "related_hyperparameter": None,
        "related_arch": None,
    },
    {
        "id": "bce_loss",
        "category": "Loss",
        "title": "Binary Cross-Entropy",
        "difficulty": 3,
        "fn_name": "bce_loss",
        "concept": "Log-loss for binary classification given probabilities.",
        "math": "BCE = -mean( y*log(p) + (1-y)*log(1-p) )",
        "intuition": (
            "BCE measures the divergence between predicted probabilities p and binary labels y. "
            "It explodes when you are confidently wrong (p→0 while y=1). Clip p away from 0 and 1 "
            "to avoid log(0) = -inf."
        ),
        "starter_code": (
            "def bce_loss(p, y):\n"
            "    # p = predicted probabilities in (0,1), y = 0/1 labels.\n"
            "    # Clip p to [eps, 1-eps] before taking logs.\n"
            "    pass\n"
        ),
        "reference_solution": (
            "def bce_loss(p, y):\n"
            "    eps = 1e-12\n"
            "    p = np.clip(p, eps, 1.0 - eps)\n"
            "    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))\n"
        ),
        "test_inputs": [
            {"p": [0.9, 0.1, 0.8], "y": [1.0, 0.0, 1.0]},
            {"p": [0.5, 0.5], "y": [1.0, 0.0]},
        ],
        "tolerance": 1e-6,
        "hints": [
            "Clip p with np.clip(p, eps, 1-eps) to avoid log(0).",
            "The formula has two symmetric terms: y*log(p) and (1-y)*log(1-p).",
            "Don't forget the leading minus sign and np.mean.",
        ],
        "common_mistakes": [
            "log(0) = -inf when p hits exactly 0 or 1 — clip first.",
            "Dropping the (1-y)*log(1-p) term.",
            "Forgetting the negative sign.",
        ],
        "related_hyperparameter": None,
        "related_arch": None,
    },
    {
        "id": "cross_entropy_logits",
        "category": "Loss",
        "title": "Cross-Entropy from Logits",
        "difficulty": 4,
        "fn_name": "cross_entropy_from_logits",
        "concept": "Multi-class loss computed directly from raw logits + integer labels.",
        "math": "CE = -mean( log_softmax(logits)[i, label_i] )",
        "intuition": (
            "Production code never softmaxes then logs separately (unstable). Instead compute "
            "log-softmax in one stable pass: z = logits - max; logsumexp = log(sum(exp(z))); "
            "log_probs = z - logsumexp. Then gather the log-prob of the true class per row."
        ),
        "starter_code": (
            "def cross_entropy_from_logits(logits, labels):\n"
            "    # logits: (N, C) raw scores. labels: (N,) integer class indices.\n"
            "    # Compute stable log-softmax, gather the true-class log-prob, return -mean.\n"
            "    pass\n"
        ),
        "reference_solution": (
            "def cross_entropy_from_logits(logits, labels):\n"
            "    z = logits - np.max(logits, axis=-1, keepdims=True)\n"
            "    log_probs = z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))\n"
            "    n = logits.shape[0]\n"
            "    idx = labels.astype(int)\n"
            "    return -np.mean(log_probs[np.arange(n), idx])\n"
        ),
        "test_inputs": [
            {"logits": [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]], "labels": [0, 2]},
            {"logits": [[1.0, 1.0, 1.0]], "labels": [1]},
        ],
        "tolerance": 1e-6,
        "hints": [
            "Subtract the per-row max from logits before exp (stability).",
            "log_softmax = z - log(sum(exp(z))) — no separate softmax then log.",
            "Gather with log_probs[np.arange(N), labels] to pick the true class.",
        ],
        "common_mistakes": [
            "Computing softmax then log separately → numerical instability.",
            "Wrong gather: using labels as a mask instead of an index.",
            "Labels as floats — cast to int for indexing.",
        ],
        "related_hyperparameter": "Label Smoothing",
        "related_arch": "ResNet",
    },

    # ===================== OPTIMIZERS (single update step) =====================
    {
        "id": "sgd_step",
        "category": "Optimizer",
        "title": "SGD Update Step",
        "difficulty": 1,
        "fn_name": "sgd_step",
        "concept": "The most basic parameter update.",
        "math": "param_new = param - lr * grad",
        "intuition": (
            "Vanilla stochastic gradient descent: step downhill along the negative gradient, "
            "scaled by the learning rate. Everything else (momentum, Adam) builds on this."
        ),
        "starter_code": "def sgd_step(param, grad, lr):\n    # Return the updated parameter array.\n    pass\n",
        "reference_solution": "def sgd_step(param, grad, lr):\n    return param - lr * grad\n",
        "test_inputs": [
            {"param": [1.0, -2.0, 0.5], "grad": [0.1, 0.2, -0.3], "lr": 0.1},
        ],
        "tolerance": 1e-7,
        "hints": ["It is a single line: subtract lr*grad from param."],
        "common_mistakes": [
            "Adding instead of subtracting the gradient (that's gradient ASCENT).",
        ],
        "related_hyperparameter": "Learning Rate",
        "related_arch": "ResNet",
    },
    {
        "id": "momentum_step",
        "category": "Optimizer",
        "title": "SGD + Momentum Step",
        "difficulty": 2,
        "fn_name": "momentum_step",
        "concept": "Accumulate a velocity to smooth and accelerate descent.",
        "math": "v_new = momentum * v + grad ;  param_new = param - lr * v_new",
        "intuition": (
            "Momentum accumulates an exponentially-decaying average of past gradients, like a "
            "ball rolling downhill. It damps oscillations across steep ravines and speeds up "
            "along consistent directions. Here, return the NEW parameter (velocity update is "
            "folded in)."
        ),
        "starter_code": (
            "def momentum_step(param, grad, v, lr, momentum):\n"
            "    # v is the previous velocity. Compute v_new, then the new param.\n"
            "    pass\n"
        ),
        "reference_solution": (
            "def momentum_step(param, grad, v, lr, momentum):\n"
            "    v_new = momentum * v + grad\n"
            "    return param - lr * v_new\n"
        ),
        "test_inputs": [
            {"param": [1.0, -2.0], "grad": [0.1, 0.2], "v": [0.0, 0.0], "lr": 0.1, "momentum": 0.9},
            {"param": [1.0, -2.0], "grad": [0.1, 0.2], "v": [0.5, -0.5], "lr": 0.1, "momentum": 0.9},
        ],
        "tolerance": 1e-7,
        "hints": [
            "First update velocity: v_new = momentum*v + grad.",
            "Then step: param - lr * v_new.",
        ],
        "common_mistakes": [
            "Using the old v instead of v_new in the param update.",
            "Multiplying grad by lr inside the velocity (convention here keeps lr outside).",
        ],
        "related_hyperparameter": "Learning Rate",
        "related_arch": None,
    },
    {
        "id": "adam_step",
        "category": "Optimizer",
        "title": "Adam Update Step",
        "difficulty": 4,
        "fn_name": "adam_step",
        "concept": "Adaptive moments with bias correction.",
        "math": (
            "m = b1*m + (1-b1)*g ; v = b2*v + (1-b2)*g^2\n"
            "m_hat = m/(1-b1^t) ; v_hat = v/(1-b2^t)\n"
            "param_new = param - lr * m_hat / (sqrt(v_hat) + eps)"
        ),
        "intuition": (
            "Adam keeps a running mean (m) and uncentered variance (v) of gradients, then "
            "normalizes the step so each parameter gets its own effective learning rate. The "
            "bias-correction (dividing by 1-beta^t) matters most in early steps when m,v start at 0."
        ),
        "starter_code": (
            "def adam_step(param, grad, m, v, t, lr, beta1, beta2, eps):\n"
            "    # m,v are running moments (same shape as param). t is the step (>=1).\n"
            "    # Update m and v, apply bias correction, return the new param.\n"
            "    pass\n"
        ),
        "reference_solution": (
            "def adam_step(param, grad, m, v, t, lr, beta1, beta2, eps):\n"
            "    m = beta1 * m + (1 - beta1) * grad\n"
            "    v = beta2 * v + (1 - beta2) * (grad ** 2)\n"
            "    m_hat = m / (1 - beta1 ** t)\n"
            "    v_hat = v / (1 - beta2 ** t)\n"
            "    return param - lr * m_hat / (np.sqrt(v_hat) + eps)\n"
        ),
        "test_inputs": [
            {"param": [0.5, -0.2], "grad": [0.1, 0.3], "m": [0.0, 0.0], "v": [0.0, 0.0],
             "t": 1, "lr": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
            {"param": [0.5, -0.2], "grad": [0.1, 0.3], "m": [0.05, 0.1], "v": [0.001, 0.02],
             "t": 5, "lr": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
        ],
        "tolerance": 1e-8,
        "hints": [
            "m mixes gradient; v mixes gradient SQUARED (grad**2).",
            "Bias correction divides by (1 - beta**t), with t the integer step.",
            "Final step: param - lr * m_hat / (sqrt(v_hat) + eps).",
        ],
        "common_mistakes": [
            "Skipping bias correction → wrong values especially at small t.",
            "Using v (variance) without sqrt in the denominator.",
            "beta1**t vs beta1*t — it is exponentiation.",
        ],
        "related_hyperparameter": "Learning Rate",
        "related_arch": "Transformer",
    },

    # ===================== LAYERS / TOOLS =====================
    {
        "id": "linear_forward",
        "category": "Layer",
        "title": "Linear (Dense) Forward",
        "difficulty": 1,
        "fn_name": "linear_forward",
        "concept": "The fundamental affine transform y = xW + b.",
        "math": "y = x @ W + b   with x:(N,in), W:(in,out), b:(out,)",
        "intuition": (
            "Every fully-connected layer is a matrix multiply plus a bias broadcast. Getting the "
            "shapes right (x:(N,in) @ W:(in,out) -> (N,out)) is the core skill."
        ),
        "starter_code": (
            "def linear_forward(x, W, b):\n"
            "    # x:(N,in), W:(in,out), b:(out,). Return (N,out).\n"
            "    pass\n"
        ),
        "reference_solution": "def linear_forward(x, W, b):\n    return x @ W + b\n",
        "test_inputs": [
            {"x": [[1.0, 2.0], [3.0, 4.0]], "W": [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], "b": [0.5, 0.5, 0.5]},
        ],
        "tolerance": 1e-6,
        "hints": [
            "Use the @ operator (or np.dot) for the matrix product.",
            "b broadcasts across the N rows automatically.",
        ],
        "common_mistakes": [
            "Transposing W unnecessarily — shapes are (N,in)@(in,out).",
            "Using * (element-wise) instead of @ (matmul).",
        ],
        "related_hyperparameter": "Hidden Dimension",
        "related_arch": None,
    },
    {
        "id": "layernorm",
        "category": "Layer",
        "title": "Layer Normalization",
        "difficulty": 3,
        "fn_name": "layernorm",
        "concept": "Normalize across features per example, then scale & shift.",
        "math": "y = gamma * (x - mean) / sqrt(var + eps) + beta   (mean/var over last axis)",
        "intuition": (
            "LayerNorm normalizes each example across its feature dimension (unlike BatchNorm "
            "which normalizes across the batch). It is the backbone of Transformer stability. "
            "gamma and beta are learnable per-feature scale and shift."
        ),
        "starter_code": (
            "def layernorm(x, gamma, beta, eps):\n"
            "    # Normalize x over the LAST axis, then apply gamma (scale) and beta (shift).\n"
            "    pass\n"
        ),
        "reference_solution": (
            "def layernorm(x, gamma, beta, eps):\n"
            "    mu = np.mean(x, axis=-1, keepdims=True)\n"
            "    var = np.var(x, axis=-1, keepdims=True)\n"
            "    return gamma * (x - mu) / np.sqrt(var + eps) + beta\n"
        ),
        "test_inputs": [
            {"x": [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], "gamma": [1.0, 1.0, 1.0], "beta": [0.0, 0.0, 0.0], "eps": 1e-5},
            {"x": [[1.0, 2.0, 3.0]], "gamma": [2.0, 2.0, 2.0], "beta": [1.0, 1.0, 1.0], "eps": 1e-5},
        ],
        "tolerance": 1e-5,
        "hints": [
            "Compute mean and variance over axis=-1 with keepdims=True.",
            "Add eps INSIDE the sqrt for numerical stability.",
            "Apply gamma and beta after normalizing.",
        ],
        "common_mistakes": [
            "Normalizing over the batch axis instead of the feature axis.",
            "eps outside the sqrt, or forgetting it entirely.",
            "Using std without eps → divide-by-zero on constant rows.",
        ],
        "related_hyperparameter": "Hidden Dimension",
        "related_arch": "Transformer",
    },
    {
        "id": "scaled_dot_product_attention",
        "category": "Layer",
        "title": "Scaled Dot-Product Attention",
        "difficulty": 5,
        "fn_name": "scaled_dot_product_attention",
        "concept": "The core operation of the Transformer.",
        "math": "Attention(Q,K,V) = softmax( Q @ K^T / sqrt(d_k) ) @ V",
        "intuition": (
            "Each query attends to all keys: similarity scores Q@K^T are scaled by 1/sqrt(d_k) "
            "(so softmax gradients don't vanish for large d_k), softmaxed into weights, then used "
            "to take a weighted sum of values V. This single operation powers all of modern NLP/vision."
        ),
        "starter_code": (
            "def scaled_dot_product_attention(q, k, v):\n"
            "    # q,k,v: (seq, d). Return (seq, d).\n"
            "    # scores = q@k^T / sqrt(d); weights = softmax(scores, last axis); out = weights@v\n"
            "    pass\n"
        ),
        "reference_solution": (
            "def scaled_dot_product_attention(q, k, v):\n"
            "    d = q.shape[-1]\n"
            "    scores = q @ k.T / np.sqrt(d)\n"
            "    z = scores - np.max(scores, axis=-1, keepdims=True)\n"
            "    w = np.exp(z)\n"
            "    w = w / np.sum(w, axis=-1, keepdims=True)\n"
            "    return w @ v\n"
        ),
        "test_inputs": [
            {"q": [[1.0, 0.0], [0.0, 1.0]], "k": [[1.0, 0.0], [0.0, 1.0]], "v": [[1.0, 2.0], [3.0, 4.0]]},
        ],
        "tolerance": 1e-6,
        "hints": [
            "d_k is the last dimension of q; divide scores by np.sqrt(d_k).",
            "Reuse your stable-softmax trick over the last axis of the scores.",
            "Final output is weights @ v.",
        ],
        "common_mistakes": [
            "Forgetting the 1/sqrt(d_k) scaling.",
            "Softmax over the wrong axis (must be over keys, the last axis).",
            "Using k instead of k.T in the score matmul.",
        ],
        "related_hyperparameter": "Attention Heads",
        "related_arch": "Transformer",
    },

    # ===================== METRICS =====================
    {
        "id": "accuracy",
        "category": "Metric",
        "title": "Classification Accuracy",
        "difficulty": 1,
        "fn_name": "accuracy",
        "concept": "Fraction of predictions that match the labels.",
        "math": "accuracy = mean( y_pred == y_true )",
        "intuition": (
            "The simplest classification metric: the fraction of correct predictions. Note it can "
            "be misleading on imbalanced data — which is why precision/recall/F1 exist."
        ),
        "starter_code": (
            "def accuracy(y_pred, y_true):\n"
            "    # y_pred and y_true are integer class arrays. Return the fraction equal.\n"
            "    pass\n"
        ),
        "reference_solution": "def accuracy(y_pred, y_true):\n    return np.mean(y_pred == y_true)\n",
        "test_inputs": [
            {"y_pred": [0, 1, 2, 1], "y_true": [0, 1, 1, 1]},
            {"y_pred": [1, 1, 1], "y_true": [0, 0, 0]},
        ],
        "tolerance": 1e-9,
        "hints": [
            "y_pred == y_true gives a boolean array; np.mean of booleans is the fraction True.",
        ],
        "common_mistakes": [
            "Summing without dividing by the count.",
        ],
        "related_hyperparameter": None,
        "related_arch": None,
    },

    # ===================== MORE LOSSES / OPTIMIZERS / LAYERS =====================
    {
        "id": "huber_loss",
        "category": "Loss",
        "title": "Huber Loss",
        "difficulty": 3,
        "fn_name": "huber_loss",
        "concept": "Quadratic for small errors, linear for large ones — robust regression.",
        "math": "L = mean( 0.5*e^2          if |e|<=delta\n          delta*(|e|-0.5*delta) otherwise )   where e = y_pred - y_true",
        "intuition": (
            "Huber blends MSE and MAE: it is smooth (quadratic) near zero for stable gradients, "
            "but switches to linear beyond delta so outliers don't dominate. Used in robust "
            "regression and as the 'smooth L1' loss in object detectors."
        ),
        "starter_code": (
            "def huber_loss(y_pred, y_true, delta):\n"
            "    # Quadratic within delta, linear beyond. Return the scalar mean.\n"
            "    pass\n"
        ),
        "reference_solution": (
            "def huber_loss(y_pred, y_true, delta):\n"
            "    e = np.abs(y_pred - y_true)\n"
            "    quad = np.minimum(e, delta)\n"
            "    lin = e - quad\n"
            "    return np.mean(0.5 * quad ** 2 + delta * lin)\n"
        ),
        "test_inputs": [
            {"y_pred": [0.0, 2.0, 5.0], "y_true": [0.0, 0.0, 0.0], "delta": 1.0},
            {"y_pred": [1.0, 1.0], "y_true": [1.2, 0.5], "delta": 1.0},
        ],
        "tolerance": 1e-6,
        "hints": [
            "Split each error into a clamped part quad=min(|e|,delta) and the leftover lin=|e|-quad.",
            "Per-element loss = 0.5*quad^2 + delta*lin; then take the mean.",
        ],
        "common_mistakes": [
            "Using a Python if on an array — vectorize with np.minimum instead.",
            "Forgetting the 0.5 factor on the quadratic region.",
        ],
        "related_hyperparameter": None,
        "related_arch": None,
    },
    {
        "id": "rmsprop_step",
        "category": "Optimizer",
        "title": "RMSProp Update Step",
        "difficulty": 3,
        "fn_name": "rmsprop_step",
        "concept": "Per-parameter learning rate from a running average of squared gradients.",
        "math": "sq = alpha*sq + (1-alpha)*g^2 ;  param_new = param - lr * g / (sqrt(sq) + eps)",
        "intuition": (
            "RMSProp divides each gradient by the root of its recent squared magnitude, so "
            "parameters with large, noisy gradients take smaller steps. It is Adam without the "
            "first-moment (momentum) term."
        ),
        "starter_code": (
            "def rmsprop_step(param, grad, sq_avg, lr, alpha, eps):\n"
            "    # Update the running mean-square sq_avg, then the parameter.\n"
            "    pass\n"
        ),
        "reference_solution": (
            "def rmsprop_step(param, grad, sq_avg, lr, alpha, eps):\n"
            "    sq_avg = alpha * sq_avg + (1 - alpha) * (grad ** 2)\n"
            "    return param - lr * grad / (np.sqrt(sq_avg) + eps)\n"
        ),
        "test_inputs": [
            {"param": [0.5, -0.2], "grad": [0.1, 0.3], "sq_avg": [0.0, 0.0],
             "lr": 0.01, "alpha": 0.9, "eps": 1e-8},
        ],
        "tolerance": 1e-7,
        "hints": [
            "sq_avg mixes the SQUARED gradient: alpha*sq_avg + (1-alpha)*grad**2.",
            "Step: param - lr * grad / (sqrt(sq_avg) + eps).",
        ],
        "common_mistakes": [
            "Using grad instead of grad**2 in the running average.",
            "Putting eps inside the sqrt instead of after it.",
        ],
        "related_hyperparameter": "Learning Rate",
        "related_arch": None,
    },
    {
        "id": "conv2d_valid",
        "category": "Layer",
        "title": "2-D Convolution (valid, single channel)",
        "difficulty": 4,
        "fn_name": "conv2d_valid",
        "concept": "Slide a kernel over a 2-D map with no padding.",
        "math": "out[i,j] = sum_{a,b} x[i+a, j+b] * kernel[a,b]   (output is (H-kh+1) x (W-kw+1))",
        "intuition": (
            "The core CNN operation: a kernel slides over the input, computing a dot product at "
            "each position. 'Valid' means no padding, so the output shrinks by kernel_size-1. "
            "Real frameworks vectorize this, but writing the loops cements how receptive fields work."
        ),
        "starter_code": (
            "def conv2d_valid(x, kernel):\n"
            "    # x:(H,W), kernel:(kh,kw). Return (H-kh+1, W-kw+1) via valid convolution.\n"
            "    pass\n"
        ),
        "reference_solution": (
            "def conv2d_valid(x, kernel):\n"
            "    H, W = x.shape\n"
            "    kh, kw = kernel.shape\n"
            "    out = np.zeros((H - kh + 1, W - kw + 1))\n"
            "    for i in range(H - kh + 1):\n"
            "        for j in range(W - kw + 1):\n"
            "            out[i, j] = np.sum(x[i:i+kh, j:j+kw] * kernel)\n"
            "    return out\n"
        ),
        "test_inputs": [
            {"x": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], "kernel": [[1.0, 0.0], [0.0, 1.0]]},
            {"x": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], "kernel": [[1.0, 1.0]]},
        ],
        "tolerance": 1e-6,
        "hints": [
            "Output shape is (H-kh+1, W-kw+1).",
            "At each (i,j), multiply the x[i:i+kh, j:j+kw] window by the kernel and sum.",
        ],
        "common_mistakes": [
            "Off-by-one in the output size or loop bounds.",
            "Element-wise window*kernel then np.sum — not a matmul.",
        ],
        "related_hyperparameter": None,
        "related_arch": "ResNet",
    },
    {
        "id": "maxpool2d",
        "category": "Layer",
        "title": "Max Pooling 2-D",
        "difficulty": 2,
        "fn_name": "maxpool2d",
        "concept": "Downsample by taking the max over non-overlapping windows.",
        "math": "out[i,j] = max( x[i*s:(i+1)*s, j*s:(j+1)*s] )",
        "intuition": (
            "Max pooling shrinks spatial size while keeping the strongest activation in each "
            "window — adding translation invariance and cutting compute. Stride equals the window "
            "size here (non-overlapping)."
        ),
        "starter_code": (
            "def maxpool2d(x, size):\n"
            "    # Non-overlapping max pool with window=size. Return the pooled map.\n"
            "    pass\n"
        ),
        "reference_solution": (
            "def maxpool2d(x, size):\n"
            "    H, W = x.shape\n"
            "    oh, ow = H // size, W // size\n"
            "    out = np.zeros((oh, ow))\n"
            "    for i in range(oh):\n"
            "        for j in range(ow):\n"
            "            out[i, j] = np.max(x[i*size:(i+1)*size, j*size:(j+1)*size])\n"
            "    return out\n"
        ),
        "test_inputs": [
            {"x": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]], "size": 2},
        ],
        "tolerance": 1e-6,
        "hints": [
            "Output size is H//size by W//size.",
            "For each output cell, take np.max over the size×size window.",
        ],
        "common_mistakes": [
            "Using mean instead of max (that's average pooling).",
            "Overlapping windows — stride should equal the window size here.",
        ],
        "related_hyperparameter": None,
        "related_arch": "ResNet",
    },

    # ===================== BACKPROP (gradients) =====================
    {
        "id": "relu_backward",
        "category": "Backprop",
        "title": "ReLU Backward",
        "difficulty": 2,
        "fn_name": "relu_backward",
        "concept": "Gradient of ReLU: pass gradient where input was positive.",
        "math": "dL/dx = grad_out * 1[x > 0]",
        "intuition": (
            "ReLU's local gradient is 1 for positive inputs and 0 otherwise. Backprop multiplies "
            "the incoming gradient by this mask — which is exactly why dead (always-negative) "
            "neurons stop learning: their gradient is zeroed."
        ),
        "starter_code": (
            "def relu_backward(grad_out, x):\n"
            "    # grad_out is dL/d(relu output). x is the ORIGINAL input. Return dL/dx.\n"
            "    pass\n"
        ),
        "reference_solution": "def relu_backward(grad_out, x):\n    return grad_out * (x > 0)\n",
        "test_inputs": [
            {"grad_out": [1.0, 1.0, 1.0, 1.0], "x": [-2.0, 0.5, -0.1, 3.0]},
            {"grad_out": [0.5, 2.0], "x": [1.0, -1.0]},
        ],
        "tolerance": 1e-6,
        "hints": [
            "The mask is (x > 0) — multiply grad_out by it.",
            "(x > 0) is a boolean array; multiplying by a float array promotes it to 0.0/1.0.",
        ],
        "common_mistakes": [
            "Using the ReLU OUTPUT instead of the input x for the mask (works for ReLU but not in general).",
            "Returning the mask alone, forgetting to multiply by grad_out.",
        ],
        "related_hyperparameter": None,
        "related_arch": "ResNet",
    },
    {
        "id": "sigmoid_backward",
        "category": "Backprop",
        "title": "Sigmoid Backward",
        "difficulty": 3,
        "fn_name": "sigmoid_backward",
        "concept": "Gradient of sigmoid expressed via its own output.",
        "math": "dL/dx = grad_out * out * (1 - out)   where out = sigmoid(x)",
        "intuition": (
            "Sigmoid has the elegant derivative s'(x) = s(x)(1-s(x)), so backprop only needs the "
            "forward OUTPUT. Note the factor peaks at 0.25 (out=0.5) and vanishes as out→0 or 1 — "
            "this is the saturation that causes vanishing gradients in deep sigmoid stacks."
        ),
        "starter_code": (
            "def sigmoid_backward(grad_out, out):\n"
            "    # out is the sigmoid output. Return dL/dx.\n"
            "    pass\n"
        ),
        "reference_solution": "def sigmoid_backward(grad_out, out):\n    return grad_out * out * (1 - out)\n",
        "test_inputs": [
            {"grad_out": [1.0, 1.0], "out": [0.5, 0.8]},
            {"grad_out": [2.0, 0.5, 1.0], "out": [0.1, 0.9, 0.5]},
        ],
        "tolerance": 1e-6,
        "hints": [
            "Use the identity s'(x) = out*(1-out); you don't need x at all.",
            "Multiply the local derivative by grad_out.",
        ],
        "common_mistakes": [
            "Recomputing sigmoid from x instead of using the provided output.",
            "Dropping the grad_out factor (that gives the local derivative, not dL/dx).",
        ],
        "related_hyperparameter": None,
        "related_arch": None,
    },
    {
        "id": "mse_backward",
        "category": "Backprop",
        "title": "MSE Backward",
        "difficulty": 2,
        "fn_name": "mse_backward",
        "concept": "Gradient of mean squared error w.r.t. predictions.",
        "math": "dL/dy_pred = 2 * (y_pred - y_true) / N",
        "intuition": (
            "Differentiating mean((y_pred-y_true)^2) gives 2*(y_pred-y_true)/N. The 1/N comes from "
            "the mean. This is the signal that flows back into the network from a regression head."
        ),
        "starter_code": (
            "def mse_backward(y_pred, y_true):\n"
            "    # Return dL/dy_pred for L = mean((y_pred - y_true)**2).\n"
            "    pass\n"
        ),
        "reference_solution": (
            "def mse_backward(y_pred, y_true):\n"
            "    n = y_pred.size\n"
            "    return 2.0 * (y_pred - y_true) / n\n"
        ),
        "test_inputs": [
            {"y_pred": [2.0, 0.0, 1.0], "y_true": [1.0, 0.0, 2.0]},
            {"y_pred": [1.0, 3.0], "y_true": [0.0, 0.0]},
        ],
        "tolerance": 1e-6,
        "hints": [
            "N is the number of elements (y_pred.size).",
            "Gradient is 2*(y_pred - y_true)/N — the 1/N comes from the mean.",
        ],
        "common_mistakes": [
            "Forgetting the factor of 2.",
            "Dividing by the wrong N (use the element count, not 1).",
        ],
        "related_hyperparameter": None,
        "related_arch": None,
    },
]


# ---------------------------------------------------------------------------
# Lookups & public projections
# ---------------------------------------------------------------------------

# Merge the "LeetCode for Data Science" problem bank into the catalog.
from core.dojo.problems import DS_PROBLEMS
EXERCISES = EXERCISES + DS_PROBLEMS

_BY_ID: Dict[str, Dict[str, Any]] = {ex["id"]: ex for ex in EXERCISES}

CATEGORY_ORDER = [
    "Activation", "Loss", "Optimizer", "Layer", "Backprop", "Metric",
    "NumPy", "Statistics", "Linear Algebra", "Probability", "ML Metrics",
    "ML Algorithms", "Data Manipulation", "Time Series", "Distances",
    "Feature Engineering", "NLP", "Algorithms",
]


def difficulty_label(d: int) -> str:
    """Map the 1-5 scale to LeetCode-style tiers."""
    if d <= 2:
        return "Easy"
    if d == 3:
        return "Medium"
    return "Hard"


def _example_count(ex: Dict[str, Any]) -> int:
    return min(2, len(ex.get("test_inputs", [])))


def get_all_topics() -> List[str]:
    """Sorted unique topic tags across the whole catalog (for the UI filter)."""
    topics = set()
    for ex in EXERCISES:
        for t in ex.get("topics", [ex["category"]]):
            topics.add(t)
    return sorted(topics)


def get_exercise_list() -> List[Dict[str, Any]]:
    """Lightweight list for the catalog sidebar (no solutions, no expected outputs)."""
    out = []
    for i, ex in enumerate(EXERCISES):
        out.append({
            "id": ex["id"],
            "number": i + 1,
            "category": ex["category"],
            "title": ex["title"],
            "difficulty": ex["difficulty"],
            "difficulty_label": difficulty_label(ex["difficulty"]),
            "topics": ex.get("topics", [ex["category"]]),
            "concept": ex["concept"],
        })
    return out


def get_public_exercise(exercise_id: str) -> Optional[Dict[str, Any]]:
    """
    Full exercise for the workspace — EXCLUDES reference_solution but INCLUDES
    expected_outputs precomputed from the reference solution.
    Returns None if the id is unknown.
    """
    ex = _BY_ID.get(exercise_id)
    if not ex:
        return None

    expected_outputs = run_reference(ex["reference_solution"], ex["fn_name"], ex["test_inputs"])

    return {
        "id": ex["id"],
        "category": ex["category"],
        "title": ex["title"],
        "difficulty": ex["difficulty"],
        "difficulty_label": difficulty_label(ex["difficulty"]),
        "topics": ex.get("topics", [ex["category"]]),
        "constraints": ex.get("constraints", ""),
        "fn_name": ex["fn_name"],
        "concept": ex["concept"],
        "math": ex["math"],
        "intuition": ex["intuition"],
        "starter_code": ex["starter_code"],
        "test_inputs": ex["test_inputs"],
        "expected_outputs": expected_outputs,
        "example_count": _example_count(ex),
        "tolerance": ex["tolerance"],
        "hints": ex["hints"],
        "common_mistakes": ex["common_mistakes"],
        "related_hyperparameter": ex.get("related_hyperparameter"),
        "related_arch": ex.get("related_arch"),
    }


def get_solution(exercise_id: str) -> Optional[Dict[str, Any]]:
    """Reference solution for the reveal action. None if unknown id."""
    ex = _BY_ID.get(exercise_id)
    if not ex:
        return None
    return {
        "id": ex["id"],
        "fn_name": ex["fn_name"],
        "reference_solution": ex["reference_solution"],
    }
