export interface Problem {
  slug: string;
  index: number;
  title: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  topics: string[];
  description: string;
  constraints: string[];
  examples: { input: string; output: string; explanation?: string }[];
  starter_code: string;
  test_input: string;
  acceptance: string;
  hints: string[];
}

export const PROBLEMS: Problem[] = [
  {
    slug: 'numpy-array-creation',
    index: 1,
    title: 'Create a NumPy Array',
    difficulty: 'Easy',
    topics: ['NumPy', 'Arrays'],
    description: `Return a 1D NumPy array containing the integers 1 through 5.

NumPy arrays are the fundamental data structure for numerical computing in Python. They are faster and more memory-efficient than Python lists for mathematical operations, and nearly every ML framework builds on top of them.`,
    constraints: [
      'Do not use a Python list as the final return value',
      'The returned object must be a numpy.ndarray',
      'Values must be exactly [1, 2, 3, 4, 5] in that order',
    ],
    examples: [
      { input: 'create_array()', output: 'array([1, 2, 3, 4, 5])' },
    ],
    starter_code: `import numpy as np\n\ndef create_array():\n    # TODO: return a 1D numpy array with values [1, 2, 3, 4, 5]\n    pass\n`,
    test_input: `import numpy as np\nprint(create_array())`,
    acceptance: '87.3%',
    hints: ['Use np.array([...]) to create an array from a Python list'],
  },
  {
    slug: 'ml-sigmoid',
    index: 2,
    title: 'Sigmoid Activation',
    difficulty: 'Easy',
    topics: ['Activation Functions', 'NumPy'],
    description: `Implement the sigmoid activation function.

The sigmoid function maps any real number to a value between 0 and 1, making it useful for binary classification outputs and as a squashing function:

  σ(x) = 1 / (1 + e^(-x))

For very negative x, sigmoid approaches 0. For very positive x, it approaches 1. At x = 0, sigmoid equals exactly 0.5.`,
    constraints: [
      'Input x can be a scalar or numpy array of any shape',
      'Must handle large positive and negative values without overflow',
      'Return type must match input type (array in → array out)',
    ],
    examples: [
      { input: 'sigmoid(0)', output: '0.5', explanation: 'At x=0: 1 / (1 + e^0) = 1 / 2 = 0.5' },
      { input: 'sigmoid(np.array([-1, 0, 1]))', output: 'array([0.2689, 0.5, 0.7311])' },
    ],
    starter_code: `import numpy as np\n\ndef sigmoid(x):\n    # TODO: implement sigmoid\n    # Formula: 1 / (1 + np.exp(-x))\n    pass\n`,
    test_input: `import numpy as np\nprint(round(float(sigmoid(0)), 4))\nprint(np.round(sigmoid(np.array([-1, 0, 1])), 4))`,
    acceptance: '79.4%',
    hints: [
      'Use np.exp() for the exponential function',
      'sigmoid(x) = 1 / (1 + np.exp(-x))',
    ],
  },
  {
    slug: 'ml-relu',
    index: 3,
    title: 'ReLU Activation',
    difficulty: 'Easy',
    topics: ['Activation Functions', 'NumPy'],
    description: `Implement the Rectified Linear Unit (ReLU) activation function.

ReLU is the most widely used activation function in deep learning. It introduces non-linearity while being computationally efficient:

  ReLU(x) = max(0, x)

For negative inputs, ReLU returns 0. For positive inputs, it returns the value unchanged. This avoids the vanishing gradient problem associated with sigmoid/tanh.`,
    constraints: [
      'Input x can be a scalar or numpy array of any shape',
      'Negative values must become exactly 0',
      'Positive values pass through unchanged',
    ],
    examples: [
      { input: 'relu(np.array([-2, -1, 0, 1, 2]))', output: 'array([0, 0, 0, 1, 2])' },
      { input: 'relu(-5.0)', output: '0.0' },
    ],
    starter_code: `import numpy as np\n\ndef relu(x):\n    # TODO: implement ReLU = max(0, x)\n    pass\n`,
    test_input: `import numpy as np\nprint(relu(np.array([-2, -1, 0, 1, 2])))`,
    acceptance: '84.7%',
    hints: [
      'np.maximum(0, x) is the cleanest approach',
      'You can also use np.clip(x, 0, None)',
    ],
  },
  {
    slug: 'ml-mse',
    index: 4,
    title: 'Mean Squared Error',
    difficulty: 'Medium',
    topics: ['Loss Functions', 'NumPy'],
    description: `Implement the Mean Squared Error (MSE) loss function.

MSE is the standard loss for regression tasks. It measures the average squared difference between predictions and ground truth:

  MSE = (1/n) * Σ(y_pred - y_true)²

Squaring the differences ensures the loss is always positive and penalises large errors disproportionately more than small ones.`,
    constraints: [
      'y_true and y_pred are numpy arrays of the same shape',
      'Return a single scalar float',
      'Do not use sklearn or other ML libraries',
    ],
    examples: [
      {
        input: 'mse(np.array([1, 2, 3]), np.array([1.5, 2.5, 3.5]))',
        output: '0.25',
        explanation: 'Each error is 0.5, squared = 0.25, mean = 0.25',
      },
      { input: 'mse(np.array([0, 0]), np.array([0, 0]))', output: '0.0' },
    ],
    starter_code: `import numpy as np\n\ndef mse(y_true, y_pred):\n    # TODO: compute Mean Squared Error\n    # mean of (y_pred - y_true)^2\n    pass\n`,
    test_input: `import numpy as np\nprint(mse(np.array([1, 2]), np.array([1.5, 2.5])))`,
    acceptance: '71.2%',
    hints: [
      'Use np.mean() to average',
      'Square the difference: (y_pred - y_true) ** 2',
    ],
  },
  {
    slug: 'numpy-dot-product',
    index: 5,
    title: 'Dot Product',
    difficulty: 'Easy',
    topics: ['Linear Algebra', 'NumPy'],
    description: `Compute the dot product of two 1D numpy arrays.

The dot product is a fundamental linear algebra operation. For two vectors a and b:

  a · b = Σ(aᵢ × bᵢ)

It multiplies corresponding elements and sums the results. The dot product appears everywhere in ML: computing weighted sums in neurons, measuring vector similarity in embeddings, and matrix multiplication in attention.`,
    constraints: [
      'Input arrays a and b have the same length',
      'Return a scalar (single number)',
      'Do not loop manually — use numpy operations',
    ],
    examples: [
      {
        input: 'dot_product(np.array([1, 2]), np.array([3, 4]))',
        output: '11',
        explanation: '(1×3) + (2×4) = 3 + 8 = 11',
      },
      { input: 'dot_product(np.array([0, 1, 0]), np.array([5, 6, 7]))', output: '6' },
    ],
    starter_code: `import numpy as np\n\ndef dot_product(a, b):\n    # TODO: compute dot product of a and b\n    pass\n`,
    test_input: `import numpy as np\nprint(dot_product(np.array([1, 2]), np.array([3, 4])))`,
    acceptance: '88.9%',
    hints: [
      'np.dot(a, b) computes the dot product',
      'You can also use the @ operator: a @ b',
    ],
  },
  {
    slug: 'ml-softmax',
    index: 6,
    title: 'Softmax Function',
    difficulty: 'Medium',
    topics: ['Activation Functions', 'NLP', 'Transformers'],
    description: `Implement the numerically stable softmax function.

Softmax converts a vector of raw scores (logits) into a probability distribution that sums to 1:

  softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)

For numerical stability, subtract the maximum value before exponentiation:

  softmax(x) = exp(x - max(x)) / Σ exp(x - max(x))

Softmax is used in the final layer of multi-class classifiers and inside every attention mechanism in transformers.`,
    constraints: [
      'Input x is a 1D numpy array of any length',
      'Output must sum to exactly 1.0',
      'Must be numerically stable (no overflow for large inputs like 1000)',
      'All output values must be positive',
    ],
    examples: [
      { input: 'softmax(np.array([1, 2, 3]))', output: 'array([0.0900, 0.2447, 0.6652])' },
      {
        input: 'softmax(np.array([0, 0, 0]))',
        output: 'array([0.3333, 0.3333, 0.3333])',
        explanation: 'Equal logits → equal probabilities',
      },
    ],
    starter_code: `import numpy as np\n\ndef softmax(x):\n    # TODO: implement numerically stable softmax\n    # Subtract max before exp to prevent overflow\n    pass\n`,
    test_input: `import numpy as np\nresult = softmax(np.array([1, 2, 3]))\nprint(np.round(result, 4))\nprint(round(float(result.sum()), 6))`,
    acceptance: '66.8%',
    hints: [
      'Subtract max first: exp_x = np.exp(x - np.max(x))',
      'Then normalise: return exp_x / np.sum(exp_x)',
    ],
  },
  {
    slug: 'ml-normalize',
    index: 7,
    title: 'Min-Max Normalization',
    difficulty: 'Medium',
    topics: ['Preprocessing', 'NumPy'],
    description: `Implement Min-Max normalization to scale an array to [0, 1].

Normalization is a crucial preprocessing step — it ensures all features live on the same scale so no single feature dominates the model due to its magnitude:

  x_norm = (x - x_min) / (x_max - x_min)

After normalization, the minimum value becomes 0 and the maximum becomes 1.`,
    constraints: [
      'Input x is a numpy array',
      'The minimum element must map to exactly 0.0',
      'The maximum element must map to exactly 1.0',
      'Handle edge case: if all values are equal, return zeros',
    ],
    examples: [
      { input: 'normalize(np.array([0, 5, 10]))', output: 'array([0., 0.5, 1.])' },
      { input: 'normalize(np.array([2, 4, 6, 8]))', output: 'array([0.   , 0.333, 0.667, 1.   ])' },
    ],
    starter_code: `import numpy as np\n\ndef normalize(x):\n    # TODO: apply min-max normalization to range [0, 1]\n    pass\n`,
    test_input: `import numpy as np\nprint(normalize(np.array([0, 5, 10])))`,
    acceptance: '68.1%',
    hints: [
      'Use x.min() and x.max()',
      'Guard against division by zero: if max == min, return zeros',
    ],
  },
  {
    slug: 'ml-cross-entropy',
    index: 8,
    title: 'Binary Cross-Entropy Loss',
    difficulty: 'Medium',
    topics: ['Loss Functions', 'NumPy'],
    description: `Implement binary cross-entropy loss.

Cross-entropy is the standard loss for classification. For binary classification:

  BCE = -(1/n) * Σ[ y * log(p) + (1-y) * log(1-p) ]

Where y is the true label (0 or 1) and p is the predicted probability.

This loss penalises confident wrong predictions exponentially more than uncertain ones. Always clip predictions before taking log to avoid log(0) = −∞.`,
    constraints: [
      'y_true contains only 0s and 1s',
      'y_pred contains probabilities in (0, 1)',
      'Clip predictions: use np.clip(y_pred, 1e-7, 1 - 1e-7)',
      'Return a single scalar',
    ],
    examples: [
      {
        input: 'bce(np.array([1, 0]), np.array([0.9, 0.1]))',
        output: '0.1054',
        explanation: 'Near-perfect predictions → low loss',
      },
      {
        input: 'bce(np.array([1, 0]), np.array([0.1, 0.9]))',
        output: '2.3026',
        explanation: 'Wrong confident predictions → very high loss',
      },
    ],
    starter_code: `import numpy as np\n\ndef bce(y_true, y_pred):\n    # TODO: implement binary cross-entropy loss\n    # Clip first: p = np.clip(y_pred, 1e-7, 1 - 1e-7)\n    pass\n`,
    test_input: `import numpy as np\nprint(round(float(bce(np.array([1, 0]), np.array([0.9, 0.1]))), 4))`,
    acceptance: '62.3%',
    hints: [
      'Always clip before log: np.clip(y_pred, 1e-7, 1 - 1e-7)',
      'Use np.log() for natural logarithm',
      'Return -np.mean(...)',
    ],
  },
  {
    slug: 'ml-gradient-descent',
    index: 9,
    title: 'Gradient Descent Step',
    difficulty: 'Medium',
    topics: ['Optimization', 'NumPy'],
    description: `Implement a single gradient descent parameter update.

Gradient descent is the backbone of neural network training. Each step nudges the parameters in the direction that reduces the loss:

  θ_new = θ - α * ∇L(θ)

Where θ are the parameters, α is the learning rate, and ∇L is the gradient of the loss with respect to θ.`,
    constraints: [
      'params and grads are numpy arrays of the same shape',
      'learning_rate is a positive scalar',
      'Return the new parameters — do not modify params in-place',
    ],
    examples: [
      {
        input: 'gradient_step(np.array([1.0, 2.0]), np.array([0.5, -0.3]), lr=0.1)',
        output: 'array([0.95, 2.03])',
        explanation: '[1.0 - 0.1*0.5, 2.0 - 0.1*(-0.3)] = [0.95, 2.03]',
      },
    ],
    starter_code: `import numpy as np\n\ndef gradient_step(params, grads, lr=0.01):\n    # TODO: perform one gradient descent update\n    # params_new = params - lr * grads\n    pass\n`,
    test_input: `import numpy as np\nprint(gradient_step(np.array([1.0, 2.0]), np.array([0.5, -0.3]), lr=0.1))`,
    acceptance: '81.6%',
    hints: ['Just return params - lr * grads'],
  },
  {
    slug: 'ml-attention',
    index: 10,
    title: 'Scaled Dot-Product Attention',
    difficulty: 'Hard',
    topics: ['Transformers', 'Attention', 'Linear Algebra'],
    description: `Implement scaled dot-product attention from "Attention Is All You Need".

The attention function maps queries and key-value pairs to outputs:

  Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V

Steps:
1. Compute raw scores: Q @ Kᵀ
2. Scale by √d_k to prevent softmax saturation
3. Apply softmax to get attention weights (each row sums to 1)
4. Multiply weights by V to get the output

This is the core operation inside every transformer model.`,
    constraints: [
      'Q shape: (seq_len, d_k)',
      'K shape: (seq_len, d_k)',
      'V shape: (seq_len, d_v)',
      'Return both the output (seq_len, d_v) and attention weights (seq_len, seq_len)',
      'Use numerically stable softmax',
    ],
    examples: [
      {
        input: 'output, weights = attention(Q, K, V)  # Q,K,V shape (4, 8)',
        output: 'output.shape == (4, 8), weights.shape == (4, 4)',
      },
      {
        input: 'weights.sum(axis=-1)',
        output: 'array([1., 1., 1., 1.])',
        explanation: 'Each row of attention weights sums to 1',
      },
    ],
    starter_code: `import numpy as np\n\ndef softmax(x, axis=-1):\n    e = np.exp(x - np.max(x, axis=axis, keepdims=True))\n    return e / e.sum(axis=axis, keepdims=True)\n\ndef attention(Q, K, V):\n    \"\"\"\n    Args:\n        Q: (seq_len, d_k)\n        K: (seq_len, d_k)\n        V: (seq_len, d_v)\n    Returns:\n        output: (seq_len, d_v)\n        weights: (seq_len, seq_len)\n    \"\"\"\n    d_k = Q.shape[-1]\n    # TODO: implement scaled dot-product attention\n    pass\n`,
    test_input: `import numpy as np\nnp.random.seed(42)\nQ = np.random.randn(4, 8)\nK = np.random.randn(4, 8)\nV = np.random.randn(4, 8)\noutput, weights = attention(Q, K, V)\nprint("output shape:", output.shape)\nprint("weights shape:", weights.shape)\nprint("weights sum:", np.round(weights.sum(axis=-1), 4))`,
    acceptance: '38.7%',
    hints: [
      'Step 1: scores = Q @ K.T',
      'Step 2: scaled = scores / np.sqrt(d_k)',
      'Step 3: weights = softmax(scaled)',
      'Step 4: output = weights @ V',
    ],
  },
];

export function getProblemBySlug(slug: string): Problem | undefined {
  return PROBLEMS.find(p => p.slug === slug);
}

export function getProblemIndex(slug: string): number {
  return PROBLEMS.findIndex(p => p.slug === slug);
}
