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
  {
    slug: 'stats-mean-var-std',
    index: 11,
    title: 'Mean, Variance, and Std Dev',
    difficulty: 'Easy',
    topics: ['Statistics', 'NumPy'],
    description: `Implement functions to compute the mean, variance, and standard deviation from scratch.

Given an array of numbers $X$:
1. Mean ($\mu$) = $\frac{1}{N} \sum x_i$
2. Variance ($\sigma^2$) = $\frac{1}{N} \sum (x_i - \mu)^2$
3. Standard Deviation ($\sigma$) = $\sqrt{\sigma^2}$

Use population variance (divide by $N$, not $N-1$).`,
    constraints: [
      'Input x is a 1D numpy array',
      'Return a tuple of (mean, variance, std)',
      'Do not use np.mean, np.var, or np.std'
    ],
    examples: [
      { input: 'stats_basics(np.array([2, 4, 4, 4, 5, 5, 7, 9]))', output: '(5.0, 4.0, 2.0)' }
    ],
    starter_code: `import numpy as np\n\ndef stats_basics(x):\n    # TODO: compute mean, var, std without np.mean/var/std\n    pass\n`,
    test_input: `import numpy as np\nprint(tuple(np.round(stats_basics(np.array([2, 4, 4, 4, 5, 5, 7, 9])), 4)))`,
    acceptance: '85.2%',
    hints: [
      'Use np.sum(x) / len(x) for the mean',
      'For variance, sum the squared differences from the mean and divide by N'
    ]
  },
  {
    slug: 'stats-covariance-correlation',
    index: 12,
    title: 'Covariance & Correlation',
    difficulty: 'Medium',
    topics: ['Statistics', 'NumPy'],
    description: `Compute the covariance and Pearson correlation coefficient between two arrays $X$ and $Y$.

Covariance measures the joint variability:
$Cov(X, Y) = \frac{1}{N} \sum (x_i - \mu_X)(y_i - \mu_Y)$

Correlation normalizes covariance to the range $[-1, 1]$:
$\rho = \frac{Cov(X, Y)}{\sigma_X \sigma_Y}$

Use population parameters ($1/N$).`,
    constraints: [
      'Inputs x and y are 1D arrays of the same length',
      'Return a tuple of (covariance, correlation)',
      'Do not use np.cov or np.corrcoef'
    ],
    examples: [
      { input: 'cov_corr(np.array([1, 2, 3]), np.array([3, 1, 2]))', output: '(-0.1667, -0.25)' }
    ],
    starter_code: `import numpy as np\n\ndef cov_corr(x, y):\n    # TODO: compute covariance and Pearson correlation\n    pass\n`,
    test_input: `import numpy as np\nprint(tuple(np.round(cov_corr(np.array([1, 2, 3]), np.array([3, 1, 2])), 4)))`,
    acceptance: '78.5%',
    hints: [
      'First compute the means of x and y',
      'Then compute the deviations (x - mean_x) and (y - mean_y)',
      'Covariance is the mean of the product of deviations'
    ]
  },
  {
    slug: 'stats-z-score',
    index: 13,
    title: 'Z-Score Standardization',
    difficulty: 'Easy',
    topics: ['Statistics', 'Preprocessing', 'NumPy'],
    description: `Implement Z-score standardization (also known as standard scaling).

Standardization transforms the data to have a mean of 0 and standard deviation of 1:
$z = \frac{x - \mu}{\sigma}$

This is crucial for ML algorithms sensitive to scale, like PCA, k-NN, and gradient descent.`,
    constraints: [
      'Input x is a 1D numpy array',
      'Return the standardized array',
      'If std is 0 (all elements equal), return an array of zeros to avoid division by zero'
    ],
    examples: [
      { input: 'z_score(np.array([1, 2, 3]))', output: 'array([-1.2247,  0.,  1.2247])' }
    ],
    starter_code: `import numpy as np\n\ndef z_score(x):\n    # TODO: standardize x to mean 0, std 1\n    pass\n`,
    test_input: `import numpy as np\nprint(np.round(z_score(np.array([1, 2, 3])), 4))`,
    acceptance: '90.1%',
    hints: [
      'Use np.mean(x) and np.std(x)',
      'Check if std == 0 before dividing'
    ]
  },
  {
    slug: 'stats-normal-pdf',
    index: 14,
    title: 'Normal Distribution PDF',
    difficulty: 'Easy',
    topics: ['Statistics', 'NumPy'],
    description: `Compute the Probability Density Function (PDF) of a Normal (Gaussian) distribution.

The PDF gives the relative likelihood of a continuous random variable taking on a specific value:
$f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2}$`,
    constraints: [
      'Input x is a scalar or numpy array',
      'mu is the mean, sigma is the standard deviation (sigma > 0)',
      'Return the PDF values for x'
    ],
    examples: [
      { input: 'normal_pdf(0, mu=0, sigma=1)', output: '0.3989', explanation: 'Standard normal at mean' }
    ],
    starter_code: `import numpy as np\n\ndef normal_pdf(x, mu=0.0, sigma=1.0):\n    # TODO: compute the Gaussian PDF\n    pass\n`,
    test_input: `import numpy as np\nprint(np.round(normal_pdf(np.array([-1, 0, 1]), 0, 1), 4))`,
    acceptance: '88.3%',
    hints: [
      'Use np.pi for π and np.exp() for the exponential function'
    ]
  },
  {
    slug: 'stats-bayes-theorem',
    index: 15,
    title: "Bayes' Theorem",
    difficulty: 'Medium',
    topics: ['Probability', 'Math'],
    description: `Implement Bayes' theorem for a discrete event.

Given:
- $P(A)$ : Prior probability of event A
- $P(B|A)$ : Probability of B given A (True Positive Rate)
- $P(B|\neg A)$ : Probability of B given not A (False Positive Rate)

Compute the posterior probability $P(A|B)$:
$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$
where $P(B) = P(B|A)P(A) + P(B|\neg A)P(\neg A)$`,
    constraints: [
      'Inputs are scalars between 0 and 1',
      'Return the posterior probability P(A|B)'
    ],
    examples: [
      {
        input: 'bayes(p_a=0.01, p_b_given_a=0.9, p_b_given_not_a=0.05)',
        output: '0.1538',
        explanation: 'Testing positive for a 1% disease with 90% sensitivity and 5% false positive rate yields only a 15% chance you have it.'
      }
    ],
    starter_code: `def bayes(p_a, p_b_given_a, p_b_given_not_a):\n    # TODO: compute P(A|B)\n    pass\n`,
    test_input: `print(round(bayes(0.01, 0.9, 0.05), 4))`,
    acceptance: '75.6%',
    hints: [
      'First calculate P(not A) = 1 - p_a',
      'Then calculate the total probability of B: P(B)'
    ]
  },
  {
    slug: 'stats-bootstrap-ci',
    index: 16,
    title: 'Bootstrap Sample Mean CI',
    difficulty: 'Hard',
    topics: ['Statistics', 'NumPy'],
    description: `Estimate the 95% confidence interval of the mean using bootstrap resampling.

Steps:
1. Draw \`n_bootstraps\` samples from the data *with replacement*, each of the same size as the original data.
2. Compute the mean of each bootstrap sample.
3. Find the 2.5th and 97.5th percentiles of these bootstrap means to form the 95% CI.`,
    constraints: [
      'Input x is a 1D numpy array',
      'Use np.random.choice for resampling',
      'Set random seed to 42 for reproducibility: np.random.seed(42)',
      'Return (lower_bound, upper_bound)'
    ],
    examples: [
      { input: 'bootstrap_ci(np.array([1, 2, 3, 4, 5]), n_bootstraps=1000)', output: '(1.8, 4.2)' }
    ],
    starter_code: `import numpy as np\n\ndef bootstrap_ci(x, n_bootstraps=1000):\n    np.random.seed(42)\n    # TODO: implement bootstrap CI\n    pass\n`,
    test_input: `import numpy as np\nprint(tuple(np.round(bootstrap_ci(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])), 4)))`,
    acceptance: '55.2%',
    hints: [
      'Use np.random.choice(x, size=len(x), replace=True) in a loop or vectorized',
      'Use np.percentile(means, [2.5, 97.5]) to get the bounds'
    ]
  },
  {
    slug: 'stats-t-test',
    index: 17,
    title: 'Two-Sample T-Test Statistic',
    difficulty: 'Medium',
    topics: ['Statistics', 'NumPy'],
    description: `Compute the t-statistic for a two-sample t-test (independent, equal variances assumed).

The t-statistic measures the difference between two group means relative to the spread of the data:
$t = \frac{\bar{X}_1 - \bar{X}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$

Where $s_p$ is the pooled standard deviation:
$s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$

Note: Use sample variance ($N-1$) for $s_1^2$ and $s_2^2$.`,
    constraints: [
      'Inputs x1 and x2 are 1D numpy arrays',
      'Return the t-statistic',
      'Use sample variance: np.var(x, ddof=1)'
    ],
    examples: [
      { input: 't_statistic(np.array([1, 2, 3]), np.array([4, 5, 6]))', output: '-3.6742' }
    ],
    starter_code: `import numpy as np\n\ndef t_statistic(x1, x2):\n    # TODO: compute the two-sample t-statistic\n    pass\n`,
    test_input: `import numpy as np\nprint(round(t_statistic(np.array([1, 2, 3]), np.array([4, 5, 6])), 4))`,
    acceptance: '64.9%',
    hints: [
      'Remember to use ddof=1 for sample variance: np.var(x, ddof=1)',
      'Calculate the pooled variance first, then plug it into the t-statistic formula'
    ]
  },
  {
    slug: 'stats-mle-gaussian',
    index: 18,
    title: 'Maximum Likelihood Estimate (MLE)',
    difficulty: 'Easy',
    topics: ['Statistics', 'Machine Learning'],
    description: `Find the Maximum Likelihood Estimate (MLE) for the parameters of a Gaussian distribution given a dataset.

For a normal distribution, the MLE for the mean ($\mu$) and variance ($\sigma^2$) are simply the sample mean and the population variance (biased variance, dividing by $N$).

Given an array of observations $X$, return the MLE for $\mu$ and $\sigma^2$.`,
    constraints: [
      'Input X is a 1D numpy array',
      'Return a tuple (mu_mle, var_mle)',
      'var_mle must be the population variance (ddof=0)'
    ],
    examples: [
      { input: 'mle_gaussian(np.array([2, 4, 6, 8]))', output: '(5.0, 5.0)' }
    ],
    starter_code: `import numpy as np\n\ndef mle_gaussian(x):\n    # TODO: compute MLE for mean and variance\n    pass\n`,
    test_input: `import numpy as np\nprint(tuple(np.round(mle_gaussian(np.array([2, 4, 6, 8])), 4)))`,
    acceptance: '92.1%',
    hints: [
      'The MLE for mean is just np.mean(x)',
      'The MLE for variance is just np.var(x)'
    ]
  },
  {
    slug: 'stats-kl-divergence',
    index: 19,
    title: 'Entropy & KL Divergence',
    difficulty: 'Medium',
    topics: ['Information Theory', 'NumPy'],
    description: `Compute the discrete entropy of a distribution $P$ and the Kullback-Leibler (KL) divergence from $Q$ to $P$.

1. Entropy (in bits): $H(P) = -\sum P(i) \log_2 P(i)$
2. KL Divergence (in bits): $D_{KL}(P || Q) = \sum P(i) \log_2 \frac{P(i)}{Q(i)}$

Handle 0 probabilities by adding a small epsilon ($1e-9$) before taking the log.`,
    constraints: [
      'P and Q are 1D numpy arrays representing probability distributions (sum to 1)',
      'Return a tuple (entropy, kl_divergence)',
      'Use log base 2 (np.log2)',
      'Clip probabilities to [1e-9, 1.0] before logging'
    ],
    examples: [
      {
        input: 'entropy_kl(np.array([0.5, 0.5]), np.array([0.1, 0.9]))',
        output: '(1.0, 0.7369)',
        explanation: 'Entropy of uniform coin is 1 bit.'
      }
    ],
    starter_code: `import numpy as np\n\ndef entropy_kl(P, Q):\n    # TODO: compute entropy of P and KL divergence D_KL(P || Q)\n    pass\n`,
    test_input: `import numpy as np\nprint(tuple(np.round(entropy_kl(np.array([0.5, 0.5]), np.array([0.1, 0.9])), 4)))`,
    acceptance: '61.4%',
    hints: [
      'np.clip(P, 1e-9, 1) prevents log(0)',
      'KL divergence is np.sum(P * np.log2(P / Q_clipped))'
    ]
  },
  {
    slug: 'stats-log-sum-exp',
    index: 20,
    title: 'Log-Sum-Exp Trick',
    difficulty: 'Medium',
    topics: ['Numerical Methods', 'NumPy'],
    description: `Implement the Log-Sum-Exp function in a numerically stable way.

When computing $\log(\sum e^{x_i})$, large values of $x_i$ will cause $e^{x_i}$ to overflow to infinity.
The log-sum-exp trick avoids this:

$\log(\sum e^{x_i}) = a + \log(\sum e^{x_i - a})$

Where $a = \max(x)$. This ensures the largest exponent is $e^0 = 1$, preventing overflow.`,
    constraints: [
      'Input x is a 1D numpy array',
      'Return a scalar',
      'Must not overflow for large inputs (e.g., x=[1000, 1000])'
    ],
    examples: [
      { input: 'log_sum_exp(np.array([1, 2, 3]))', output: '3.4076' },
      { input: 'log_sum_exp(np.array([1000, 1000]))', output: '1000.6931' }
    ],
    starter_code: `import numpy as np\n\ndef log_sum_exp(x):\n    # TODO: implement numerically stable log-sum-exp\n    pass\n`,
    test_input: `import numpy as np\nprint(round(float(log_sum_exp(np.array([1000, 1001, 1002]))), 4))`,
    acceptance: '73.2%',
    hints: [
      'Find the maximum: a = np.max(x)',
      'Compute the sum of exp(x - a)',
      'Return a + np.log(sum_exp)'
    ]
  },
  {
    slug: 'ml-train-test-split',
    index: 21,
    title: 'Train/Test Split',
    difficulty: 'Easy',
    topics: ['Preprocessing', 'NumPy'],
    description: `Implement a function to split data into training and testing sets.

Given features $X$ and labels $y$, split them randomly so that the test set contains a specific proportion of the data. To ensure reproducibility, set the random seed before shuffling.

Steps:
1. Set the random seed to \`random_state\`.
2. Generate a random permutation of indices from $0$ to $N-1$.
3. Split the indices into train and test parts based on \`test_size\`.
4. Return \`(X_train, X_test, y_train, y_test)\`.`,
    constraints: [
      'Inputs X and y are numpy arrays of the same length (dim 0)',
      'test_size is a float between 0.0 and 1.0 (e.g., 0.2)',
      'random_state is an integer for np.random.seed()'
    ],
    examples: [
      { input: 'train_test_split(X, y, test_size=0.2, random_state=42)', output: '(X_train, X_test, y_train, y_test)' }
    ],
    starter_code: `import numpy as np\n\ndef train_test_split(X, y, test_size=0.2, random_state=42):\n    # TODO: implement train/test split\n    pass\n`,
    test_input: `import numpy as np\nX = np.arange(10)\ny = np.arange(10) * 2\n_, X_test, _, y_test = train_test_split(X, y, 0.2, 42)\nprint("X_test:", X_test)\nprint("y_test:", y_test)`,
    acceptance: '89.4%',
    hints: [
      'Use np.random.seed(random_state) first',
      'np.random.permutation(len(X)) gives shuffled indices',
      'test_count = int(len(X) * test_size)'
    ]
  },
  {
    slug: 'ml-linear-regression-normal',
    index: 22,
    title: 'Linear Regression (Normal Equation)',
    difficulty: 'Medium',
    topics: ['Supervised Learning', 'Linear Algebra', 'NumPy'],
    description: `Implement exact Linear Regression using the Normal Equation.

The normal equation finds the optimal weights $\theta$ that minimize the Mean Squared Error in a single step:
$\theta = (X^T X)^{-1} X^T y$

Important: You must prepend a column of 1s to $X$ to account for the bias (intercept) term before computing $\theta$.`,
    constraints: [
      'Input X is a 2D numpy array of shape (n_samples, n_features)',
      'Input y is a 1D numpy array of shape (n_samples,)',
      'Return the optimal weights (including bias) as a 1D array of shape (n_features + 1,)'
    ],
    examples: [
      { input: 'lin_reg_normal(X, y)', output: 'array([bias, w1, w2, ...])' }
    ],
    starter_code: `import numpy as np\n\ndef lin_reg_normal(X, y):\n    # TODO: solve for theta using the normal equation\n    pass\n`,
    test_input: `import numpy as np\nX = np.array([[1], [2], [3]])\ny = np.array([2, 4, 6])\nprint(np.round(lin_reg_normal(X, y), 4))`,
    acceptance: '71.5%',
    hints: [
      'Add a column of ones to X using np.c_[np.ones(len(X)), X] or np.concatenate',
      'Use np.linalg.inv() to invert the matrix'
    ]
  },
  {
    slug: 'ml-linear-regression-gd',
    index: 23,
    title: 'Linear Regression (Gradient Descent)',
    difficulty: 'Medium',
    topics: ['Supervised Learning', 'Optimization', 'NumPy'],
    description: `Implement Linear Regression using Gradient Descent.

Unlike the normal equation, gradient descent iteratively updates weights to minimize MSE.
For $N$ samples, the gradient of the MSE with respect to weights $\theta$ is:
$\nabla J(\theta) = \frac{2}{N} X^T (X\theta - y)$

Steps for each iteration:
1. Compute predictions: $\hat{y} = X\theta$
2. Compute gradient: $g = \frac{2}{N} X^T (\hat{y} - y)$
3. Update weights: $\theta = \theta - \alpha g$

Prepend a column of 1s to $X$ for the bias. Initialize $\theta$ to zeros.`,
    constraints: [
      'Input X is a 2D array, y is a 1D array',
      'lr is learning rate alpha, epochs is the number of iterations',
      'Initialize weights to zeros',
      'Return the final weights array'
    ],
    examples: [
      { input: 'lin_reg_gd(X, y, lr=0.01, epochs=1000)', output: 'array([bias, w1, w2, ...])' }
    ],
    starter_code: `import numpy as np\n\ndef lin_reg_gd(X, y, lr=0.01, epochs=1000):\n    # TODO: implement linear regression via gradient descent\n    pass\n`,
    test_input: `import numpy as np\nX = np.array([[1], [2], [3]])\ny = np.array([2, 4, 6])\nprint(np.round(lin_reg_gd(X, y, 0.01, 1000), 2))`,
    acceptance: '64.8%',
    hints: [
      'Add a column of ones to X first',
      'Initialize theta = np.zeros(X.shape[1])',
      'Loop for epochs, updating theta each time'
    ]
  },
  {
    slug: 'ml-logistic-regression-gd',
    index: 24,
    title: 'Logistic Regression (Gradient Descent)',
    difficulty: 'Medium',
    topics: ['Supervised Learning', 'Classification', 'NumPy'],
    description: `Implement Logistic Regression using Gradient Descent.

Logistic regression uses the sigmoid function to predict probabilities:
$\hat{y} = \sigma(X\theta) = \frac{1}{1 + e^{-X\theta}}$

The gradient of the Binary Cross-Entropy loss with respect to $\theta$ is beautifully similar to linear regression:
$\nabla J(\theta) = \frac{1}{N} X^T (\hat{y} - y)$

Steps:
1. Add bias column to $X$, init $\theta$ to zeros.
2. For each epoch, compute $\hat{y} = \sigma(X\theta)$.
3. Compute gradient and update $\theta$.`,
    constraints: [
      'y contains binary labels (0 or 1)',
      'Return the final weights array',
      'Use the standard sigmoid function'
    ],
    examples: [
      { input: 'log_reg_gd(X, y, lr=0.1, epochs=1000)', output: 'array([bias, w1, w2, ...])' }
    ],
    starter_code: `import numpy as np\n\ndef sigmoid(z):\n    return 1 / (1 + np.exp(-z))\n\ndef log_reg_gd(X, y, lr=0.1, epochs=1000):\n    # TODO: implement logistic regression via gradient descent\n    pass\n`,
    test_input: `import numpy as np\nX = np.array([[0.1], [0.5], [2.0], [3.0]])\ny = np.array([0, 0, 1, 1])\nprint(np.round(log_reg_gd(X, y, 0.5, 1000), 2))`,
    acceptance: '61.2%',
    hints: [
      'The gradient does not have a factor of 2 like MSE does, it is just (1/N) * X.T @ (y_hat - y)'
    ]
  },
  {
    slug: 'ml-classification-metrics',
    index: 25,
    title: 'Classification Metrics',
    difficulty: 'Easy',
    topics: ['Evaluation', 'Metrics'],
    description: `Implement Accuracy, Precision, Recall, and F1 Score for binary classification.

Given true labels $y$ and binary predictions $\hat{y}$:
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP)  (Of all positive predictions, how many were right?)
- **Recall**: TP / (TP + FN)  (Of all actual positives, how many did we find?)
- **F1 Score**: $2 \times \frac{Precision \times Recall}{Precision + Recall}$ (Harmonic mean)`,
    constraints: [
      'Inputs are 1D arrays of 0s and 1s',
      'Return a tuple: (accuracy, precision, recall, f1)',
      'Handle division by zero (return 0.0 for metrics if denominator is 0)'
    ],
    examples: [
      { input: 'metrics(np.array([1,0,1,1]), np.array([1,0,0,1]))', output: '(0.75, 1.0, 0.6667, 0.8)' }
    ],
    starter_code: `import numpy as np\n\ndef calc_metrics(y_true, y_pred):\n    # TODO: compute accuracy, precision, recall, f1\n    pass\n`,
    test_input: `import numpy as np\ny_true = np.array([1, 0, 1, 1, 0, 1])\ny_pred = np.array([1, 0, 0, 1, 1, 1])\nprint(tuple(np.round(calc_metrics(y_true, y_pred), 4)))`,
    acceptance: '82.7%',
    hints: [
      'Count True Positives (TP) where true==1 and pred==1',
      'Use np.sum((y_true == 1) & (y_pred == 1))'
    ]
  },
  {
    slug: 'ml-roc-auc',
    index: 26,
    title: 'ROC Curve & AUC',
    difficulty: 'Hard',
    topics: ['Evaluation', 'NumPy'],
    description: `Compute the Receiver Operating Characteristic (ROC) curve and Area Under Curve (AUC).

The ROC curve plots True Positive Rate (TPR) vs False Positive Rate (FPR) at various threshold settings.
1. Sort the true labels in descending order by the predicted probabilities.
2. Iterate through the sorted labels. Each step changes the threshold.
3. Compute TPR and FPR at each step.
4. Calculate AUC using the trapezoidal rule.`,
    constraints: [
      'y_true contains 0s and 1s',
      'y_prob contains probabilities between 0 and 1',
      'Return the AUC (a single scalar float)'
    ],
    examples: [
      { input: 'roc_auc(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.4]))', output: '1.0' }
    ],
    starter_code: `import numpy as np\n\ndef roc_auc(y_true, y_prob):\n    # TODO: compute the exact ROC AUC\n    pass\n`,
    test_input: `import numpy as np\ny_true = np.array([0, 1, 0, 1])\ny_prob = np.array([0.1, 0.4, 0.35, 0.8])\nprint(round(roc_auc(y_true, y_prob), 4))`,
    acceptance: '41.3%',
    hints: [
      'Sort indices: order = np.argsort(y_prob)[::-1]',
      'Iterate through the sorted y_true. If 1, increase TP. If 0, increase FP.',
      'AUC is the sum of (FP_diff / Total_Neg) * (TP / Total_Pos)'
    ]
  },
  {
    slug: 'ml-k-fold-cv',
    index: 27,
    title: 'K-Fold Cross-Validation',
    difficulty: 'Medium',
    topics: ['Evaluation', 'NumPy'],
    description: `Generate indices for K-Fold Cross-Validation.

K-Fold splits the dataset into $K$ non-overlapping folds. In each iteration, one fold is used for testing and the remaining $K-1$ folds are used for training.

Return a list of $K$ tuples: \`[(train_indices_1, test_indices_1), ...]\`.`,
    constraints: [
      'N is the total number of samples',
      'k is the number of folds',
      'Folds should be as equal in size as possible (use integer division and remainder)',
      'Do not shuffle the indices; split them sequentially'
    ],
    examples: [
      { input: 'k_fold(N=5, k=2)', output: '[( [2,3,4], [0,1] ), ( [0,1], [2,3,4] )]' }
    ],
    starter_code: `import numpy as np\n\ndef k_fold(N, k):\n    # TODO: return list of (train_idx, test_idx) tuples\n    pass\n`,
    test_input: `for tr, te in k_fold(5, 2):\n    print(list(tr), list(te))`,
    acceptance: '67.9%',
    hints: [
      'Base fold size is N // k, with N % k folds having size + 1',
      'Keep track of the start and end index for each fold'
    ]
  },
  {
    slug: 'ml-knn-classifier',
    index: 28,
    title: 'K-Nearest Neighbours (k-NN)',
    difficulty: 'Easy',
    topics: ['Supervised Learning', 'NumPy'],
    description: `Implement the K-Nearest Neighbours classifier for a single test point.

To predict the class of a new point $x_{test}$:
1. Compute the Euclidean distance between $x_{test}$ and all training points in $X$.
2. Find the $k$ nearest training points.
3. Return the most common label among those $k$ points (majority vote).`,
    constraints: [
      'X is a 2D array, y is a 1D array of integer labels',
      'x_test is a 1D array representing a single test point',
      'In case of a tie in voting, returning any of the tied classes is acceptable'
    ],
    examples: [
      { input: 'knn(X, y, x_test=[0,0], k=3)', output: 'label_with_majority_vote' }
    ],
    starter_code: `import numpy as np\n\ndef knn(X, y, x_test, k=3):\n    # TODO: implement k-NN classification\n    pass\n`,
    test_input: `import numpy as np\nX = np.array([[1,1], [1.2,1.2], [5,5], [5.1,5.2]])\ny = np.array([0, 0, 1, 1])\nprint(knn(X, y, np.array([2,2]), k=3))`,
    acceptance: '85.4%',
    hints: [
      'Use np.linalg.norm(X - x_test, axis=1) for distances',
      'Use np.argsort(distances)[:k] to get the k nearest indices',
      'Use np.bincount() and np.argmax() to find the most common label'
    ]
  },
  {
    slug: 'ml-k-means',
    index: 29,
    title: 'K-Means Clustering',
    difficulty: 'Hard',
    topics: ['Unsupervised Learning', 'NumPy'],
    description: `Implement K-Means Clustering (Lloyd's Algorithm).

Partition the dataset into $K$ distinct clusters:
1. Initialize $K$ centroids randomly (for simplicity, pick $K$ random points from $X$).
2. Assign each point to the nearest centroid.
3. Update each centroid to be the mean of the points assigned to it.
4. Repeat until centroids do not change (or max_iters is reached).`,
    constraints: [
      'X is a 2D numpy array',
      'Return the final centroids and the array of cluster labels for each point',
      'Set np.random.seed(random_state) before initializing'
    ],
    examples: [
      { input: 'kmeans(X, k=2)', output: '(centroids, labels)' }
    ],
    starter_code: `import numpy as np\n\ndef kmeans(X, k, max_iters=100, random_state=42):\n    np.random.seed(random_state)\n    # TODO: implement K-Means clustering\n    pass\n`,
    test_input: `import numpy as np\nX = np.array([[1,1], [1.2,1.2], [5,5], [5.2,5.1]])\ncentroids, labels = kmeans(X, k=2)\nprint("labels:", labels)\nprint("centroids:", np.round(centroids, 2))`,
    acceptance: '53.1%',
    hints: [
      'Initial centroids: X[np.random.choice(X.shape[0], k, replace=False)]',
      'Compute distances between all points and all centroids using broadcasting'
    ]
  },
  {
    slug: 'ml-pca',
    index: 30,
    title: 'Principal Component Analysis (PCA)',
    difficulty: 'Medium',
    topics: ['Unsupervised Learning', 'Dimensionality Reduction', 'Linear Algebra'],
    description: `Implement PCA via eigendecomposition.

PCA projects data into a lower-dimensional subspace while preserving maximum variance.
1. Center the data by subtracting the mean of each feature.
2. Compute the covariance matrix of the centered data.
3. Compute the eigenvalues and eigenvectors of the covariance matrix.
4. Sort eigenvectors by eigenvalues in descending order.
5. Project the data onto the top $k$ eigenvectors.`,
    constraints: [
      'Return the projected 2D array of shape (N, k)',
      'Use np.linalg.eigh or np.linalg.eig'
    ],
    examples: [
      { input: 'pca(X, k=2)', output: 'projected_X' }
    ],
    starter_code: `import numpy as np\n\ndef pca(X, k):\n    # TODO: implement PCA\n    pass\n`,
    test_input: `import numpy as np\nX = np.array([[1, 2], [3, 4], [5, 6]])\nprint(np.round(np.abs(pca(X, k=1)), 4))`,
    acceptance: '62.0%',
    hints: [
      'Covariance matrix = np.cov(X_centered, rowvar=False)',
      'np.linalg.eigh(cov_matrix) returns sorted values/vectors, but in ascending order (reverse them)'
    ]
  },
  {
    slug: 'ml-decision-tree-gini',
    index: 31,
    title: 'Decision Tree Best Split (Gini)',
    difficulty: 'Hard',
    topics: ['Trees', 'NumPy'],
    description: `Find the best feature and threshold to split a dataset based on Gini Impurity.

The Gini Impurity of a node is $1 - \sum p_i^2$, where $p_i$ is the probability of class $i$.
To evaluate a split, compute the weighted average Gini of the left and right child nodes:
$Cost = \frac{N_{left}}{N} Gini(left) + \frac{N_{right}}{N} Gini(right)$

Iterate over all features and all unique values of those features to find the split that minimizes this cost.`,
    constraints: [
      'X is a 2D array of features, y is a 1D array of binary labels (0/1)',
      'Return a tuple: (best_feature_index, best_threshold)',
      'If no split improves impurity, return (None, None)'
    ],
    examples: [
      { input: 'best_split(X, y)', output: '(0, 2.5)' }
    ],
    starter_code: `import numpy as np\n\ndef gini(y):\n    if len(y) == 0: return 0.0\n    p = np.mean(y)\n    return 1.0 - (p**2 + (1-p)**2)\n\ndef best_split(X, y):\n    # TODO: iterate all features and thresholds to find the minimum weighted gini\n    pass\n`,
    test_input: `import numpy as np\nX = np.array([[1, 2], [2, 1], [4, 5], [5, 4]])\ny = np.array([0, 0, 1, 1])\nprint(best_split(X, y))`,
    acceptance: '32.5%',
    hints: [
      'For each feature (column), unique values serve as candidate thresholds',
      'Split condition: left_mask = X[:, feature] <= threshold',
      'Calculate weighted gini for the split. Keep track of the minimum.'
    ]
  },
  {
    slug: 'ml-gaussian-naive-bayes',
    index: 32,
    title: 'Gaussian Naive Bayes',
    difficulty: 'Medium',
    topics: ['Supervised Learning', 'Probability'],
    description: `Implement prediction for Gaussian Naive Bayes.

Naive Bayes assumes features are conditionally independent given the class.
For a continuous feature, we model it with a Gaussian distribution:
$P(x_i | y=c) = \frac{1}{\sqrt{2\pi\sigma_{c,i}^2}} e^{ - \frac{(x_i - \mu_{c,i})^2}{2\sigma_{c,i}^2} }$

Prediction rule (using log probabilities to prevent underflow):
$\hat{y} = \arg\max_c \left[ \log P(y=c) + \sum_i \log P(x_i | y=c) \right]$`,
    constraints: [
      'We provide you the fitted parameters: priors (dict of class: prob), means (dict of class: array of feature means), vars (dict of class: array of feature vars)',
      'x_test is a single 1D array',
      'Return the predicted class label (int)'
    ],
    examples: [
      { input: 'gnb_predict(x_test, priors, means, vars)', output: '1' }
    ],
    starter_code: `import numpy as np\n\ndef gnb_predict(x_test, priors, means, vars):\n    # TODO: predict the class for x_test\n    pass\n`,
    test_input: `import numpy as np\npriors = {0: 0.5, 1: 0.5}\nmeans = {0: np.array([0, 0]), 1: np.array([5, 5])}\nvars = {0: np.array([1, 1]), 1: np.array([1, 1])}\nprint(gnb_predict(np.array([4, 4]), priors, means, vars))`,
    acceptance: '58.8%',
    hints: [
      'Compute the log of the Gaussian PDF for each feature, and sum them up',
      'Add the log of the prior',
      'Return the class that yields the maximum value'
    ]
  },
  {
    slug: 'dl-backprop-single-neuron',
    index: 33,
    title: 'Backprop: Single Neuron',
    difficulty: 'Medium',
    topics: ['Deep Learning', 'Calculus'],
    description: `Implement the backward pass for a single neuron with sigmoid activation and MSE loss.

Forward pass:
$z = w \cdot x + b$
$a = \sigma(z)$
$L = (a - y)^2$

Using the chain rule, compute the gradients $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$:
$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \frac{\partial a}{\partial z} \frac{\partial z}{\partial w}$

Where:
- $\frac{\partial L}{\partial a} = 2(a - y)$
- $\frac{\partial a}{\partial z} = a(1 - a)$ (derivative of sigmoid)
- $\frac{\partial z}{\partial w} = x$
- $\frac{\partial z}{\partial b} = 1$`,
    constraints: [
      'Inputs w, b, x, y are scalars',
      'Return a tuple: (grad_w, grad_b)'
    ],
    examples: [
      { input: 'backprop_neuron(w=0.5, b=0.1, x=2.0, y=1.0)', output: '(grad_w, grad_b)' }
    ],
    starter_code: `import numpy as np\n\ndef sigmoid(z):\n    return 1 / (1 + np.exp(-z))\n\ndef backprop_neuron(w, b, x, y):\n    # TODO: compute the gradients of MSE loss with respect to w and b\n    pass\n`,
    test_input: `import numpy as np\nprint(tuple(np.round(backprop_neuron(0.5, 0.1, 2.0, 1.0), 4)))`,
    acceptance: '64.2%',
    hints: [
      'First compute the forward pass to get a',
      'Compute the derivative of the loss w.r.t a',
      'Multiply by the derivative of sigmoid w.r.t z',
      'Multiply by x for grad_w, and by 1 for grad_b'
    ]
  },
  {
    slug: 'dl-2-layer-mlp',
    index: 34,
    title: '2-Layer MLP (Forward + Backward)',
    difficulty: 'Hard',
    topics: ['Deep Learning', 'NumPy'],
    description: `Implement a full forward and backward pass for a 2-layer Multi-Layer Perceptron (MLP) for binary classification.

Architecture:
- Input $X \in \mathbb{R}^{N \times D_{in}}$
- Hidden Layer: $Z_1 = X W_1 + b_1$, $A_1 = \text{ReLU}(Z_1)$
- Output Layer: $Z_2 = A_1 W_2 + b_2$, $\hat{y} = \sigma(Z_2)$
- Loss: Binary Cross-Entropy

Return the gradients of all parameters: $dW_1, db_1, dW_2, db_2$.
Assume $dW$ includes the $1/N$ averaging factor (mean over the batch).`,
    constraints: [
      'Inputs: X, y, W1, b1, W2, b2',
      'Use ReLU for the hidden layer and Sigmoid for the output layer',
      'Return a dictionary: {"dW1": ..., "db1": ..., "dW2": ..., "db2": ...}'
    ],
    examples: [
      { input: 'mlp_pass(X, y, W1, b1, W2, b2)', output: 'dict of gradients' }
    ],
    starter_code: `import numpy as np\n\ndef mlp_pass(X, y, W1, b1, W2, b2):\n    N = X.shape[0]\n    # TODO: Forward pass\n    \n    # TODO: Backward pass (compute dW2, db2, dW1, db1)\n    \n    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}\n`,
    test_input: `import numpy as np\nnp.random.seed(42)\nX = np.random.randn(2, 3)\ny = np.array([[1], [0]])\nW1 = np.random.randn(3, 4)\nb1 = np.zeros(4)\nW2 = np.random.randn(4, 1)\nb2 = np.zeros(1)\ngrads = mlp_pass(X, y, W1, b1, W2, b2)\nprint(np.round(grads['dW2'].flatten(), 4))`,
    acceptance: '28.5%',
    hints: [
      'Derivative of BCE with sigmoid simplifies beautifully to: dZ2 = (y_hat - y) / N',
      'dW2 = A1.T @ dZ2, db2 = np.sum(dZ2, axis=0)',
      'dA1 = dZ2 @ W2.T',
      'dZ1 = dA1 * (Z1 > 0)',
      'dW1 = X.T @ dZ1, db1 = np.sum(dZ1, axis=0)'
    ]
  },
  {
    slug: 'dl-batch-norm',
    index: 35,
    title: 'Batch Normalization (Forward)',
    difficulty: 'Medium',
    topics: ['Deep Learning', 'NumPy'],
    description: `Implement the forward pass of Batch Normalization.

Batch norm stabilizes training by normalizing the inputs of a layer to have zero mean and unit variance across the batch dimension, then scaling and shifting them.

For a batch $X$:
1. $\mu_B = \frac{1}{N} \sum X_i$
2. $\sigma^2_B = \frac{1}{N} \sum (X_i - \mu_B)^2$
3. $\hat{X_i} = \frac{X_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}$
4. $Y_i = \gamma \hat{X_i} + \beta$`,
    constraints: [
      'Input X is a 2D array of shape (batch_size, num_features)',
      'gamma and beta are 1D arrays of shape (num_features,)',
      'epsilon is a small scalar to prevent division by zero (default 1e-5)',
      'Return the normalized and scaled array Y'
    ],
    examples: [
      { input: 'batch_norm(X, gamma, beta)', output: 'normalized_scaled_X' }
    ],
    starter_code: `import numpy as np\n\ndef batch_norm(X, gamma, beta, eps=1e-5):\n    # TODO: implement batch normalization forward pass\n    pass\n`,
    test_input: `import numpy as np\nX = np.array([[1.0, 2.0], [3.0, 4.0]])\ngamma = np.array([1.0, 1.0])\nbeta = np.array([0.0, 0.0])\nprint(np.round(batch_norm(X, gamma, beta), 4))`,
    acceptance: '76.4%',
    hints: [
      'Compute mean and variance along axis 0 (the batch dimension)',
      'Use keepdims=True to ensure proper broadcasting'
    ]
  },
  {
    slug: 'dl-layer-norm',
    index: 36,
    title: 'Layer Normalization',
    difficulty: 'Medium',
    topics: ['Deep Learning', 'Transformers', 'NumPy'],
    description: `Implement Layer Normalization.

Unlike Batch Norm, Layer Norm normalizes across the *feature* dimension for each individual sample, making it independent of batch size. It is the standard normalization used in Transformers.

For a single sample vector $x$:
1. $\mu = \frac{1}{D} \sum x_j$
2. $\sigma^2 = \frac{1}{D} \sum (x_j - \mu)^2$
3. $\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$
4. $y = \gamma \hat{x} + \beta$`,
    constraints: [
      'Input X is a 2D array of shape (batch_size, num_features)',
      'Compute mean and variance over the feature dimension (axis -1)',
      'gamma and beta are 1D arrays of shape (num_features,)',
      'Return the normalized array'
    ],
    examples: [
      { input: 'layer_norm(X, gamma, beta)', output: 'normalized_scaled_X' }
    ],
    starter_code: `import numpy as np\n\ndef layer_norm(X, gamma, beta, eps=1e-5):\n    # TODO: implement layer normalization\n    pass\n`,
    test_input: `import numpy as np\nX = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\ngamma = np.ones(3)\nbeta = np.zeros(3)\nprint(np.round(layer_norm(X, gamma, beta), 4))`,
    acceptance: '71.1%',
    hints: [
      'The only difference from BatchNorm is the axis. Compute mean and variance along axis -1 (the feature dimension).',
      'Use keepdims=True.'
    ]
  },
  {
    slug: 'dl-dropout',
    index: 37,
    title: 'Dropout',
    difficulty: 'Easy',
    topics: ['Deep Learning', 'Regularization'],
    description: `Implement Dropout, a regularization technique.

During training, dropout randomly zeroes out some of the elements of the input tensor with probability $p$.
To maintain the expected value of the activations, the remaining elements are scaled by $\frac{1}{1-p}$ (Inverted Dropout).

During evaluation (inference), dropout does nothing; the input is returned unchanged.`,
    constraints: [
      'Input X is a numpy array',
      'p is the dropout probability (0.0 to 1.0)',
      'is_training is a boolean flag',
      'Set np.random.seed(42) inside the function before generating the mask',
      'Return the modified array'
    ],
    examples: [
      { input: 'dropout(X, p=0.5, is_training=True)', output: 'array_with_half_zeros_scaled' }
    ],
    starter_code: `import numpy as np\n\ndef dropout(X, p, is_training=True):\n    np.random.seed(42)\n    # TODO: implement inverted dropout\n    pass\n`,
    test_input: `import numpy as np\nX = np.ones(10)\nprint(np.round(dropout(X, p=0.5, is_training=True), 2))`,
    acceptance: '84.9%',
    hints: [
      'If not is_training, just return X',
      'Generate a mask of booleans: mask = np.random.rand(*X.shape) > p',
      'Return (X * mask) / (1.0 - p)'
    ]
  },
  {
    slug: 'dl-conv2d',
    index: 38,
    title: 'Conv2D (Naive implementation)',
    difficulty: 'Hard',
    topics: ['Deep Learning', 'Computer Vision'],
    description: `Implement a naive 2D Convolution operation.

Given an input image and a 2D filter (kernel), slide the filter over the image and compute the element-wise multiplication and sum.

Assume:
- Stride is 1
- No padding (valid convolution)
- Single channel input and single channel filter

Output dimension: $(H - F + 1) \times (W - F + 1)$, where $F$ is the filter size.`,
    constraints: [
      'Input image is a 2D array of shape (H, W)',
      'Filter is a 2D array of shape (F, F)',
      'Use nested loops (no advanced NumPy vectorization required)',
      'Return the 2D output feature map'
    ],
    examples: [
      { input: 'conv2d(image, kernel)', output: 'feature_map' }
    ],
    starter_code: `import numpy as np\n\ndef conv2d(image, kernel):\n    # TODO: perform 2D convolution\n    pass\n`,
    test_input: `import numpy as np\nimg = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\nkernel = np.array([[1, 0], [0, -1]])\nprint(conv2d(img, kernel))`,
    acceptance: '49.8%',
    hints: [
      'Initialize an output array of zeros with shape (H-F+1, W-F+1)',
      'Iterate row i from 0 to H-F, and col j from 0 to W-F',
      'Extract the image patch: image[i:i+F, j:j+F]',
      'Compute np.sum(patch * kernel) and assign to output[i, j]'
    ]
  },
  {
    slug: 'dl-max-pooling',
    index: 39,
    title: 'Max Pooling 2D',
    difficulty: 'Medium',
    topics: ['Deep Learning', 'Computer Vision'],
    description: `Implement a 2D Max Pooling operation.

Max pooling reduces the spatial dimensions of an image by sliding a window over the image and taking the maximum value in that window.

Assume:
- Window size is (pool_size, pool_size)
- Stride is equal to pool_size (non-overlapping windows)
- Single channel input

Output dimension: $(H // pool\_size) \times (W // pool\_size)$.`,
    constraints: [
      'Input image is a 2D array of shape (H, W)',
      'Return the downsampled 2D array',
      'H and W are guaranteed to be divisible by pool_size'
    ],
    examples: [
      { input: 'max_pool2d(image, pool_size=2)', output: 'downsampled_image' }
    ],
    starter_code: `import numpy as np\n\ndef max_pool2d(image, pool_size):\n    # TODO: implement non-overlapping max pooling\n    pass\n`,
    test_input: `import numpy as np\nimg = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])\nprint(max_pool2d(img, 2))`,
    acceptance: '72.3%',
    hints: [
      'Output shape is (H // pool_size, W // pool_size)',
      'Nested loops: i from 0 to out_H, j from 0 to out_W',
      'Extract patch: image[i*pool_size : (i+1)*pool_size, j*pool_size : (j+1)*pool_size]',
      'Take np.max(patch)'
    ]
  },
  {
    slug: 'dl-sgd-momentum',
    index: 40,
    title: 'SGD with Momentum',
    difficulty: 'Medium',
    topics: ['Optimization', 'Deep Learning'],
    description: `Implement a single step of Stochastic Gradient Descent with Momentum.

Momentum helps accelerate SGD in the relevant direction and dampens oscillations by keeping a running velocity.

Update rules:
1. $v_{new} = \beta v_{old} + (1 - \beta) \nabla L$ (Note: PyTorch usually implements this without the $(1-\beta)$ factor, but we will use the Exponential Moving Average version).
2. $\theta_{new} = \theta_{old} - \alpha v_{new}$

Where $\alpha$ is learning rate, $\beta$ is momentum coefficient, and $v$ is velocity.`,
    constraints: [
      'params, grads, and velocity are numpy arrays of the same shape',
      'Return a tuple: (new_params, new_velocity)'
    ],
    examples: [
      { input: 'sgd_momentum(w, dw, v, lr=0.1, beta=0.9)', output: '(w_new, v_new)' }
    ],
    starter_code: `import numpy as np\n\ndef sgd_momentum(params, grads, velocity, lr=0.1, beta=0.9):\n    # TODO: update velocity and parameters\n    pass\n`,
    test_input: `import numpy as np\nw = np.array([1.0, 2.0])\ndw = np.array([0.5, -0.2])\nv = np.array([0.1, -0.1])\nw_new, v_new = sgd_momentum(w, dw, v, 0.1, 0.9)\nprint(np.round(w_new, 4))`,
    acceptance: '66.1%',
    hints: [
      'new_velocity = beta * velocity + (1 - beta) * grads',
      'new_params = params - lr * new_velocity'
    ]
  },
  {
    slug: 'dl-adam-optimizer',
    index: 41,
    title: 'Adam Optimizer (One Step)',
    difficulty: 'Medium',
    topics: ['Optimization', 'Deep Learning'],
    description: `Implement a single step of the Adam optimizer.

Adam maintains both first-order (momentum) and second-order (RMSprop) moments.

Update rules for timestep $t$:
1. Update biased first moment: $m = \beta_1 m + (1 - \beta_1) g$
2. Update biased second raw moment: $v = \beta_2 v + (1 - \beta_2) g^2$
3. Compute bias-corrected first moment: $\hat{m} = \frac{m}{1 - \beta_1^t}$
4. Compute bias-corrected second raw moment: $\hat{v} = \frac{v}{1 - \beta_2^t}$
5. Update parameters: $\theta = \theta - \frac{\alpha}{\sqrt{\hat{v}} + \epsilon} \hat{m}$`,
    constraints: [
      'params, grads, m, v are arrays of the same shape',
      't is the current timestep (integer > 0)',
      'Return a tuple: (new_params, new_m, new_v)'
    ],
    examples: [
      { input: 'adam_step(w, dw, m, v, t, lr, b1, b2, eps)', output: '(w_new, m_new, v_new)' }
    ],
    starter_code: `import numpy as np\n\ndef adam_step(params, grads, m, v, t, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):\n    # TODO: perform one step of Adam optimization\n    pass\n`,
    test_input: `import numpy as np\nw = np.array([1.0, 2.0])\ndw = np.array([0.5, -0.2])\nm = np.zeros(2)\nv = np.zeros(2)\nw_new, _, _ = adam_step(w, dw, m, v, t=1)\nprint(np.round(w_new, 4))`,
    acceptance: '51.9%',
    hints: [
      'Ensure you use b1**t and b2**t for the bias correction',
      'Do not forget the epsilon in the denominator to prevent division by zero'
    ]
  },
  {
    slug: 'dl-embedding-lookup',
    index: 42,
    title: 'Embedding Lookup + Gradient',
    difficulty: 'Medium',
    topics: ['Deep Learning', 'NLP'],
    description: `Implement an embedding layer forward and backward pass.

In NLP, categorical tokens (like words) are mapped to dense vectors via an embedding matrix.
Forward pass:
For an array of token indices, return the corresponding rows of the embedding matrix.

Backward pass:
Given the gradient of the loss with respect to the output vectors ($dOut$), compute the gradient with respect to the embedding matrix ($dE$).`,
    constraints: [
      'indices is a 1D array of integers of length N',
      'embeddings is a 2D array of shape (vocab_size, hidden_dim)',
      'dOut is a 2D array of shape (N, hidden_dim)',
      'Return a tuple: (output, dE)'
    ],
    examples: [
      { input: 'embedding_layer(indices, embeddings, dOut)', output: '(output, dE)' }
    ],
    starter_code: `import numpy as np\n\ndef embedding_layer(indices, embeddings, dOut):\n    # TODO: Forward pass (lookup)\n    \n    # TODO: Backward pass (compute gradient w.r.t embeddings)\n    pass\n`,
    test_input: `import numpy as np\nemb = np.array([[0,0], [1,1], [2,2], [3,3]])\nidx = np.array([1, 3, 1])\ndOut = np.array([[0.1, 0.1], [0.3, 0.3], [0.5, 0.5]])\nout, dE = embedding_layer(idx, emb, dOut)\nprint("out:\\n", out)\nprint("dE:\\n", np.round(dE, 2))`,
    acceptance: '44.3%',
    hints: [
      'Forward is just slicing: output = embeddings[indices]',
      'Backward requires accumulating gradients. Initialize dE = np.zeros_like(embeddings)',
      'Use np.add.at(dE, indices, dOut) to handle repeated indices correctly'
    ]
  },
  {
    slug: 'dl-multi-head-attention',
    index: 43,
    title: 'Multi-Head Attention (MHA)',
    difficulty: 'Hard',
    topics: ['Deep Learning', 'Transformers', 'Paper Implementation'],
    description: `Implement Multi-Head Attention from "Attention Is All You Need".

Given inputs $X$ (shape $N \times d_{model}$), project them into Query, Key, and Value matrices.
Then, split them into $h$ heads, compute scaled dot-product attention for each head, concatenate the results, and project back to $d_{model}$.

Assume:
- $W_q, W_k, W_v \in \mathbb{R}^{d_{model} \times d_{model}}$
- $W_o \in \mathbb{R}^{d_{model} \times d_{model}}$
- $d_{head} = d_{model} / h$
- No masking

Return the final output matrix.`,
    constraints: [
      'Input X has shape (N, d_model)',
      'W_q, W_k, W_v, W_o have shape (d_model, d_model)',
      'h is the number of heads (integer)',
      'Return a 2D array of shape (N, d_model)'
    ],
    examples: [
      { input: 'mha(X, Wq, Wk, Wv, Wo, h=2)', output: 'attended_X' }
    ],
    starter_code: `import numpy as np\n\ndef mha(X, W_q, W_k, W_v, W_o, h):\n    # TODO: Implement Multi-Head Attention\n    pass\n`,
    test_input: `import numpy as np\nnp.random.seed(42)\nX = np.random.randn(3, 4)\nWq, Wk, Wv, Wo = np.random.randn(4,4), np.random.randn(4,4), np.random.randn(4,4), np.random.randn(4,4)\nprint(np.round(mha(X, Wq, Wk, Wv, Wo, 2), 4))`,
    acceptance: '22.1%',
    hints: [
      'Project X to Q, K, V',
      'Reshape to (N, h, d_head) and transpose to (h, N, d_head)',
      'Compute attention scores: Q @ K.transpose(0, 2, 1) / sqrt(d_head)',
      'Apply softmax along the last axis (-1)',
      'Multiply by V, transpose back, reshape to (N, d_model), and project with W_o'
    ]
  },
  {
    slug: 'dl-transformer-encoder-block',
    index: 44,
    title: 'Transformer Encoder Block',
    difficulty: 'Hard',
    topics: ['Deep Learning', 'Transformers', 'Paper Implementation'],
    description: `Implement a single Transformer Encoder Block.

The block consists of two sub-layers:
1. Multi-Head Attention (assume you have a function \`mha(X)\` available).
2. Feed-Forward Network (two linear transformations with a ReLU activation in between).

Each sub-layer is surrounded by a residual connection and followed by Layer Normalization (assume \`layer_norm(X)\` is available).

Forward pass:
$X_1 = \text{LayerNorm}(X + \text{MHA}(X))$
$X_2 = \text{LayerNorm}(X_1 + \text{FFN}(X_1))$

Wait, the original paper puts LayerNorm *after* the residual addition (Post-LN). Implement the Post-LN version.`,
    constraints: [
      'Input X has shape (N, d_model)',
      'W1 (d_model, d_ff), b1 (d_ff)',
      'W2 (d_ff, d_model), b2 (d_model)',
      'Use the provided mha and layer_norm helper functions',
      'Return the output matrix of shape (N, d_model)'
    ],
    examples: [
      { input: 'transformer_block(X, W1, b1, W2, b2, mha_fn, ln_fn)', output: 'encoder_output' }
    ],
    starter_code: `import numpy as np\n\ndef transformer_block(X, W1, b1, W2, b2, mha_fn, ln_fn):\n    # TODO: Implement Transformer Encoder Block (Post-LN)\n    pass\n`,
    test_input: `import numpy as np\nnp.random.seed(42)\nX = np.random.randn(2, 4)\nW1, b1 = np.random.randn(4, 8), np.random.randn(8)\nW2, b2 = np.random.randn(8, 4), np.random.randn(4)\n# Mock functions\nmha_fn = lambda x: x * 0.1\nln_fn = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-5)\nprint(np.round(transformer_block(X, W1, b1, W2, b2, mha_fn, ln_fn), 4))`,
    acceptance: '35.4%',
    hints: [
      'MHA output: out1 = mha_fn(X)',
      'Residual + LN 1: X1 = ln_fn(X + out1)',
      'FFN: ffn_out = np.maximum(0, X1 @ W1 + b1) @ W2 + b2',
      'Residual + LN 2: out2 = ln_fn(X1 + ffn_out)'
    ]
  },
  {
    slug: 'dl-resnet-block',
    index: 45,
    title: 'ResNet Residual Block',
    difficulty: 'Medium',
    topics: ['Deep Learning', 'Computer Vision', 'Paper Implementation'],
    description: `Implement a basic ResNet Residual Block.

The block applies:
1. Conv2D -> BatchNorm -> ReLU
2. Conv2D -> BatchNorm
3. Add the skip connection (original input $X$)
4. ReLU

Assume:
- The input $X$ and the output of the second BatchNorm have the exact same shape (no projection shortcut needed).
- Helper functions \`conv2d_fn\` and \`batch_norm_fn\` are provided.`,
    constraints: [
      'Input X is a feature map',
      'W1, W2 are conv weights',
      'Use conv2d_fn(input, weight) and batch_norm_fn(input)',
      'Return the output feature map'
    ],
    examples: [
      { input: 'resnet_block(X, W1, W2, conv2d_fn, bn_fn)', output: 'output_feature_map' }
    ],
    starter_code: `import numpy as np\n\ndef resnet_block(X, W1, W2, conv2d_fn, bn_fn):\n    # TODO: Implement Residual Block\n    pass\n`,
    test_input: `import numpy as np\nnp.random.seed(42)\nX = np.random.randn(3, 3)\nW1, W2 = np.random.randn(2, 2), np.random.randn(2, 2)\n# Mock functions to simulate same-shape convolutions\nconv2d_fn = lambda x, w: x * np.sum(w)\nbn_fn = lambda x: x * 0.5\nprint(np.round(resnet_block(X, W1, W2, conv2d_fn, bn_fn), 4))`,
    acceptance: '58.2%',
    hints: [
      'Compute out1 = bn_fn(conv2d_fn(X, W1))',
      'Apply ReLU: out1 = np.maximum(0, out1)',
      'Compute out2 = bn_fn(conv2d_fn(out1, W2))',
      'Add skip connection and apply ReLU: return np.maximum(0, out2 + X)'
    ]
  },
  {
    slug: 'dl-bert-mlm-loss',
    index: 46,
    title: 'BERT MLM Loss',
    difficulty: 'Medium',
    topics: ['Deep Learning', 'NLP', 'Paper Implementation'],
    description: `Implement the Masked Language Model (MLM) Loss used in BERT.

Given the model's unnormalized logit predictions for a sequence of tokens, and the true token IDs, compute the Cross-Entropy loss.
Crucially, in BERT, the loss is *only* computed over the masked tokens.

You are given a \`mask_indices\` array indicating which sequence positions were masked (1 for masked, 0 for unmasked).`,
    constraints: [
      'logits is a 2D array of shape (seq_len, vocab_size)',
      'targets is a 1D array of integers of shape (seq_len,)',
      'mask_indices is a 1D array of 0s and 1s of shape (seq_len,)',
      'Return the scalar average cross-entropy loss over ONLY the masked tokens',
      'If no tokens are masked, return 0.0'
    ],
    examples: [
      { input: 'bert_mlm_loss(logits, targets, mask)', output: 'scalar_loss' }
    ],
    starter_code: `import numpy as np\n\ndef bert_mlm_loss(logits, targets, mask_indices):\n    # TODO: Compute cross-entropy loss over masked tokens\n    pass\n`,
    test_input: `import numpy as np\nlogits = np.array([[2.0, 0.5, -1.0], [0.1, 3.0, 0.1], [-0.5, 0.0, 2.5]])\ntargets = np.array([0, 2, 2])\nmask = np.array([1, 0, 1])\nprint(np.round(bert_mlm_loss(logits, targets, mask), 4))`,
    acceptance: '48.9%',
    hints: [
      'Filter logits and targets using mask_indices (e.g., logits[mask_indices == 1])',
      'Compute softmax on the filtered logits',
      'Extract the probability of the target classes',
      'Compute -np.mean(np.log(probs))'
    ]
  },
  {
    slug: 'dl-vit-patch-embed',
    index: 47,
    title: 'ViT Patch Embeddings',
    difficulty: 'Medium',
    topics: ['Deep Learning', 'Transformers', 'Computer Vision'],
    description: `Implement the patch extraction and embedding step of the Vision Transformer (ViT).

An image $X \in \mathbb{R}^{H \times W \times C}$ is reshaped into a sequence of flattened 2D patches $X_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where $(P, P)$ is the patch size, and $N = H W / P^2$ is the number of patches.
Then, the flattened patches are linearly projected to a hidden dimension $D$ using a weight matrix $W$.`,
    constraints: [
      'image is a 3D array of shape (H, W, C)',
      'patch_size is an integer P',
      'W is a 2D array of shape (P*P*C, D)',
      'Return the projected sequence of shape (N, D)',
      'Assume H and W are divisible by P'
    ],
    examples: [
      { input: 'vit_patch_embed(image, patch_size=2, W)', output: 'sequence_of_vectors' }
    ],
    starter_code: `import numpy as np\n\ndef vit_patch_embed(image, patch_size, W):\n    # TODO: Extract patches, flatten them, and project using W\n    pass\n`,
    test_input: `import numpy as np\nimg = np.arange(16).reshape(4, 4, 1)\nW = np.ones((4, 2))\nprint(vit_patch_embed(img, 2, W))`,
    acceptance: '53.6%',
    hints: [
      'Find the number of patches along height and width: H // P, W // P',
      'You can use nested loops or reshape/transpose to extract patches',
      'With reshape: image.reshape(H//P, P, W//P, P, C).transpose(0, 2, 1, 3, 4).reshape(-1, P*P*C)',
      'Multiply the resulting (N, P*P*C) matrix by W'
    ]
  },
  {
    slug: 'dl-lora-forward',
    index: 48,
    title: 'LoRA Forward Pass',
    difficulty: 'Easy',
    topics: ['Deep Learning', 'LLMs', 'Paper Implementation'],
    description: `Implement the forward pass of Low-Rank Adaptation (LoRA).

Instead of fine-tuning a large pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA freezes $W_0$ and injects trainable rank decomposition matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$.

The forward pass becomes:
$h = X W_0^T + \frac{\alpha}{r} X A^T B^T$

Where $X$ is the input, and $\alpha$ is a scaling scalar. (Note: in standard PyTorch linear layers, $y = x W^T$).`,
    constraints: [
      'X has shape (N, k)',
      'W0 has shape (d, k)',
      'A has shape (r, k)',
      'B has shape (d, r)',
      'alpha and r are scalars',
      'Return the output matrix of shape (N, d)'
    ],
    examples: [
      { input: 'lora_forward(X, W0, A, B, alpha=16, r=4)', output: 'adapted_output' }
    ],
    starter_code: `import numpy as np\n\ndef lora_forward(X, W0, A, B, alpha, r):\n    # TODO: Implement LoRA forward pass\n    pass\n`,
    test_input: `import numpy as np\nX = np.ones((2, 4))\nW0 = np.ones((3, 4)) * 2\nA = np.ones((2, 4)) * 0.5\nB = np.ones((3, 2)) * 0.5\nprint(lora_forward(X, W0, A, B, 4, 2))`,
    acceptance: '89.2%',
    hints: [
      'Compute base output: out_base = X @ W0.T',
      'Compute lora output: lora_out = (X @ A.T) @ B.T',
      'Scale lora output by alpha / r',
      'Return out_base + scaled_lora_out'
    ]
  },
  {
    slug: 'dl-flash-attention-conceptual',
    index: 49,
    title: 'FlashAttention (Tiled Forward)',
    difficulty: 'Hard',
    topics: ['Deep Learning', 'Optimization', 'Paper Implementation'],
    description: `Implement a simplified, conceptual version of FlashAttention's tiled forward pass.

Standard attention computes $S = Q K^T$ (size $N \times N$), which uses too much memory.
FlashAttention computes the softmax block-by-block to avoid materializing $S$.

Algorithm (Simplified, Online Softmax):
For a given query vector $q$ and blocks of keys $K_j$ and values $V_j$:
Keep track of a running maximum $m$ and a running denominator $l$.
For each block $j$:
1. $s_j = q K_j^T$
2. $m_{new} = \max(m, \max(s_j))$
3. $p_j = \exp(s_j - m_{new})$
4. $l_{new} = l \cdot \exp(m - m_{new}) + \sum p_j$
5. $O_{new} = \frac{1}{l_{new}} \left( O \cdot l \cdot \exp(m - m_{new}) + p_j V_j \right)$
6. $m = m_{new}, l = l_{new}, O = O_{new}$

Implement this for a *single query vector* $q$ processing $K$ and $V$ in chunks.`,
    constraints: [
      'q has shape (1, d)',
      'K and V have shape (N, d)',
      'block_size is an integer (e.g., process K and V in chunks of this size)',
      'Return the output vector of shape (1, d)'
    ],
    examples: [
      { input: 'flash_attn_single_query(q, K, V, block_size=2)', output: 'attended_vector' }
    ],
    starter_code: `import numpy as np\n\ndef flash_attn_single_query(q, K, V, block_size):\n    N, d = K.shape\n    m = -np.inf\n    l = 0.0\n    O = np.zeros((1, d))\n    \n    # TODO: Process K and V in chunks of block_size\n    \n    return O\n`,
    test_input: `import numpy as np\nq = np.array([[1.0, 0.0]])\nK = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])\nV = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])\nprint(np.round(flash_attn_single_query(q, K, V, 2), 4))`,
    acceptance: '18.7%',
    hints: [
      'Iterate j from 0 to N with step block_size',
      'K_j = K[j : j + block_size], V_j = V[j : j + block_size]',
      's_j = q @ K_j.T',
      'Update m, l, and O according to the formulas in the description. Be very careful with the exponential scaling factors!'
    ]
  }
];

export function getProblemBySlug(slug: string): Problem | undefined {
  return PROBLEMS.find(p => p.slug === slug);
}

export function getProblemIndex(slug: string): number {
  return PROBLEMS.findIndex(p => p.slug === slug);
}
