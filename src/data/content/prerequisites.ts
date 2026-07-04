export type PrereqEdge = { from: string; to: string; domain: string; section: string };

export const PREREQ_EDGES: PrereqEdge[] = [
  {
    "from": "Set Theory",
    "to": "Logic",
    "domain": "MATHEMATICS",
    "section": "Foundations"
  },
  {
    "from": "Logic",
    "to": "Proof Techniques",
    "domain": "MATHEMATICS",
    "section": "Foundations"
  },
  {
    "from": "Proof Techniques",
    "to": "Mathematical Induction",
    "domain": "MATHEMATICS",
    "section": "Foundations"
  },
  {
    "from": "Set Theory",
    "to": "Functions",
    "domain": "MATHEMATICS",
    "section": "Foundations"
  },
  {
    "from": "Functions",
    "to": "Composition of Functions",
    "domain": "MATHEMATICS",
    "section": "Foundations"
  },
  {
    "from": "Functions",
    "to": "Inverse Functions",
    "domain": "MATHEMATICS",
    "section": "Foundations"
  },
  {
    "from": "Number Systems",
    "to": "Real Numbers",
    "domain": "MATHEMATICS",
    "section": "Foundations"
  },
  {
    "from": "Real Numbers",
    "to": "Complex Numbers",
    "domain": "MATHEMATICS",
    "section": "Foundations"
  },
  {
    "from": "Real Numbers",
    "to": "Rational Numbers",
    "domain": "MATHEMATICS",
    "section": "Foundations"
  },
  {
    "from": "Mathematical Induction",
    "to": "Recursive Definitions",
    "domain": "MATHEMATICS",
    "section": "Foundations"
  },
  {
    "from": "Real Numbers",
    "to": "Vectors",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Vectors",
    "to": "Vector Addition",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Vectors",
    "to": "Scalar Multiplication",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Vectors",
    "to": "Dot Product",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Dot Product",
    "to": "Cosine Similarity",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Cosine Similarity",
    "to": "Semantic Similarity",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Vectors",
    "to": "Cross Product",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Vectors",
    "to": "Vector Spaces",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Vector Spaces",
    "to": "Basis",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Basis",
    "to": "Span",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Basis",
    "to": "Linear Independence",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Linear Independence",
    "to": "Rank",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Rank",
    "to": "Null Space",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Rank",
    "to": "Column Space",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Vectors",
    "to": "Matrices",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Matrices",
    "to": "Matrix Addition",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Matrices",
    "to": "Matrix Multiplication",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Matrix Multiplication",
    "to": "Transpose",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Matrix Multiplication",
    "to": "Inverse Matrix",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Inverse Matrix",
    "to": "System of Linear Equations",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "System of Linear Equations",
    "to": "Gaussian Elimination",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Gaussian Elimination",
    "to": "LU Decomposition",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Matrices",
    "to": "Determinant",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Determinant",
    "to": "Cramer's Rule",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Matrices",
    "to": "Identity Matrix",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Identity Matrix",
    "to": "Orthogonal Matrices",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Orthogonal Matrices",
    "to": "Rotation Matrices",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Matrices",
    "to": "Trace",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Trace",
    "to": "Frobenius Norm",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Matrices",
    "to": "Eigenvalues",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Eigenvalues",
    "to": "Characteristic Polynomial",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Eigenvalues",
    "to": "Eigenvectors",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Eigenvectors",
    "to": "Eigendecomposition",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Eigendecomposition",
    "to": "Diagonalization",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Eigendecomposition",
    "to": "Power Method",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Eigendecomposition",
    "to": "SVD",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "SVD",
    "to": "Pseudoinverse",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "SVD",
    "to": "PCA",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "SVD",
    "to": "Matrix Factorization",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "SVD",
    "to": "Low-Rank Approximation",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "SVD",
    "to": "Truncated SVD",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Matrix Factorization",
    "to": "Recommender Systems",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Low-Rank Approximation",
    "to": "LoRA",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Matrices",
    "to": "Norms",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Norms",
    "to": "L1 Norm",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Norms",
    "to": "L2 Norm",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Norms",
    "to": "Frobenius Norm",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Norms",
    "to": "Spectral Norm",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Spectral Norm",
    "to": "GAN Training Stability",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Vector Spaces",
    "to": "Inner Product Spaces",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Inner Product Spaces",
    "to": "Orthogonality",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Orthogonality",
    "to": "Gram-Schmidt Process",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Gram-Schmidt Process",
    "to": "QR Decomposition",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Matrices",
    "to": "Positive Definite Matrices",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Positive Definite Matrices",
    "to": "Cholesky Decomposition",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Cholesky Decomposition",
    "to": "Gaussian Processes",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Inner Product Spaces",
    "to": "Hilbert Spaces",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Hilbert Spaces",
    "to": "Kernel Methods",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Kernel Methods",
    "to": "Kernel Trick",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Kernel Trick",
    "to": "SVM",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Kernel Trick",
    "to": "RBF Kernel",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Kernel Trick",
    "to": "Polynomial Kernel",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Kernel Trick",
    "to": "Gaussian Processes",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Positive Definite Matrices",
    "to": "Covariance Matrix",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Covariance Matrix",
    "to": "Multivariate Gaussian",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Multivariate Gaussian",
    "to": "Gaussian Processes",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Gaussian Processes",
    "to": "Bayesian Optimization",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Bayesian Optimization",
    "to": "Hyperparameter Search",
    "domain": "MATHEMATICS",
    "section": "Linear Algebra"
  },
  {
    "from": "Real Numbers",
    "to": "Limits",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Limits",
    "to": "Continuity",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Continuity",
    "to": "Derivatives",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Derivatives",
    "to": "Chain Rule",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Chain Rule",
    "to": "Backpropagation",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Chain Rule",
    "to": "Partial Derivatives",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Partial Derivatives",
    "to": "Gradient",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Gradient",
    "to": "Directional Derivative",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Gradient",
    "to": "Gradient Descent",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Gradient",
    "to": "Jacobian",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Jacobian",
    "to": "Backpropagation (Multivariable)",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Jacobian",
    "to": "Hessian",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Hessian",
    "to": "Newton's Method",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Hessian",
    "to": "Curvature",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Curvature",
    "to": "Second-Order Optimization",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Newton's Method",
    "to": "Quasi-Newton Methods",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Quasi-Newton Methods",
    "to": "L-BFGS",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Derivatives",
    "to": "Integration",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Integration",
    "to": "Fundamental Theorem of Calculus",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Integration",
    "to": "Multiple Integrals",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Multiple Integrals",
    "to": "Change of Variables",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Multiple Integrals",
    "to": "Normalizing Constants",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Derivatives",
    "to": "Taylor Series",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Taylor Series",
    "to": "Local Approximations",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Taylor Series",
    "to": "Softplus Approximation",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Derivatives",
    "to": "Implicit Differentiation",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Limits",
    "to": "L'Hopital's Rule",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Calculus",
    "to": "Vector Calculus",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Vector Calculus",
    "to": "Divergence",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Vector Calculus",
    "to": "Curl",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Vector Calculus",
    "to": "Gradient (Vector Field)",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Calculus",
    "to": "Differential Equations",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Differential Equations",
    "to": "ODEs",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "ODEs",
    "to": "Neural ODEs",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Neural ODEs",
    "to": "Continuous Normalizing Flows",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Continuous Normalizing Flows",
    "to": "Flow Matching",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Flow Matching",
    "to": "Rectified Flows",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Rectified Flows",
    "to": "Flux Diffusion",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Differential Equations",
    "to": "PDEs",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "PDEs",
    "to": "Physics-Informed Neural Networks",
    "domain": "MATHEMATICS",
    "section": "Calculus"
  },
  {
    "from": "Set Theory",
    "to": "Probability Theory",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Probability Theory",
    "to": "Sample Space",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Probability Theory",
    "to": "Events",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Events",
    "to": "Conditional Probability",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Conditional Probability",
    "to": "Bayes' Theorem",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Bayes' Theorem",
    "to": "Bayesian Inference",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Bayesian Inference",
    "to": "Prior Distributions",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Bayesian Inference",
    "to": "Posterior Distributions",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Bayesian Inference",
    "to": "MAP Estimation",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Bayesian Inference",
    "to": "Bayesian Neural Networks",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Probability Theory",
    "to": "Random Variables",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Random Variables",
    "to": "Discrete Distributions",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Discrete Distributions",
    "to": "Bernoulli Distribution",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Discrete Distributions",
    "to": "Binomial Distribution",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Discrete Distributions",
    "to": "Categorical Distribution",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Categorical Distribution",
    "to": "Softmax",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Discrete Distributions",
    "to": "Poisson Distribution",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Random Variables",
    "to": "Continuous Distributions",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Continuous Distributions",
    "to": "Uniform Distribution",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Continuous Distributions",
    "to": "Gaussian Distribution",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Gaussian Distribution",
    "to": "Standard Normal",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Gaussian Distribution",
    "to": "Multivariate Gaussian",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Gaussian Distribution",
    "to": "Maximum Likelihood Estimation",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Maximum Likelihood Estimation",
    "to": "Cross-Entropy Loss",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Maximum Likelihood Estimation",
    "to": "MAP Estimation",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Continuous Distributions",
    "to": "Exponential Distribution",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Continuous Distributions",
    "to": "Beta Distribution",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Beta Distribution",
    "to": "Dirichlet Distribution",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Dirichlet Distribution",
    "to": "Topic Models",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Dirichlet Distribution",
    "to": "LDA",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Random Variables",
    "to": "Expected Value",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Expected Value",
    "to": "Variance",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Variance",
    "to": "Standard Deviation",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Variance",
    "to": "Covariance",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Covariance",
    "to": "Correlation",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Covariance",
    "to": "Covariance Matrix",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Probability Theory",
    "to": "Law of Large Numbers",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Law of Large Numbers",
    "to": "Central Limit Theorem",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Central Limit Theorem",
    "to": "Sampling Distributions",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Sampling Distributions",
    "to": "Confidence Intervals",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Sampling Distributions",
    "to": "Hypothesis Testing",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Hypothesis Testing",
    "to": "p-value",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Hypothesis Testing",
    "to": "Type I Error",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Hypothesis Testing",
    "to": "Type II Error",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Hypothesis Testing",
    "to": "Statistical Power",
    "domain": "MATHEMATICS",
    "section": "Probability & Statistics"
  },
  {
    "from": "Probability Theory",
    "to": "Information Theory",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "Information Theory",
    "to": "Entropy",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "Entropy",
    "to": "Shannon Entropy",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "Entropy",
    "to": "Cross-Entropy",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "Cross-Entropy",
    "to": "Cross-Entropy Loss",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "Cross-Entropy Loss",
    "to": "Classification Training",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "Entropy",
    "to": "KL Divergence",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "KL Divergence",
    "to": "Reverse KL",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "KL Divergence",
    "to": "Forward KL",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "KL Divergence",
    "to": "RLHF KL Penalty",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "KL Divergence",
    "to": "VAE ELBO",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "KL Divergence",
    "to": "Information Gain",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "Information Gain",
    "to": "Mutual Information",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "Mutual Information",
    "to": "Contrastive Learning",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "Mutual Information",
    "to": "Feature Selection (Info-Theoretic)",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "Information Theory",
    "to": "Bits",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "Bits",
    "to": "Data Compression",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "Data Compression",
    "to": "Arithmetic Coding",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "Entropy",
    "to": "Perplexity",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "Perplexity",
    "to": "Language Model Evaluation",
    "domain": "MATHEMATICS",
    "section": "Information Theory"
  },
  {
    "from": "Gradient",
    "to": "Gradient Descent",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Gradient Descent",
    "to": "Stochastic Gradient Descent",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Stochastic Gradient Descent",
    "to": "Mini-batch Gradient Descent",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Mini-batch Gradient Descent",
    "to": "Batch Size",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Gradient Descent",
    "to": "Learning Rate",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Learning Rate",
    "to": "Learning Rate Schedules",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Learning Rate Schedules",
    "to": "Warmup",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Learning Rate Schedules",
    "to": "Cosine Annealing",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Learning Rate Schedules",
    "to": "Linear Decay",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Learning Rate Schedules",
    "to": "Step Decay",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Gradient Descent",
    "to": "Momentum",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Momentum",
    "to": "Nesterov Momentum",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Gradient Descent",
    "to": "Adaptive Learning Rates",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Adaptive Learning Rates",
    "to": "AdaGrad",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "AdaGrad",
    "to": "RMSProp",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "RMSProp",
    "to": "Adam",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Adam",
    "to": "AdamW",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "AdamW",
    "to": "Weight Decay",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "AdamW",
    "to": "Decoupled Regularization",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Adam",
    "to": "AMSGrad",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Adam",
    "to": "Nadam",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "AdamW",
    "to": "Lion Optimizer",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Convexity",
    "to": "Convex Optimization",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Convex Optimization",
    "to": "Global Minimum",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Convex Optimization",
    "to": "Lagrangian",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Lagrangian",
    "to": "KKT Conditions",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "KKT Conditions",
    "to": "SVM",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Non-Convex Optimization",
    "to": "Local Minima",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Local Minima",
    "to": "Saddle Points",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Saddle Points",
    "to": "Escaping Saddle Points",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Gradient Descent",
    "to": "Loss Landscape",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Loss Landscape",
    "to": "Sharp Minima",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Sharp Minima",
    "to": "Generalization",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Sharp Minima",
    "to": "Sharpness-Aware Minimization",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Gradient Descent",
    "to": "Gradient Clipping",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Gradient Clipping",
    "to": "Stable RNN Training",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Gradient Clipping",
    "to": "LLM Training Stability",
    "domain": "MATHEMATICS",
    "section": "Optimization"
  },
  {
    "from": "Statistics",
    "to": "Machine Learning",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Optimization",
    "to": "Machine Learning",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Machine Learning",
    "to": "Supervised Learning",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Machine Learning",
    "to": "Unsupervised Learning",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Machine Learning",
    "to": "Semi-supervised Learning",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Machine Learning",
    "to": "Self-supervised Learning",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Supervised Learning",
    "to": "Classification",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Supervised Learning",
    "to": "Regression",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Machine Learning",
    "to": "Features",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Features",
    "to": "Feature Engineering",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Feature Engineering",
    "to": "Feature Selection",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Feature Engineering",
    "to": "Feature Scaling",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Feature Scaling",
    "to": "Normalization",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Feature Scaling",
    "to": "Standardization",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Feature Selection",
    "to": "Mutual Information (Feature)",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Feature Selection",
    "to": "Recursive Feature Elimination",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "PCA",
    "to": "Dimensionality Reduction",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Dimensionality Reduction",
    "to": "t-SNE",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Dimensionality Reduction",
    "to": "UMAP",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "t-SNE",
    "to": "Embedding Visualization",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "UMAP",
    "to": "Embedding Visualization",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "UMAP",
    "to": "Manifold Learning",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Machine Learning",
    "to": "Generalization",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Generalization",
    "to": "Bias-Variance Tradeoff",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Bias-Variance Tradeoff",
    "to": "Underfitting",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Bias-Variance Tradeoff",
    "to": "Overfitting",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Overfitting",
    "to": "Regularization",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Regularization",
    "to": "L1 Regularization",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Regularization",
    "to": "L2 Regularization",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "L1 Regularization",
    "to": "Lasso Regression",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "L2 Regularization",
    "to": "Ridge Regression",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Ridge Regression",
    "to": "ElasticNet",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Regularization",
    "to": "Dropout",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Regularization",
    "to": "Early Stopping",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Regularization",
    "to": "Data Augmentation",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Machine Learning",
    "to": "Model Evaluation",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Model Evaluation",
    "to": "Train-Test Split",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Train-Test Split",
    "to": "Cross-Validation",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Cross-Validation",
    "to": "K-Fold",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Cross-Validation",
    "to": "Stratified K-Fold",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Model Evaluation",
    "to": "Accuracy",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Model Evaluation",
    "to": "Precision",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Model Evaluation",
    "to": "Recall",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Precision",
    "to": "F1 Score",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Recall",
    "to": "F1 Score",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Model Evaluation",
    "to": "ROC Curve",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "ROC Curve",
    "to": "AUC",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Model Evaluation",
    "to": "Confusion Matrix",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Model Evaluation",
    "to": "MSE",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "MSE",
    "to": "RMSE",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "MSE",
    "to": "R-Squared",
    "domain": "MACHINE LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Linear Algebra",
    "to": "Linear Regression",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Linear Regression",
    "to": "Ordinary Least Squares",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Linear Regression",
    "to": "Gradient Descent Regression",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Linear Regression",
    "to": "Ridge Regression",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Linear Regression",
    "to": "Lasso Regression",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Logistic Regression",
    "to": "Binary Classification",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Logistic Regression",
    "to": "Sigmoid Activation",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Sigmoid Activation",
    "to": "Neural Networks",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Classification",
    "to": "Decision Trees",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Decision Trees",
    "to": "CART",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Decision Trees",
    "to": "Gini Impurity",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Decision Trees",
    "to": "Information Gain (Tree)",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "CART",
    "to": "Random Forest",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Random Forest",
    "to": "Bagging",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Bagging",
    "to": "Bootstrap Sampling",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Bootstrap Sampling",
    "to": "Bootstrap Aggregation",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Bagging",
    "to": "Ensemble Methods",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Ensemble Methods",
    "to": "Boosting",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Boosting",
    "to": "AdaBoost",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "AdaBoost",
    "to": "Gradient Boosting",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Gradient Boosting",
    "to": "XGBoost",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "XGBoost",
    "to": "LightGBM",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "LightGBM",
    "to": "CatBoost",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Ensemble Methods",
    "to": "Stacking",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Stacking",
    "to": "Meta-Learner",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "KKT Conditions",
    "to": "SVM",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "SVM",
    "to": "Support Vectors",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "SVM",
    "to": "Margin Maximization",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "SVM",
    "to": "Soft-Margin SVM",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "SVM",
    "to": "Multi-class SVM",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Kernel Trick",
    "to": "Kernel SVM",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Classification",
    "to": "K-Nearest Neighbors",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "K-Nearest Neighbors",
    "to": "Distance Metrics",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Distance Metrics",
    "to": "Euclidean Distance",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Distance Metrics",
    "to": "Manhattan Distance",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Distance Metrics",
    "to": "Cosine Distance",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Classification",
    "to": "Naive Bayes",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Bayes' Theorem",
    "to": "Naive Bayes",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Naive Bayes",
    "to": "Text Classification",
    "domain": "MACHINE LEARNING",
    "section": "Supervised Models"
  },
  {
    "from": "Unsupervised Learning",
    "to": "Clustering",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "Clustering",
    "to": "K-Means",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "K-Means",
    "to": "K-Means++",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "K-Means",
    "to": "Mini-batch K-Means",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "Clustering",
    "to": "DBSCAN",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "DBSCAN",
    "to": "Density-Based Clustering",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "Clustering",
    "to": "Hierarchical Clustering",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "Hierarchical Clustering",
    "to": "Agglomerative Clustering",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "Hierarchical Clustering",
    "to": "Dendrogram",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "Dimensionality Reduction",
    "to": "Autoencoders",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "Autoencoders",
    "to": "Encoder",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "Autoencoders",
    "to": "Decoder",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "Autoencoders",
    "to": "Latent Space",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "Latent Space",
    "to": "Representation Learning",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "Representation Learning",
    "to": "Self-supervised Learning",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "Autoencoders",
    "to": "Denoising Autoencoders",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "Autoencoders",
    "to": "Sparse Autoencoders",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "Autoencoders",
    "to": "Variational Autoencoders",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "Variational Autoencoders",
    "to": "VAE ELBO",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "VAE ELBO",
    "to": "Latent Variable Models",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "Latent Variable Models",
    "to": "Generative Modeling",
    "domain": "MACHINE LEARNING",
    "section": "Unsupervised Learning"
  },
  {
    "from": "Logistic Regression",
    "to": "Perceptron",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Perceptron",
    "to": "Multi-Layer Perceptron",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Multi-Layer Perceptron",
    "to": "Hidden Layers",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Hidden Layers",
    "to": "Universal Approximation Theorem",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Multi-Layer Perceptron",
    "to": "Feedforward Neural Network",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Feedforward Neural Network",
    "to": "Backpropagation",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Backpropagation",
    "to": "Chain Rule",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Backpropagation",
    "to": "Computation Graph",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Computation Graph",
    "to": "Automatic Differentiation",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Automatic Differentiation",
    "to": "PyTorch Autograd",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Automatic Differentiation",
    "to": "JAX",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Automatic Differentiation",
    "to": "TensorFlow",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Backpropagation",
    "to": "Vanishing Gradient Problem",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Vanishing Gradient Problem",
    "to": "Residual Connections",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Vanishing Gradient Problem",
    "to": "Batch Normalization",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Vanishing Gradient Problem",
    "to": "Gradient Clipping",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Vanishing Gradient Problem",
    "to": "LSTM (Motivation)",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Feedforward Neural Network",
    "to": "Activation Functions",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Activation Functions",
    "to": "Sigmoid",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Activation Functions",
    "to": "Tanh",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Activation Functions",
    "to": "ReLU",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "ReLU",
    "to": "Dying ReLU Problem",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "ReLU",
    "to": "Leaky ReLU",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "ReLU",
    "to": "ELU",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "ReLU",
    "to": "PReLU",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "ReLU",
    "to": "GELU",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "GELU",
    "to": "Transformer Activations",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "GELU",
    "to": "SwiGLU",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "SwiGLU",
    "to": "LLaMA FFN",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Activation Functions",
    "to": "Softmax",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Softmax",
    "to": "Attention Weights",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Feedforward Neural Network",
    "to": "Weight Initialization",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Weight Initialization",
    "to": "Xavier Initialization",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Weight Initialization",
    "to": "He Initialization",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "He Initialization",
    "to": "ReLU Networks",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Feedforward Neural Network",
    "to": "Loss Functions",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Loss Functions",
    "to": "Cross-Entropy Loss",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Loss Functions",
    "to": "MSE Loss",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Loss Functions",
    "to": "MAE Loss",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Loss Functions",
    "to": "Huber Loss",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Loss Functions",
    "to": "Contrastive Loss",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Contrastive Loss",
    "to": "Triplet Loss",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Triplet Loss",
    "to": "Metric Learning",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Metric Learning",
    "to": "Embedding Spaces",
    "domain": "DEEP LEARNING",
    "section": "Neural Networks"
  },
  {
    "from": "Batch Normalization",
    "to": "Covariate Shift",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "Batch Normalization",
    "to": "Training Stability",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "Batch Normalization",
    "to": "Layer Normalization",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "Layer Normalization",
    "to": "Transformer Pre-Norm",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "Layer Normalization",
    "to": "Transformer Post-Norm",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "Layer Normalization",
    "to": "RMS Norm",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "RMS Norm",
    "to": "LLaMA Architecture",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "Batch Normalization",
    "to": "Instance Normalization",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "Instance Normalization",
    "to": "Style Transfer",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "Instance Normalization",
    "to": "Group Normalization",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "Group Normalization",
    "to": "Vision Tasks (Small Batch)",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "Dropout",
    "to": "Test-Time Averaging",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "Dropout",
    "to": "MC Dropout",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "MC Dropout",
    "to": "Uncertainty Estimation",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "Weight Decay",
    "to": "L2 Regularization",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "Regularization",
    "to": "DropConnect",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "Regularization",
    "to": "Mixup",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "Mixup",
    "to": "CutMix",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "CutMix",
    "to": "Strong Augmentation",
    "domain": "DEEP LEARNING",
    "section": "Normalization & Regularization"
  },
  {
    "from": "Feedforward Neural Network",
    "to": "CNNs",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Convolution Operation",
    "to": "Feature Maps",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Feature Maps",
    "to": "Pooling",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Pooling",
    "to": "Max Pooling",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Pooling",
    "to": "Average Pooling",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Pooling",
    "to": "Global Average Pooling",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Global Average Pooling",
    "to": "GoogLeNet",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "CNNs",
    "to": "Receptive Field",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Receptive Field",
    "to": "Dilated Convolutions",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Dilated Convolutions",
    "to": "DeepLab",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "CNNs",
    "to": "Padding",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "CNNs",
    "to": "Stride",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Stride",
    "to": "Downsampling",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Downsampling",
    "to": "Feature Hierarchy",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Feature Hierarchy",
    "to": "FPN",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "FPN",
    "to": "Multi-Scale Detection",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "CNNs",
    "to": "Depthwise Separable Convolutions",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Depthwise Separable Convolutions",
    "to": "MobileNet",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "MobileNet",
    "to": "Mobile Inference",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "CNNs",
    "to": "Transposed Convolutions",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Transposed Convolutions",
    "to": "Upsampling",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Upsampling",
    "to": "Decoder Networks",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Decoder Networks",
    "to": "U-Net",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Decoder Networks",
    "to": "Segmentation",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Convolution Operation",
    "to": "1x1 Convolution",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "1x1 Convolution",
    "to": "Channel Mixing",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Channel Mixing",
    "to": "Bottleneck Blocks",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Bottleneck Blocks",
    "to": "ResNet Bottleneck",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "ResNet Bottleneck",
    "to": "ResNet-50",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "ResNet",
    "to": "Residual Connections",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Residual Connections",
    "to": "ResNeXt",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Residual Connections",
    "to": "DenseNet",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "DenseNet",
    "to": "Dense Connectivity",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "CNNs",
    "to": "Batch Norm + Conv + ReLU",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Batch Norm + Conv + ReLU",
    "to": "Standard CNN Block",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Standard CNN Block",
    "to": "VGGNet",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Standard CNN Block",
    "to": "AlexNet",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "VGGNet",
    "to": "Transfer Learning",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Transfer Learning",
    "to": "Fine-tuning",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Fine-tuning",
    "to": "Domain Adaptation",
    "domain": "DEEP LEARNING",
    "section": "Convolutional Networks"
  },
  {
    "from": "Sequences",
    "to": "RNN",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "RNN",
    "to": "Hidden State",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "Hidden State",
    "to": "Sequential Memory",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "RNN",
    "to": "BPTT (Backprop Through Time)",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "BPTT",
    "to": "Vanishing Gradient (Sequences)",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "Vanishing Gradient (Sequences)",
    "to": "LSTM",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "LSTM",
    "to": "Cell State",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "LSTM",
    "to": "Forget Gate",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "LSTM",
    "to": "Input Gate",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "LSTM",
    "to": "Output Gate",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "LSTM",
    "to": "Bidirectional LSTM",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "Bidirectional LSTM",
    "to": "ELMo",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "LSTM",
    "to": "Stacked LSTM",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "LSTM",
    "to": "Seq2Seq",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "RNN",
    "to": "GRU",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "GRU",
    "to": "Simplified Gating",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "GRU",
    "to": "Efficient Sequence Modeling",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "Seq2Seq",
    "to": "Encoder-Decoder",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "Encoder-Decoder",
    "to": "Context Vector",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "Context Vector",
    "to": "Bottleneck Problem",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "Bottleneck Problem",
    "to": "Attention Mechanism",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "LSTM",
    "to": "Time Series Forecasting",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "LSTM",
    "to": "Speech Recognition",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "LSTM",
    "to": "Text Generation",
    "domain": "DEEP LEARNING",
    "section": "Recurrent Networks"
  },
  {
    "from": "Attention Mechanism",
    "to": "Bahdanau Attention",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Bahdanau Attention",
    "to": "Alignment Model",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Bahdanau Attention",
    "to": "Self-Attention",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Self-Attention",
    "to": "Query Key Value",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Query Key Value",
    "to": "Scaled Dot-Product Attention",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Scaled Dot-Product Attention",
    "to": "Attention Weights",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Attention Weights",
    "to": "Softmax",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Scaled Dot-Product Attention",
    "to": "Multi-Head Attention",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Multi-Head Attention",
    "to": "Parallel Attention Heads",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Multi-Head Attention",
    "to": "Concatenation of Heads",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Multi-Head Attention",
    "to": "Transformer",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Transformer",
    "to": "Encoder Stack",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Transformer",
    "to": "Decoder Stack",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Transformer",
    "to": "Positional Encoding",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Positional Encoding",
    "to": "Sinusoidal Encoding",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Sinusoidal Encoding",
    "to": "Absolute Positional Encoding",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Positional Encoding",
    "to": "Learned Positional Encoding",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Positional Encoding",
    "to": "Relative Positional Encoding",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Relative Positional Encoding",
    "to": "RoPE",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "RoPE",
    "to": "Rotary Embeddings",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "RoPE",
    "to": "LLaMA",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "RoPE Extension",
    "to": "YaRN",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Relative Positional Encoding",
    "to": "ALiBi",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Transformer",
    "to": "Feed-Forward Sublayer",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Feed-Forward Sublayer",
    "to": "Position-wise FFN",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Position-wise FFN",
    "to": "Two Linear Layers",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Two Linear Layers",
    "to": "Bottleneck Expansion",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Transformer",
    "to": "Residual Connections (Transformer)",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Transformer",
    "to": "Layer Normalization (Transformer)",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Layer Normalization (Transformer)",
    "to": "Pre-Norm",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Pre-Norm",
    "to": "GPT Style",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Pre-Norm",
    "to": "LLaMA Style",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Layer Normalization (Transformer)",
    "to": "Post-Norm",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Post-Norm",
    "to": "Original Transformer Style",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Transformer",
    "to": "BERT",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Transformer",
    "to": "GPT",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Transformer",
    "to": "T5",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Transformer",
    "to": "BART",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Multi-Head Attention",
    "to": "Grouped Query Attention",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Grouped Query Attention",
    "to": "LLaMA 2",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Grouped Query Attention",
    "to": "Mistral 7B",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Multi-Head Attention",
    "to": "Multi-Query Attention",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Multi-Query Attention",
    "to": "Fast Inference",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Fast Inference",
    "to": "KV Cache",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "KV Cache",
    "to": "Efficient LLM Inference",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Multi-Head Attention",
    "to": "Sparse Attention",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Sparse Attention",
    "to": "BigBird",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Sparse Attention",
    "to": "Longformer",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Sparse Attention",
    "to": "Sliding Window Attention",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Sliding Window Attention",
    "to": "Mistral 7B (Window)",
    "domain": "DEEP LEARNING",
    "section": "Attention & Transformers"
  },
  {
    "from": "Variational Autoencoders",
    "to": "VAE ELBO",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "VAE ELBO",
    "to": "Reparameterization Trick",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Reparameterization Trick",
    "to": "Differentiable Sampling",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Latent Space",
    "to": "Generative Models",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Generative Models",
    "to": "GANs",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "GANs",
    "to": "Generator Network",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "GANs",
    "to": "Discriminator Network",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "GANs",
    "to": "Adversarial Training",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "GANs",
    "to": "Mode Collapse",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Mode Collapse",
    "to": "Wasserstein GAN",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Wasserstein GAN",
    "to": "Gradient Penalty",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Gradient Penalty",
    "to": "WGAN-GP",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Adversarial Training",
    "to": "DCGAN",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "DCGAN",
    "to": "Progressive Training",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Progressive Training",
    "to": "ProGAN",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "ProGAN",
    "to": "StyleGAN",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "StyleGAN",
    "to": "Mapping Network",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "StyleGAN",
    "to": "AdaIN",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "StyleGAN",
    "to": "StyleGAN2",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Generative Models",
    "to": "Normalizing Flows",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Normalizing Flows",
    "to": "Invertible Networks",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Invertible Networks",
    "to": "Glow",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Score Matching",
    "to": "Diffusion Models",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Diffusion Models",
    "to": "Forward Diffusion",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Forward Diffusion",
    "to": "Noise Schedule",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Noise Schedule",
    "to": "Beta Schedule",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Diffusion Models",
    "to": "Reverse Diffusion",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Reverse Diffusion",
    "to": "Denoising",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Denoising",
    "to": "DDPM",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "DDPM",
    "to": "U-Net Backbone (Diffusion)",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "DDPM",
    "to": "DDIM",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "DDIM",
    "to": "Deterministic Sampling",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "DDIM",
    "to": "Fewer Steps",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "DDIM",
    "to": "Latent Diffusion",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Latent Diffusion",
    "to": "VQ-VAE",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Latent Diffusion",
    "to": "Stable Diffusion",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Stable Diffusion",
    "to": "CLIP Conditioning",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Stable Diffusion",
    "to": "ControlNet",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "DDPM",
    "to": "Classifier-Free Guidance",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Classifier-Free Guidance",
    "to": "Conditional Generation",
    "domain": "DEEP LEARNING",
    "section": "Generative Models"
  },
  {
    "from": "Text",
    "to": "Tokenization",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "Tokenization",
    "to": "Word Tokenization",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "Tokenization",
    "to": "Character Tokenization",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "Tokenization",
    "to": "Subword Tokenization",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "Subword Tokenization",
    "to": "Byte Pair Encoding",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "Byte Pair Encoding",
    "to": "GPT Tokenizer",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "Byte Pair Encoding",
    "to": "WordPiece",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "WordPiece",
    "to": "BERT Tokenizer",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "WordPiece",
    "to": "SentencePiece",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "SentencePiece",
    "to": "LLaMA Tokenizer",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "SentencePiece",
    "to": "Multilingual Tokenizer",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "Text",
    "to": "Vocabulary",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "Vocabulary",
    "to": "One-Hot Encoding",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "One-Hot Encoding",
    "to": "Bag of Words",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "Bag of Words",
    "to": "TF-IDF",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "TF-IDF",
    "to": "Information Retrieval",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "TF-IDF",
    "to": "BM25",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "BM25",
    "to": "Sparse Retrieval",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "Word2Vec",
    "to": "Word Embeddings",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "Word Embeddings",
    "to": "GloVe",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "GloVe",
    "to": "Pre-trained Embeddings",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "GloVe",
    "to": "FastText",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "FastText",
    "to": "Subword Embeddings",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "FastText",
    "to": "OOV Handling",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "Word Embeddings",
    "to": "Embedding Layer",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "Embedding Layer",
    "to": "Contextual Embeddings",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "Contextual Embeddings",
    "to": "ELMo",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "ELMo",
    "to": "Deep Contextualization",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "Deep Contextualization",
    "to": "BERT",
    "domain": "NLP",
    "section": "Text Representations"
  },
  {
    "from": "Probability Theory",
    "to": "Language Models",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Language Models",
    "to": "N-gram Language Models",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "N-gram Language Models",
    "to": "Smoothing",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Smoothing",
    "to": "Kneser-Ney",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Language Models",
    "to": "Perplexity",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Perplexity",
    "to": "Model Comparison",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Language Models",
    "to": "Neural Language Models",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Neural Language Models",
    "to": "RNN Language Models",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "RNN Language Models",
    "to": "LSTM Language Models",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "LSTM Language Models",
    "to": "Sequential LM",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Sequential LM",
    "to": "Autoregressive Models",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Autoregressive Models",
    "to": "GPT",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "GPT",
    "to": "Causal Language Modeling",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Causal Language Modeling",
    "to": "Next Token Prediction",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Next Token Prediction",
    "to": "Language Generation",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Language Generation",
    "to": "Greedy Decoding",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Language Generation",
    "to": "Beam Search",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Language Generation",
    "to": "Sampling",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Sampling",
    "to": "Temperature Sampling",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Temperature Sampling",
    "to": "Top-k Sampling",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Top-k Sampling",
    "to": "Top-p (Nucleus) Sampling",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Top-p (Nucleus) Sampling",
    "to": "Diverse Generation",
    "domain": "NLP",
    "section": "Language Models"
  },
  {
    "from": "Transformers",
    "to": "BERT",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "BERT",
    "to": "Masked Language Modeling",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "BERT",
    "to": "Next Sentence Prediction",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "BERT",
    "to": "Fine-tuning (NLP)",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "Fine-tuning (NLP)",
    "to": "BERT for Classification",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "Fine-tuning (NLP)",
    "to": "BERT for NER",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "Fine-tuning (NLP)",
    "to": "BERT for QA",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "BERT",
    "to": "RoBERTa",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "RoBERTa",
    "to": "Improved Pretraining",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "BERT",
    "to": "ALBERT",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "ALBERT",
    "to": "Cross-layer Parameter Sharing",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "BERT",
    "to": "DistilBERT",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "DistilBERT",
    "to": "Knowledge Distillation",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "BERT",
    "to": "ELECTRA",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "ELECTRA",
    "to": "Replaced Token Detection",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "ELECTRA",
    "to": "Sample Efficiency",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "BERT",
    "to": "DeBERTa",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "DeBERTa",
    "to": "Disentangled Attention",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "Transformers",
    "to": "T5",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "T5",
    "to": "Text-to-Text Framework",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "Text-to-Text Framework",
    "to": "Unified NLP",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "T5",
    "to": "Flan-T5",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "Flan-T5",
    "to": "Instruction Tuning",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "T5",
    "to": "mT5",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "mT5",
    "to": "Multilingual",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "Transformers",
    "to": "BART",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "BART",
    "to": "Denoising Autoencoder",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "BART",
    "to": "Summarization",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "Transformers",
    "to": "GPT-2",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "GPT-2",
    "to": "Zero-shot Generation",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "Zero-shot Generation",
    "to": "Prompt-based Learning",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "BERT",
    "to": "Sentence-BERT",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "Sentence-BERT",
    "to": "Sentence Embeddings",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "Sentence Embeddings",
    "to": "Semantic Textual Similarity",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "Semantic Textual Similarity",
    "to": "Semantic Search",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "Semantic Search",
    "to": "Dense Retrieval",
    "domain": "NLP",
    "section": "Transformer-based NLP"
  },
  {
    "from": "NLP",
    "to": "Text Classification",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Text Classification",
    "to": "Sentiment Analysis",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Sentiment Analysis",
    "to": "Aspect-based Sentiment",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Text Classification",
    "to": "Topic Classification",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Text Classification",
    "to": "Intent Detection",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "NLP",
    "to": "Named Entity Recognition",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Named Entity Recognition",
    "to": "Information Extraction",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Information Extraction",
    "to": "Relation Extraction",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Relation Extraction",
    "to": "Knowledge Graph Population",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Information Extraction",
    "to": "Event Extraction",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "NLP",
    "to": "Part-of-Speech Tagging",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Part-of-Speech Tagging",
    "to": "Parsing",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Parsing",
    "to": "Dependency Parsing",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Parsing",
    "to": "Constituency Parsing",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "NLP",
    "to": "Machine Translation",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Machine Translation",
    "to": "Neural Machine Translation",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Neural Machine Translation",
    "to": "Seq2Seq",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Seq2Seq",
    "to": "Transformer-based NMT",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "NLP",
    "to": "Text Summarization",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Text Summarization",
    "to": "Extractive Summarization",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Text Summarization",
    "to": "Abstractive Summarization",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Abstractive Summarization",
    "to": "BART",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Abstractive Summarization",
    "to": "T5",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "NLP",
    "to": "Question Answering",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Question Answering",
    "to": "Extractive QA",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Extractive QA",
    "to": "SQuAD",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Extractive QA",
    "to": "BERT-QA",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Question Answering",
    "to": "Generative QA",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Generative QA",
    "to": "RAG",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "NLP",
    "to": "Coreference Resolution",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Coreference Resolution",
    "to": "SpanBERT",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "NLP",
    "to": "Natural Language Inference",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Natural Language Inference",
    "to": "MNLI",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "Natural Language Inference",
    "to": "RTE",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "NLP",
    "to": "Semantic Role Labeling",
    "domain": "NLP",
    "section": "Tasks"
  },
  {
    "from": "GPT-2",
    "to": "GPT-3",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "GPT-3",
    "to": "Large Language Models",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Large Language Models",
    "to": "Decoder-Only Architecture",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Decoder-Only Architecture",
    "to": "Causal Masking",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Causal Masking",
    "to": "Autoregressive Pretraining",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Autoregressive Pretraining",
    "to": "GPT-3",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "GPT-3",
    "to": "In-Context Learning",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "In-Context Learning",
    "to": "Few-Shot Prompting",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "In-Context Learning",
    "to": "Zero-Shot Prompting",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "GPT-3",
    "to": "Emergent Capabilities",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Emergent Capabilities",
    "to": "Chain-of-Thought",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Emergent Capabilities",
    "to": "Arithmetic Reasoning",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Emergent Capabilities",
    "to": "Code Generation",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Code Generation",
    "to": "Codex",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Codex",
    "to": "GitHub Copilot",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Large Language Models",
    "to": "Scaling Laws",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Scaling Laws",
    "to": "Chinchilla Scaling Laws",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Chinchilla Scaling Laws",
    "to": "Compute-Optimal Training",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Compute-Optimal Training",
    "to": "Data-Model Balance",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Compute-Optimal Training",
    "to": "LLaMA (efficient pretraining)",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "LLaMA",
    "to": "Open-Weights LLMs",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Open-Weights LLMs",
    "to": "LLaMA 2",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Open-Weights LLMs",
    "to": "Mistral 7B",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "LLaMA",
    "to": "RoPE (LLaMA)",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "LLaMA",
    "to": "SwiGLU (LLaMA)",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "LLaMA",
    "to": "RMS Norm (LLaMA)",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "LLaMA",
    "to": "Pre-normalization (LLaMA)",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "LLaMA 2",
    "to": "Grouped Query Attention",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "LLaMA 2",
    "to": "Chat Fine-tuning",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Mistral 7B",
    "to": "Sliding Window Attention",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Mistral 7B",
    "to": "Rolling Buffer Cache",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Mixture of Experts",
    "to": "Sparse Models",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Sparse Models",
    "to": "Mixtral 8x7B",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Mixtral 8x7B",
    "to": "Top-2 Expert Routing",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Top-2 Expert Routing",
    "to": "Efficient MoE",
    "domain": "LLMs",
    "section": "Architecture"
  },
  {
    "from": "Large Language Models",
    "to": "Pre-training",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Pre-training",
    "to": "Data Collection",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Data Collection",
    "to": "Web Crawls",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Web Crawls",
    "to": "Common Crawl",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Common Crawl",
    "to": "Data Filtering",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Data Filtering",
    "to": "Deduplication",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Deduplication",
    "to": "MinHash LSH",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Deduplication",
    "to": "Exact Dedup",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Data Collection",
    "to": "Books",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Data Collection",
    "to": "GitHub Code",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Data Collection",
    "to": "Wikipedia",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Pre-training",
    "to": "Distributed Training",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Distributed Training",
    "to": "Data Parallelism",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Data Parallelism",
    "to": "DDP (PyTorch)",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Distributed Training",
    "to": "Model Parallelism",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Model Parallelism",
    "to": "Tensor Parallelism",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Model Parallelism",
    "to": "Pipeline Parallelism",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Distributed Training",
    "to": "ZeRO Optimization",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "ZeRO Optimization",
    "to": "DeepSpeed",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "ZeRO Optimization",
    "to": "FSDP",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Pre-training",
    "to": "Mixed Precision Training",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Mixed Precision Training",
    "to": "FP16",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Mixed Precision Training",
    "to": "BF16",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "BF16",
    "to": "LLM Training Stability",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Pre-training",
    "to": "Gradient Checkpointing",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Gradient Checkpointing",
    "to": "Memory Efficiency",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Pre-training",
    "to": "Flash Attention",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Flash Attention",
    "to": "IO-Aware Computation",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Flash Attention",
    "to": "FlashAttention-2",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "FlashAttention-2",
    "to": "Training Throughput",
    "domain": "LLMs",
    "section": "Training"
  },
  {
    "from": "Large Language Models",
    "to": "Alignment",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Alignment",
    "to": "Instruction Tuning",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Instruction Tuning",
    "to": "FLAN",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "FLAN",
    "to": "Flan-T5",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Instruction Tuning",
    "to": "Alpaca",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Alpaca",
    "to": "Self-Instruct",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Instruction Tuning",
    "to": "Vicuna",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Instruction Tuning",
    "to": "WizardLM",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Alignment",
    "to": "RLHF",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "RLHF",
    "to": "Reward Model Training",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Reward Model Training",
    "to": "Human Preference Data",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Human Preference Data",
    "to": "Comparative Annotations",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "RLHF",
    "to": "Supervised Fine-tuning",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Supervised Fine-tuning",
    "to": "SFT Model",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "SFT Model",
    "to": "PPO Fine-tuning",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "PPO Fine-tuning",
    "to": "InstructGPT",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "InstructGPT",
    "to": "ChatGPT",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "RLHF",
    "to": "KL Penalty",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "KL Penalty",
    "to": "KL Divergence (RLHF)",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Alignment",
    "to": "Constitutional AI",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Constitutional AI",
    "to": "Self-Critique",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Constitutional AI",
    "to": "RLAIF",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "RLAIF",
    "to": "AI-Generated Preference Data",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "RLAIF",
    "to": "Scalable Oversight",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Alignment",
    "to": "DPO",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "DPO",
    "to": "Bradley-Terry Model",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "DPO",
    "to": "Reference Model",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "DPO",
    "to": "ORPO",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "ORPO",
    "to": "Odds Ratio Preference",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Safety Fine-tuning",
    "to": "Refusal Training",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Refusal Training",
    "to": "Red-Teaming",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Red-Teaming",
    "to": "Jailbreaks",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Jailbreaks",
    "to": "Adversarial Prompts",
    "domain": "LLMs",
    "section": "Alignment"
  },
  {
    "from": "Fine-tuning",
    "to": "Parameter-Efficient Fine-Tuning",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "Parameter-Efficient Fine-Tuning",
    "to": "Adapter Layers",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "Adapter Layers",
    "to": "Houlsby Adapters",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "Adapter Layers",
    "to": "Bottleneck Adapters",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "Parameter-Efficient Fine-Tuning",
    "to": "Prefix Tuning",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "Prefix Tuning",
    "to": "Soft Prompts",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "Soft Prompts",
    "to": "Prompt Tuning",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "Parameter-Efficient Fine-Tuning",
    "to": "LoRA",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "LoRA",
    "to": "Low-Rank Decomposition",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "Low-Rank Decomposition",
    "to": "A Matrix x B Matrix",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "LoRA",
    "to": "Rank Selection",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "Rank Selection",
    "to": "LoRA Hyperparameters",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "LoRA",
    "to": "QLoRA",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "QLoRA",
    "to": "4-bit Quantization",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "4-bit Quantization",
    "to": "NF4 Quantization",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "QLoRA",
    "to": "Double Quantization",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "LoRA",
    "to": "LoRA+",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "LoRA",
    "to": "DoRA",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "DoRA",
    "to": "Weight Decomposition",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "LoRA",
    "to": "Merged Weights",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "Merged Weights",
    "to": "Zero Inference Overhead",
    "domain": "LLMs",
    "section": "Efficient Fine-tuning"
  },
  {
    "from": "Large Language Models",
    "to": "Inference",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Inference",
    "to": "Autoregressive Decoding",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Autoregressive Decoding",
    "to": "KV Cache",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "KV Cache",
    "to": "Memory Bottleneck",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "KV Cache",
    "to": "Multi-Query Attention",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Multi-Query Attention",
    "to": "Grouped Query Attention",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "KV Cache",
    "to": "PagedAttention",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "PagedAttention",
    "to": "vLLM",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "vLLM",
    "to": "High-Throughput Inference",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Inference",
    "to": "Quantization",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Quantization",
    "to": "Post-Training Quantization",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Post-Training Quantization",
    "to": "INT8 Quantization",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Post-Training Quantization",
    "to": "INT4 Quantization",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Post-Training Quantization",
    "to": "GPTQ",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "GPTQ",
    "to": "Weight-Only Quantization",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Post-Training Quantization",
    "to": "AWQ",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "AWQ",
    "to": "Activation-Aware Quantization",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Quantization",
    "to": "Quantization-Aware Training",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Inference",
    "to": "Speculative Decoding",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Speculative Decoding",
    "to": "Draft Model",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Draft Model",
    "to": "Verification Step",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Speculative Decoding",
    "to": "Self-Speculative Decoding",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Inference",
    "to": "Continuous Batching",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Continuous Batching",
    "to": "Dynamic Batching",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Dynamic Batching",
    "to": "Throughput Optimization",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Inference",
    "to": "Model Pruning",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Model Pruning",
    "to": "Unstructured Pruning",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Model Pruning",
    "to": "Structured Pruning",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Structured Pruning",
    "to": "LLM.int8",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Inference",
    "to": "Knowledge Distillation (LLM)",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Knowledge Distillation (LLM)",
    "to": "DistilGPT",
    "domain": "LLMs",
    "section": "Inference Optimization"
  },
  {
    "from": "Context Window",
    "to": "Positional Encoding",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Positional Encoding",
    "to": "Context Length Limitations",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Context Length Limitations",
    "to": "Long-Context Models",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Long-Context Models",
    "to": "RoPE Extension",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "RoPE Extension",
    "to": "YaRN",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "RoPE Extension",
    "to": "Linear Scaling",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Long-Context Models",
    "to": "Sliding Window Attention",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Long-Context Models",
    "to": "Ring Attention",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Ring Attention",
    "to": "Cross-Device Context",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Long-Context Models",
    "to": "Sparse Attention (Long Context)",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Large Language Models",
    "to": "Prompting",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Prompting",
    "to": "Zero-Shot Prompting",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Prompting",
    "to": "Few-Shot Prompting",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Few-Shot Prompting",
    "to": "In-Context Examples",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "In-Context Examples",
    "to": "Example Selection",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Example Selection",
    "to": "KATE",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "In-Context Learning",
    "to": "Emergent Capabilities (Prompting)",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Chain-of-Thought",
    "to": "Zero-Shot CoT",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Zero-Shot CoT",
    "to": "\"Let's Think Step by Step\"",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Chain-of-Thought",
    "to": "Few-Shot CoT",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Few-Shot CoT",
    "to": "Manual CoT Examples",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Chain-of-Thought",
    "to": "Self-Consistency",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Self-Consistency",
    "to": "Majority Voting",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Self-Consistency",
    "to": "Ensemble Reasoning",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Chain-of-Thought",
    "to": "Tree of Thought",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Tree of Thought",
    "to": "Search-Based Reasoning",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Tree of Thought",
    "to": "BFS/DFS over Thoughts",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Chain-of-Thought",
    "to": "Program of Thought",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Program of Thought",
    "to": "Code Execution",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Prompting",
    "to": "Structured Outputs",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Structured Outputs",
    "to": "JSON Mode",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Structured Outputs",
    "to": "Tool Calling",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Tool Calling",
    "to": "Function Calling API",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Function Calling API",
    "to": "LLM Agents",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Prompting",
    "to": "Retrieval Augmentation",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "Retrieval Augmentation",
    "to": "RAG",
    "domain": "LLMs",
    "section": "Context & Prompting"
  },
  {
    "from": "TF-IDF",
    "to": "Sparse Retrieval",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Sparse Retrieval",
    "to": "BM25",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "BM25",
    "to": "Inverted Index",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Inverted Index",
    "to": "Elasticsearch",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Inverted Index",
    "to": "Solr",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "BM25",
    "to": "Term Frequency",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "BM25",
    "to": "Inverse Document Frequency",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "BM25",
    "to": "Document Length Normalization",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Semantic Search",
    "to": "Dense Retrieval",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Dense Retrieval",
    "to": "Bi-Encoder Architecture",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Bi-Encoder Architecture",
    "to": "Query Encoder",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Bi-Encoder Architecture",
    "to": "Document Encoder",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Dense Retrieval",
    "to": "Sentence Embeddings",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Sentence Embeddings",
    "to": "Sentence-BERT (Retrieval)",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Sentence Embeddings",
    "to": "E5 Embeddings",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Sentence Embeddings",
    "to": "BGE Embeddings",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Sentence Embeddings",
    "to": "OpenAI Embeddings",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Dense Retrieval",
    "to": "Contrastive Training (Retrieval)",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Contrastive Training (Retrieval)",
    "to": "In-batch Negatives",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "In-batch Negatives",
    "to": "Hard Negatives",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Hard Negatives",
    "to": "Mining Strategies",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Dense Retrieval",
    "to": "Approximate Nearest Neighbors",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Approximate Nearest Neighbors",
    "to": "FAISS",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Approximate Nearest Neighbors",
    "to": "HNSW",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "HNSW",
    "to": "Hierarchical Navigable Small World",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "HNSW",
    "to": "Logarithmic Search",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "FAISS",
    "to": "IVF Index",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "FAISS",
    "to": "Product Quantization",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Product Quantization",
    "to": "Memory Compression (Vector)",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Vector Databases",
    "to": "Pinecone",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Vector Databases",
    "to": "Weaviate",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Vector Databases",
    "to": "Qdrant",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Vector Databases",
    "to": "Chroma",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Vector Databases",
    "to": "Milvus",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Re-ranking",
    "to": "Cross-Encoder",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Cross-Encoder",
    "to": "BERT Re-ranker",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Cross-Encoder",
    "to": "Score Distillation",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Re-ranking",
    "to": "ColBERT",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "ColBERT",
    "to": "Late Interaction",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Late Interaction",
    "to": "Token-Level Matching",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Re-ranking",
    "to": "Listwise Re-ranking",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Sparse Retrieval",
    "to": "Hybrid Search",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Dense Retrieval",
    "to": "Hybrid Search",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "Hybrid Search",
    "to": "RRF (Reciprocal Rank Fusion)",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "RRF",
    "to": "Score Normalization",
    "domain": "RAG",
    "section": "Information Retrieval"
  },
  {
    "from": "RAG",
    "to": "Document Ingestion",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Document Ingestion",
    "to": "PDF Extraction",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "PDF Extraction",
    "to": "pdfminer",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "PDF Extraction",
    "to": "PyMuPDF",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Document Ingestion",
    "to": "HTML Parsing",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "HTML Parsing",
    "to": "BeautifulSoup",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Document Ingestion",
    "to": "Text Splitting",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Text Splitting",
    "to": "Fixed-Size Chunking",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Text Splitting",
    "to": "Recursive Splitting",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Text Splitting",
    "to": "Semantic Chunking",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Text Splitting",
    "to": "Overlap",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Semantic Chunking",
    "to": "Embedding-Based Segmentation",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Text Splitting",
    "to": "Parent Document Retrieval",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Parent Document Retrieval",
    "to": "Small-to-Big Retrieval",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Document Ingestion",
    "to": "Embedding Generation",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Embedding Generation",
    "to": "Embedding Model",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Embedding Model",
    "to": "FAISS Index",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "FAISS Index",
    "to": "Similarity Search",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Similarity Search",
    "to": "Top-K Retrieval",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Top-K Retrieval",
    "to": "Context Assembly",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Context Assembly",
    "to": "Context Window",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Context Assembly",
    "to": "Augmented Prompt",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Augmented Prompt",
    "to": "LLM Generation",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "LLM Generation",
    "to": "Grounded Answer",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "RAG",
    "to": "Query Processing",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Query Processing",
    "to": "Query Embedding",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Query Embedding",
    "to": "Nearest Neighbor Search",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "RAG",
    "to": "Advanced RAG",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Advanced RAG",
    "to": "HyDE",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "HyDE",
    "to": "Hypothetical Document Embedding",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "HyDE",
    "to": "Query Expansion",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Query Expansion",
    "to": "Multi-Query Retrieval",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Multi-Query Retrieval",
    "to": "RAG Fusion",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "RAG Fusion",
    "to": "Multiple Perspectives",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Advanced RAG",
    "to": "Self-RAG",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Self-RAG",
    "to": "Retrieval Decision",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Self-RAG",
    "to": "Reflection Tokens",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Self-RAG",
    "to": "Critique",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Advanced RAG",
    "to": "Corrective RAG",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Corrective RAG",
    "to": "Retrieval Refinement",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Advanced RAG",
    "to": "Hierarchical Retrieval",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Advanced RAG",
    "to": "Multi-Hop Retrieval",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Multi-Hop Retrieval",
    "to": "Iterative Retrieval",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "RAG",
    "to": "Metadata Filtering",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Metadata Filtering",
    "to": "Structured RAG",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Structured RAG",
    "to": "Text-to-SQL",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Text-to-SQL",
    "to": "SQL Execution",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "SQL Execution",
    "to": "Database Query",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "RAG",
    "to": "Query Routing",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Query Routing",
    "to": "Multi-Index RAG",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "Multi-Index RAG",
    "to": "Domain-Specific Retrieval",
    "domain": "RAG",
    "section": "Pipeline"
  },
  {
    "from": "RAG Evaluation",
    "to": "Faithfulness",
    "domain": "RAG",
    "section": "Evaluation"
  },
  {
    "from": "RAG Evaluation",
    "to": "Answer Relevance",
    "domain": "RAG",
    "section": "Evaluation"
  },
  {
    "from": "RAG Evaluation",
    "to": "Context Precision",
    "domain": "RAG",
    "section": "Evaluation"
  },
  {
    "from": "RAG Evaluation",
    "to": "Context Recall",
    "domain": "RAG",
    "section": "Evaluation"
  },
  {
    "from": "RAGAS",
    "to": "Faithfulness Score",
    "domain": "RAG",
    "section": "Evaluation"
  },
  {
    "from": "RAGAS",
    "to": "Answer Relevance Score",
    "domain": "RAG",
    "section": "Evaluation"
  },
  {
    "from": "RAGAS",
    "to": "Context Precision Score",
    "domain": "RAG",
    "section": "Evaluation"
  },
  {
    "from": "RAGAS",
    "to": "Context Recall Score",
    "domain": "RAG",
    "section": "Evaluation"
  },
  {
    "from": "RAG Evaluation",
    "to": "LLM-as-Judge",
    "domain": "RAG",
    "section": "Evaluation"
  },
  {
    "from": "LLM-as-Judge",
    "to": "G-Eval",
    "domain": "RAG",
    "section": "Evaluation"
  },
  {
    "from": "G-Eval",
    "to": "Criteria-Based Scoring",
    "domain": "RAG",
    "section": "Evaluation"
  },
  {
    "from": "LLM-as-Judge",
    "to": "MT-Bench",
    "domain": "RAG",
    "section": "Evaluation"
  },
  {
    "from": "MT-Bench",
    "to": "Multi-Turn Evaluation",
    "domain": "RAG",
    "section": "Evaluation"
  },
  {
    "from": "RAG Evaluation",
    "to": "Hallucination Detection",
    "domain": "RAG",
    "section": "Evaluation"
  },
  {
    "from": "Hallucination Detection",
    "to": "NLI-Based Detection",
    "domain": "RAG",
    "section": "Evaluation"
  },
  {
    "from": "Hallucination Detection",
    "to": "Token Attribution",
    "domain": "RAG",
    "section": "Evaluation"
  },
  {
    "from": "Large Language Models",
    "to": "LLM Agents",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "LLM Agents",
    "to": "Reasoning",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "LLM Agents",
    "to": "Acting",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "LLM Agents",
    "to": "Tool Use",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Reasoning",
    "to": "Chain-of-Thought (Agents)",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Chain-of-Thought (Agents)",
    "to": "Decomposed Reasoning",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Acting",
    "to": "Tool Selection",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Tool Selection",
    "to": "Tool Descriptions",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Tool Descriptions",
    "to": "Tool Schemas",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Tool Use",
    "to": "Function Calling",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Function Calling",
    "to": "JSON Arguments",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Function Calling",
    "to": "Tool Results",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Tool Use",
    "to": "Web Search",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Web Search",
    "to": "Search Engine API",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Tool Use",
    "to": "Code Execution",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Code Execution",
    "to": "Python Interpreter",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Code Execution",
    "to": "Sandboxed Execution",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Tool Use",
    "to": "File Operations",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Tool Use",
    "to": "Database Queries",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Tool Use",
    "to": "Calculator",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "LLM Agents",
    "to": "ReAct",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "ReAct",
    "to": "Reasoning Traces",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "ReAct",
    "to": "Action Execution",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "ReAct",
    "to": "Observation Integration",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "ReAct",
    "to": "Thought-Action-Observation Loop",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "ReAct",
    "to": "Iterative Refinement",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "LLM Agents",
    "to": "Agent Loop",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Agent Loop",
    "to": "Stopping Condition",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Agent Loop",
    "to": "Max Iterations",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "Agent Loop",
    "to": "Task Completion Check",
    "domain": "AGENTS",
    "section": "Foundations"
  },
  {
    "from": "LLM Agents",
    "to": "Memory Systems",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Memory Systems",
    "to": "In-Context Memory",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "In-Context Memory",
    "to": "Context Window (Agents)",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Memory Systems",
    "to": "External Memory",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "External Memory",
    "to": "Vector Memory",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Vector Memory",
    "to": "Episodic Memory Store",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Episodic Memory Store",
    "to": "Past Experiences",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Memory Systems",
    "to": "Semantic Memory",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Semantic Memory",
    "to": "Knowledge Bases",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Knowledge Bases",
    "to": "Structured Knowledge",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Knowledge Bases",
    "to": "Knowledge Graphs (Agents)",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Memory Systems",
    "to": "Working Memory",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Working Memory",
    "to": "Scratchpad",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Scratchpad",
    "to": "Intermediate Results",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Memory Systems",
    "to": "Long-term Memory",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Long-term Memory",
    "to": "Persistent Storage",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Persistent Storage",
    "to": "Redis",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Persistent Storage",
    "to": "SQLite",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Memory",
    "to": "MemGPT",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "MemGPT",
    "to": "Memory Management Protocol",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Memory",
    "to": "Generative Agents",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Generative Agents",
    "to": "Memory Stream",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Memory Stream",
    "to": "Retrieval-Based Memory",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "Retrieval-Based Memory",
    "to": "Relevance Scoring",
    "domain": "AGENTS",
    "section": "Memory"
  },
  {
    "from": "LLM Agents",
    "to": "Planning",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Planning",
    "to": "Task Decomposition",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Task Decomposition",
    "to": "Subgoal Generation",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Subgoal Generation",
    "to": "Hierarchical Planning",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Hierarchical Planning",
    "to": "High-Level Plans",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Hierarchical Planning",
    "to": "Low-Level Actions",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Planning",
    "to": "Plan-and-Execute",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Plan-and-Execute",
    "to": "Planner",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Plan-and-Execute",
    "to": "Executor",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Planner",
    "to": "Static Planning",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Planner",
    "to": "Dynamic Re-planning",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Dynamic Re-planning",
    "to": "Error Recovery",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Planning",
    "to": "Tree Search (Agents)",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Tree Search (Agents)",
    "to": "BFS Planning",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Tree Search (Agents)",
    "to": "DFS Planning",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Tree Search (Agents)",
    "to": "MCTS (Agents)",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "MCTS (Agents)",
    "to": "Self-Play",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "MCTS (Agents)",
    "to": "Rollout Policy",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Planning",
    "to": "Self-Refinement",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Self-Refinement",
    "to": "Critic",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Critic",
    "to": "Feedback Loop",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Feedback Loop",
    "to": "Iterative Improvement",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "Planning",
    "to": "Backtracking",
    "domain": "AGENTS",
    "section": "Planning"
  },
  {
    "from": "LLM Agents",
    "to": "Multi-Agent Systems",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "Multi-Agent Systems",
    "to": "Agent Orchestration",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "Agent Orchestration",
    "to": "AutoGen",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "Agent Orchestration",
    "to": "LangGraph",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "LangGraph",
    "to": "State Machine Agents",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "LangGraph",
    "to": "Conditional Edges",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "Multi-Agent Systems",
    "to": "Role-Based Agents",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "Role-Based Agents",
    "to": "Manager Agent",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "Role-Based Agents",
    "to": "Specialist Agents",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "Role-Based Agents",
    "to": "Debater Agents",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "Debater Agents",
    "to": "Multi-Agent Debate",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "Multi-Agent Debate",
    "to": "Consensus",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "Multi-Agent Systems",
    "to": "Agent Communication",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "Agent Communication",
    "to": "Message Passing (Agents)",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "Message Passing (Agents)",
    "to": "Shared State",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "Multi-Agent Systems",
    "to": "CrewAI",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "CrewAI",
    "to": "Crew",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "CrewAI",
    "to": "Task Assignment",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "Multi-Agent Systems",
    "to": "MetaGPT",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "MetaGPT",
    "to": "Software Development Pipeline",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "MetaGPT",
    "to": "Role Prompts",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "LLM Agents",
    "to": "LangChain",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "LangChain",
    "to": "Chains",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "LangChain",
    "to": "Memory (LangChain)",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "LangChain",
    "to": "Agents (LangChain)",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "LangChain",
    "to": "LangSmith",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "LangSmith",
    "to": "Tracing",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "LangSmith",
    "to": "Evaluation (Agents)",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "LLM Agents",
    "to": "LlamaIndex",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "LlamaIndex",
    "to": "Data Connectors",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "LlamaIndex",
    "to": "Query Engine",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "LlamaIndex",
    "to": "Agentic RAG",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "Agentic RAG",
    "to": "Tool-Augmented RAG",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "Tool-Augmented RAG",
    "to": "Search + Generate",
    "domain": "AGENTS",
    "section": "Multi-Agent & Frameworks"
  },
  {
    "from": "Probability Theory",
    "to": "Reinforcement Learning",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Reinforcement Learning",
    "to": "MDP",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "MDP",
    "to": "States",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "MDP",
    "to": "Actions",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "MDP",
    "to": "Rewards",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "MDP",
    "to": "Transition Dynamics",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "MDP",
    "to": "Discount Factor",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "MDP",
    "to": "Episode",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "MDP",
    "to": "Policy",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Policy",
    "to": "Deterministic Policy",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Policy",
    "to": "Stochastic Policy",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Stochastic Policy",
    "to": "Policy Entropy",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "MDP",
    "to": "Value Functions",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Value Functions",
    "to": "State Value Function V",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Value Functions",
    "to": "Action-Value Function Q",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "State Value Function V",
    "to": "Bellman Equation (V)",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Action-Value Function Q",
    "to": "Bellman Equation (Q)",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Bellman Equation (Q)",
    "to": "Q-Learning",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Bellman Equation (V)",
    "to": "Value Iteration",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Value Iteration",
    "to": "Policy Iteration",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Policy Iteration",
    "to": "Policy Evaluation",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Policy Iteration",
    "to": "Policy Improvement",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Policy Improvement",
    "to": "Greedy Policy",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Greedy Policy",
    "to": "Exploration-Exploitation Tradeoff",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Exploration-Exploitation Tradeoff",
    "to": "Epsilon-Greedy",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Exploration-Exploitation Tradeoff",
    "to": "UCB",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "UCB",
    "to": "Upper Confidence Bound",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Exploration-Exploitation Tradeoff",
    "to": "Thompson Sampling",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Thompson Sampling",
    "to": "Bayesian Exploration",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Reinforcement Learning",
    "to": "Model-Free RL",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Reinforcement Learning",
    "to": "Model-Based RL",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Model-Free RL",
    "to": "On-Policy Methods",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Model-Free RL",
    "to": "Off-Policy Methods",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "On-Policy Methods",
    "to": "PPO",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "On-Policy Methods",
    "to": "TRPO",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "On-Policy Methods",
    "to": "A3C",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Off-Policy Methods",
    "to": "Q-Learning",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Off-Policy Methods",
    "to": "DQN",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Off-Policy Methods",
    "to": "SAC",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Off-Policy Methods",
    "to": "DDPG",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Foundations"
  },
  {
    "from": "Q-Learning",
    "to": "Tabular Q-Learning",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Tabular Q-Learning",
    "to": "Deep Q-Learning",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Deep Q-Learning",
    "to": "DQN",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "DQN",
    "to": "Experience Replay",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Experience Replay",
    "to": "Replay Buffer",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "DQN",
    "to": "Target Network",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Target Network",
    "to": "Fixed Q-Targets",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "DQN",
    "to": "Double DQN",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Double DQN",
    "to": "Overestimation Correction",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "DQN",
    "to": "Dueling DQN",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Dueling DQN",
    "to": "Value Stream",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Dueling DQN",
    "to": "Advantage Stream",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "DQN",
    "to": "Prioritized Experience Replay",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Prioritized Experience Replay",
    "to": "TD Error Priority",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "DQN",
    "to": "Noisy Networks",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Noisy Networks",
    "to": "Stochastic Exploration",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "DQN",
    "to": "N-Step Returns",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "N-Step Returns",
    "to": "Multi-Step TD",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "DQN",
    "to": "Distributional RL",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Distributional RL",
    "to": "C51",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "C51",
    "to": "Categorical DQN",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Distributional RL",
    "to": "IQN",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Double DQN",
    "to": "Rainbow",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Dueling DQN",
    "to": "Rainbow",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Prioritized Experience Replay",
    "to": "Rainbow",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Noisy Networks",
    "to": "Rainbow",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "N-Step Returns",
    "to": "Rainbow",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Distributional RL",
    "to": "Rainbow",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Rainbow",
    "to": "State-of-the-Art Atari",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Value-Based"
  },
  {
    "from": "Policy Gradient",
    "to": "REINFORCE",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "REINFORCE",
    "to": "Monte Carlo Returns",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "Monte Carlo Returns",
    "to": "High Variance",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "High Variance",
    "to": "Baseline Subtraction",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "Baseline Subtraction",
    "to": "Actor-Critic",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "Actor-Critic",
    "to": "Actor (Policy)",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "Actor-Critic",
    "to": "Critic (Value)",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "Actor-Critic",
    "to": "A2C",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "A2C",
    "to": "Synchronous Updates",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "Actor-Critic",
    "to": "A3C",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "A3C",
    "to": "Asynchronous Actors",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "Actor-Critic",
    "to": "PPO",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "PPO",
    "to": "Clipped Surrogate Objective",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "Clipped Surrogate Objective",
    "to": "Trust Region",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "PPO",
    "to": "GAE (Generalized Advantage Estimation)",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "GAE",
    "to": "Advantage Function",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "Actor-Critic",
    "to": "TRPO",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "TRPO",
    "to": "Conjugate Gradient",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "TRPO",
    "to": "Trust Region Constraint",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "Actor-Critic",
    "to": "SAC",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "SAC",
    "to": "Maximum Entropy RL",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "Maximum Entropy RL",
    "to": "Entropy Bonus",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "SAC",
    "to": "Twin Critics",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "Twin Critics",
    "to": "Overestimation Prevention",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "Actor-Critic",
    "to": "DDPG",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "DDPG",
    "to": "Deterministic Policy Gradient",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "DDPG",
    "to": "Ornstein-Uhlenbeck Noise",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "DDPG",
    "to": "TD3",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "TD3",
    "to": "Target Policy Smoothing",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "TD3",
    "to": "Delayed Updates",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "TD3",
    "to": "Twin Critics (TD3)",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Policy Gradient"
  },
  {
    "from": "Model-Based RL",
    "to": "World Models",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "World Models",
    "to": "Environment Model",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "Environment Model",
    "to": "Transition Prediction",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "Environment Model",
    "to": "Reward Prediction",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "World Models",
    "to": "Dreamer",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "Dreamer",
    "to": "RSSM",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "RSSM",
    "to": "Recurrent State Space",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "RSSM",
    "to": "Imagination Rollouts",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "Imagination Rollouts",
    "to": "Latent Space Planning",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "Dreamer",
    "to": "DreamerV2",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "DreamerV2",
    "to": "Discrete Latent States",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "DreamerV2",
    "to": "KL Balancing",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "Model-Based RL",
    "to": "MCTS",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "MCTS",
    "to": "Simulation",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "MCTS",
    "to": "Selection",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "MCTS",
    "to": "Expansion",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "MCTS",
    "to": "Backpropagation (MCTS)",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "MCTS",
    "to": "AlphaGo",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "AlphaGo",
    "to": "Policy Network",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "AlphaGo",
    "to": "Value Network",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "AlphaGo",
    "to": "Self-Play Data",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "Self-Play Data",
    "to": "AlphaGo Zero",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "AlphaGo Zero",
    "to": "Tabula Rasa Learning",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "AlphaGo Zero",
    "to": "AlphaZero",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "AlphaZero",
    "to": "Generalized Self-Play",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "AlphaZero",
    "to": "MuZero",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "MuZero",
    "to": "Learned Dynamics",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "Learned Dynamics",
    "to": "Latent State Transitions",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "Model-Based RL",
    "to": "Dyna-Q",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "Dyna-Q",
    "to": "Simulated Experience",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "Simulated Experience",
    "to": "Augmented Replay",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Model-Based"
  },
  {
    "from": "Reinforcement Learning",
    "to": "Exploration",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Exploration",
    "to": "Count-Based Exploration",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Count-Based Exploration",
    "to": "Pseudo-Counts",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Exploration",
    "to": "Intrinsic Motivation",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Intrinsic Motivation",
    "to": "Curiosity-Driven Exploration",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Curiosity-Driven Exploration",
    "to": "ICM",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "ICM",
    "to": "Prediction Error as Reward",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Exploration",
    "to": "Random Network Distillation",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Random Network Distillation",
    "to": "Novelty Detection",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Reinforcement Learning",
    "to": "Reward Shaping",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Reward Shaping",
    "to": "Potential-Based Shaping",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Reinforcement Learning",
    "to": "Sparse Rewards",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Sparse Rewards",
    "to": "HER",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "HER",
    "to": "Hindsight Goal Relabeling",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Reinforcement Learning",
    "to": "Multi-Task RL",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Multi-Task RL",
    "to": "Transfer in RL",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Transfer in RL",
    "to": "Meta-RL",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Meta-RL",
    "to": "MAML (RL)",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "MAML (RL)",
    "to": "Few-Shot Adaptation",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Reinforcement Learning",
    "to": "Offline RL",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Offline RL",
    "to": "Conservative Q-Learning",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Conservative Q-Learning",
    "to": "CQL",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Offline RL",
    "to": "Decision Transformer",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Decision Transformer",
    "to": "Return-Conditioned Generation",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Decision Transformer",
    "to": "LLM as Policy",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Reinforcement Learning",
    "to": "Multi-Agent RL",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Multi-Agent RL",
    "to": "Cooperative RL",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Multi-Agent RL",
    "to": "Competitive RL",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Competitive RL",
    "to": "Nash Equilibrium",
    "domain": "REINFORCEMENT LEARNING",
    "section": "Special Topics"
  },
  {
    "from": "Images",
    "to": "Pixels",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Pixels",
    "to": "Color Channels",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Color Channels",
    "to": "RGB",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "RGB",
    "to": "HSV",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "RGB",
    "to": "Grayscale",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Images",
    "to": "Image Resolution",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Images",
    "to": "Aspect Ratio",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Images",
    "to": "Image Representation",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Image Representation",
    "to": "Histograms",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Histograms",
    "to": "Histogram of Oriented Gradients",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Histogram of Oriented Gradients",
    "to": "Object Detection (Classical)",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Images",
    "to": "Image Processing",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Image Processing",
    "to": "Convolution (Signal)",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Convolution (Signal)",
    "to": "Edge Detection",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Edge Detection",
    "to": "Sobel Filter",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Edge Detection",
    "to": "Canny Edge Detector",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Canny Edge Detector",
    "to": "Hysteresis Thresholding",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Image Processing",
    "to": "Filtering",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Filtering",
    "to": "Gaussian Blur",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Filtering",
    "to": "Median Filter",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Image Processing",
    "to": "Morphological Operations",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Morphological Operations",
    "to": "Erosion",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Morphological Operations",
    "to": "Dilation",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Image Processing",
    "to": "Frequency Domain",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Frequency Domain",
    "to": "Fourier Transform (Images)",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Fourier Transform (Images)",
    "to": "Frequency Filtering",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Image Processing",
    "to": "Image Normalization",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Image Normalization",
    "to": "Zero-Mean",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Image Normalization",
    "to": "Channel Standardization",
    "domain": "COMPUTER VISION",
    "section": "Foundations"
  },
  {
    "from": "Image Processing",
    "to": "Feature Detection",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Feature Detection",
    "to": "SIFT",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "SIFT",
    "to": "Scale-Invariant Keypoints",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Feature Detection",
    "to": "SURF",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Feature Detection",
    "to": "ORB",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Feature Detection",
    "to": "Harris Corner Detector",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Harris Corner Detector",
    "to": "Corner Response",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Feature Matching",
    "to": "Homography",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Homography",
    "to": "Image Stitching",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Image Processing",
    "to": "Optical Flow",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Optical Flow",
    "to": "Lucas-Kanade",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Optical Flow",
    "to": "Horn-Schunck",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Image Processing",
    "to": "Classical Segmentation",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Classical Segmentation",
    "to": "Watershed",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Classical Segmentation",
    "to": "GrabCut",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "GrabCut",
    "to": "Graph Cuts",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Graph Cuts",
    "to": "Energy Minimization",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Image Processing",
    "to": "Template Matching",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Template Matching",
    "to": "Sliding Window",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Sliding Window",
    "to": "Early Detection",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Bag of Visual Words",
    "to": "K-Means Clustering (Vision)",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "K-Means Clustering (Vision)",
    "to": "Visual Vocabulary",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Visual Vocabulary",
    "to": "Fisher Vectors",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "Fisher Vectors",
    "to": "SVM Classification",
    "domain": "COMPUTER VISION",
    "section": "Classical Methods"
  },
  {
    "from": "CNNs",
    "to": "Image Classification",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "Image Classification",
    "to": "AlexNet",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "AlexNet",
    "to": "ReLU (Vision)",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "AlexNet",
    "to": "Dropout (Vision)",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "AlexNet",
    "to": "Data Augmentation (Vision)",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "AlexNet",
    "to": "VGGNet",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "VGGNet",
    "to": "Deep Stacking",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "VGGNet",
    "to": "Transfer Learning (Vision)",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "VGGNet",
    "to": "ResNet",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "ResNet",
    "to": "Skip Connections (Vision)",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "ResNet",
    "to": "Bottleneck Blocks (Vision)",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "ResNet",
    "to": "ResNet-50",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "ResNet",
    "to": "ResNet-101",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "ResNet-101",
    "to": "Feature Backbone",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "Feature Backbone",
    "to": "Object Detection (Deep)",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "Feature Backbone",
    "to": "Semantic Segmentation (Deep)",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "ResNet",
    "to": "ResNeXt",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "ResNet",
    "to": "DenseNet",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "DenseNet",
    "to": "Feature Reuse",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "Feature Reuse",
    "to": "Dense Connections",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "ResNet",
    "to": "EfficientNet",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "EfficientNet",
    "to": "Compound Scaling",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "Compound Scaling",
    "to": "Width Scaling",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "Compound Scaling",
    "to": "Depth Scaling",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "Compound Scaling",
    "to": "Resolution Scaling",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "EfficientNet",
    "to": "EfficientDet",
    "domain": "COMPUTER VISION",
    "section": "Deep CNN Models"
  },
  {
    "from": "Object Detection (Deep)",
    "to": "R-CNN",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "R-CNN",
    "to": "Selective Search",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "R-CNN",
    "to": "CNN Feature Extraction",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "R-CNN",
    "to": "Fast R-CNN",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "Fast R-CNN",
    "to": "RoI Pooling",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "Fast R-CNN",
    "to": "Faster R-CNN",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "Faster R-CNN",
    "to": "RPN",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "RPN",
    "to": "Anchor Boxes",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "Anchor Boxes",
    "to": "Multi-Scale Anchors",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "Faster R-CNN",
    "to": "Feature Pyramid Networks",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "Feature Pyramid Networks",
    "to": "Multi-Scale Features",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "Feature Pyramid Networks",
    "to": "Top-Down Pathway",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "Feature Pyramid Networks",
    "to": "Lateral Connections",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "Object Detection (Deep)",
    "to": "YOLO",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "YOLO",
    "to": "Grid Cell Prediction",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "YOLO",
    "to": "Bounding Box Regression",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "YOLO",
    "to": "Objectness Score",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "YOLO",
    "to": "YOLOv2",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "YOLOv2",
    "to": "YOLOv3",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "YOLOv3",
    "to": "Multi-Scale Prediction",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "YOLOv3",
    "to": "Darknet-53",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "YOLOv3",
    "to": "YOLOv4",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "YOLOv4",
    "to": "CSP Backbone",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "YOLOv4",
    "to": "PANet",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "YOLOv4",
    "to": "YOLOv5",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "YOLOv5",
    "to": "YOLOv8",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "YOLOv8",
    "to": "Anchor-Free",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "Anchor-Free",
    "to": "CenterNet",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "Anchor-Free",
    "to": "FCOS",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "FCOS",
    "to": "Centerness Branch",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "Object Detection (Deep)",
    "to": "SSD",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "SSD",
    "to": "Default Boxes",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "SSD",
    "to": "Multi-Box",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "SSD",
    "to": "RetinaNet",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "RetinaNet",
    "to": "Focal Loss",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "Focal Loss",
    "to": "Class Imbalance Solution",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "RetinaNet",
    "to": "EfficientDet",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "EfficientDet",
    "to": "BiFPN",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "BiFPN",
    "to": "Bidirectional FPN",
    "domain": "COMPUTER VISION",
    "section": "Object Detection"
  },
  {
    "from": "Transformers",
    "to": "Vision Transformers",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "Vision Transformers",
    "to": "ViT",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "ViT",
    "to": "Patch Embeddings",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "Patch Embeddings",
    "to": "Linear Projection",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "ViT",
    "to": "Class Token",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "ViT",
    "to": "Position Embeddings (Vision)",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "ViT",
    "to": "DeiT",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "DeiT",
    "to": "Distillation Token",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "DeiT",
    "to": "Data-Efficient Training",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "ViT",
    "to": "Swin Transformer",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "Swin Transformer",
    "to": "Shifted Windows",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "Shifted Windows",
    "to": "Local Attention",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "Swin Transformer",
    "to": "Hierarchical Features (Swin)",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "Hierarchical Features (Swin)",
    "to": "Dense Prediction",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "Dense Prediction",
    "to": "Detection (Swin)",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "Dense Prediction",
    "to": "Segmentation (Swin)",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "ViT",
    "to": "MAE",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "MAE",
    "to": "Masked Image Modeling",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "Masked Image Modeling",
    "to": "75% Masking",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "75% Masking",
    "to": "Efficient Pre-training",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "MAE",
    "to": "Self-Supervised Vision",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "Self-Supervised Vision",
    "to": "DINO",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "DINO",
    "to": "Self-Distillation",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "Self-Distillation",
    "to": "Knowledge Distillation (Vision)",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "DINO",
    "to": "DINOv2",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "DINOv2",
    "to": "Curated Data",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "DINOv2",
    "to": "Dense Features (Vision)",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "Vision Transformers",
    "to": "DETR",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "DETR",
    "to": "Object Queries",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "Object Queries",
    "to": "Bipartite Matching",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "Bipartite Matching",
    "to": "No NMS",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "DETR",
    "to": "Deformable DETR",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "Deformable DETR",
    "to": "Deformable Attention",
    "domain": "COMPUTER VISION",
    "section": "Transformers for Vision"
  },
  {
    "from": "Semantic Segmentation",
    "to": "FCN",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "FCN",
    "to": "Fully Convolutional",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "FCN",
    "to": "Upsampling",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "Upsampling",
    "to": "Bilinear Interpolation",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "Upsampling",
    "to": "Transposed Convolution",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "FCN",
    "to": "U-Net",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "U-Net",
    "to": "Skip Connections (UNet)",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "Skip Connections (UNet)",
    "to": "Precise Localization",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "U-Net",
    "to": "Medical Image Segmentation",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "Medical Image Segmentation",
    "to": "3D U-Net",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "3D U-Net",
    "to": "Volumetric Segmentation",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "Semantic Segmentation",
    "to": "DeepLab",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "DeepLab",
    "to": "Atrous Convolutions",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "Atrous Convolutions",
    "to": "Dilated Convolution",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "DeepLab",
    "to": "ASPP",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "ASPP",
    "to": "Multi-Scale Context",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "DeepLab",
    "to": "CRF Post-processing",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "CRF Post-processing",
    "to": "DeepLab v3+",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "DeepLab v3+",
    "to": "Encoder-Decoder Segmentation",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "Semantic Segmentation",
    "to": "SegFormer",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "SegFormer",
    "to": "Mix Transformer",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "Mix Transformer",
    "to": "Hierarchical Representation",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "Instance Segmentation",
    "to": "Mask R-CNN",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "Mask R-CNN",
    "to": "RoIAlign",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "Mask R-CNN",
    "to": "Mask Head",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "Mask R-CNN",
    "to": "Panoptic Segmentation",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "Panoptic Segmentation",
    "to": "Stuff Classes",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "Panoptic Segmentation",
    "to": "Things Classes",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "Semantic Segmentation",
    "to": "SAM",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "SAM",
    "to": "Promptable Segmentation",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "SAM",
    "to": "Point Prompt",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "SAM",
    "to": "Box Prompt",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "SAM",
    "to": "Mask Decoder",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "SAM",
    "to": "SAM 2",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "SAM 2",
    "to": "Video Segmentation",
    "domain": "COMPUTER VISION",
    "section": "Segmentation"
  },
  {
    "from": "ViT",
    "to": "CLIP",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "CLIP",
    "to": "Image Encoder",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "CLIP",
    "to": "Text Encoder",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "CLIP",
    "to": "Contrastive Pretraining (Vision)",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "Contrastive Pretraining (Vision)",
    "to": "Image-Text Alignment",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "Image-Text Alignment",
    "to": "Zero-Shot Classification (Vision)",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "Zero-Shot Classification (Vision)",
    "to": "Open-Vocabulary Detection",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "Open-Vocabulary Detection",
    "to": "Grounding DINO",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "CLIP",
    "to": "Image Retrieval",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "Image Retrieval",
    "to": "Reverse Image Search",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "CLIP",
    "to": "DALL-E",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "DALL-E",
    "to": "dVAE",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "dVAE",
    "to": "Visual Tokens",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "Visual Tokens",
    "to": "GPT-Image",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "CLIP",
    "to": "DALL-E 2",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "DALL-E 2",
    "to": "CLIP Image Prior",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "CLIP",
    "to": "Stable Diffusion",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "Stable Diffusion",
    "to": "Text Conditioning",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "Stable Diffusion",
    "to": "ControlNet",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "ControlNet",
    "to": "Conditioning Images",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "Conditioning Images",
    "to": "Pose-Conditioned Generation",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "Conditioning Images",
    "to": "Edge-Conditioned Generation",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "CLIP",
    "to": "LLaVA",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "LLaVA",
    "to": "Visual Instruction Tuning",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "Visual Instruction Tuning",
    "to": "Multimodal LLMs",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "Multimodal LLMs",
    "to": "GPT-4V",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "Multimodal LLMs",
    "to": "Gemini Vision",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "Multimodal LLMs",
    "to": "BLIP-2",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "BLIP-2",
    "to": "Q-Former",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "Q-Former",
    "to": "Visual-Language Bridge",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "Multimodal LLMs",
    "to": "Image Captioning",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "Image Captioning",
    "to": "VQA",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "VQA",
    "to": "Visual Reasoning",
    "domain": "COMPUTER VISION",
    "section": "Multimodal & Generation"
  },
  {
    "from": "3D Vision",
    "to": "Point Clouds",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Point Clouds",
    "to": "PointNet",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "PointNet",
    "to": "PointNet++",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "PointNet++",
    "to": "Hierarchical Point Learning",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "3D Vision",
    "to": "Depth Estimation",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Depth Estimation",
    "to": "Monocular Depth",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Monocular Depth",
    "to": "MiDaS",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Monocular Depth",
    "to": "DPT",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "DPT",
    "to": "Dense Prediction Transformer",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "3D Vision",
    "to": "3D Object Detection",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "3D Object Detection",
    "to": "LiDAR-Based Detection",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "LiDAR-Based Detection",
    "to": "PointPillar",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "LiDAR-Based Detection",
    "to": "VoxelNet",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "3D Vision",
    "to": "NeRF",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "NeRF",
    "to": "Implicit Neural Representation",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Implicit Neural Representation",
    "to": "Volume Rendering",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Volume Rendering",
    "to": "Ray Marching",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "NeRF",
    "to": "3D Gaussian Splatting",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "3D Gaussian Splatting",
    "to": "Real-Time 3D Rendering",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Video",
    "to": "Temporal Modeling",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Temporal Modeling",
    "to": "3D CNNs",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "3D CNNs",
    "to": "C3D",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "C3D",
    "to": "Temporal Convolutions",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Temporal Modeling",
    "to": "Two-Stream Networks",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Two-Stream Networks",
    "to": "RGB Stream",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Two-Stream Networks",
    "to": "Optical Flow Stream",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Temporal Modeling",
    "to": "I3D",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "I3D",
    "to": "Inflated 3D Convolutions",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Temporal Modeling",
    "to": "Video Transformers",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Video Transformers",
    "to": "TimeSformer",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Video Transformers",
    "to": "VideoMAE",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "VideoMAE",
    "to": "Masked Video Modeling",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Video",
    "to": "Action Recognition",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Action Recognition",
    "to": "Kinetics Dataset",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Kinetics Dataset",
    "to": "Pre-trained Video Models",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Video",
    "to": "Optical Flow (Deep)",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "Optical Flow (Deep)",
    "to": "FlowNet",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "FlowNet",
    "to": "PWC-Net",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "PWC-Net",
    "to": "Real-Time Optical Flow",
    "domain": "COMPUTER VISION",
    "section": "3D & Video"
  },
  {
    "from": "SVD",
    "to": "Word Embeddings",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "SVD",
    "to": "Collaborative Filtering",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "SVD",
    "to": "Matrix Factorization (RecSys)",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Eigenvalues",
    "to": "PCA",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "PCA",
    "to": "Whitening",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Whitening",
    "to": "ICA",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Cosine Similarity",
    "to": "Semantic Search",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Cosine Similarity",
    "to": "Recommendation Similarity",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Gradient Descent",
    "to": "Neural Network Training",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Adam",
    "to": "BERT Training",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Adam",
    "to": "GPT Training",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Adam",
    "to": "ViT Training",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Cross-Entropy Loss",
    "to": "Language Model Training",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Cross-Entropy Loss",
    "to": "Image Classification Training",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "KL Divergence",
    "to": "VAE Loss",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "KL Divergence",
    "to": "RLHF KL Penalty",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "KL Divergence",
    "to": "Knowledge Distillation Loss",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Mutual Information",
    "to": "CLIP Loss",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Mutual Information",
    "to": "Contrastive Learning",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Contrastive Learning",
    "to": "SimCLR",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "SimCLR",
    "to": "MoCo",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "MoCo",
    "to": "DINO (Self-supervised)",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Contrastive Learning",
    "to": "CLIP",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Contrastive Learning",
    "to": "SimCSE",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "SimCSE",
    "to": "NLP Sentence Embeddings",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Backpropagation",
    "to": "Training All Models",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Residual Connections",
    "to": "ResNet",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Residual Connections",
    "to": "Transformer Sublayers",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Residual Connections",
    "to": "Highway Networks",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Attention Mechanism",
    "to": "NLP",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Attention Mechanism",
    "to": "Vision",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Attention Mechanism",
    "to": "Multimodal",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Attention Mechanism",
    "to": "Code",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Transformers",
    "to": "NLP",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Transformers",
    "to": "Vision",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Transformers",
    "to": "RL (Decision Transformer)",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Transformers",
    "to": "Speech",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Transformers",
    "to": "Multimodal",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Transfer Learning",
    "to": "NLP",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Transfer Learning",
    "to": "Vision",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Transfer Learning",
    "to": "Multimodal",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Transfer Learning",
    "to": "Time Series",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "BERT",
    "to": "Semantic Search (Cross-domain)",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Sentence-BERT",
    "to": "RAG Retriever",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Dense Retrieval",
    "to": "RAG",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "RAG",
    "to": "LLM Agents",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "LLM Agents",
    "to": "RL (Framing)",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "PPO",
    "to": "RLHF",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "RLHF",
    "to": "InstructGPT",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "RLHF",
    "to": "Constitutional AI",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "RL",
    "to": "Robotics",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Robotics",
    "to": "Sim-to-Real",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Sim-to-Real",
    "to": "Domain Randomization",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Domain Randomization",
    "to": "Robust Policies",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Object Detection",
    "to": "Autonomous Driving",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Autonomous Driving",
    "to": "End-to-End RL",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "End-to-End RL",
    "to": "Waypoint Prediction",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Computer Vision",
    "to": "Multimodal LLMs",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Multimodal LLMs",
    "to": "Multimodal Agents",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Multimodal Agents",
    "to": "Visual Tool Use",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Visual Tool Use",
    "to": "Screenshot Agents",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Screenshot Agents",
    "to": "Computer Use",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Knowledge Graphs",
    "to": "RAG (Structured)",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Knowledge Graphs",
    "to": "Entity Linking",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Entity Linking",
    "to": "Named Entity Recognition",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Named Entity Recognition",
    "to": "Information Extraction",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Information Extraction",
    "to": "Knowledge Base Population",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Knowledge Base Population",
    "to": "Open-Domain QA",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Open-Domain QA",
    "to": "RAG",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Graph Neural Networks",
    "to": "Recommendation Systems",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Graph Neural Networks",
    "to": "Drug Discovery",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Graph Neural Networks",
    "to": "Fraud Detection",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "GCN",
    "to": "PinSage",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "PinSage",
    "to": "Pinterest Recommendation",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "GNN",
    "to": "Molecular Property Prediction",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Molecular Property Prediction",
    "to": "Drug-Target Interaction",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Model Compression",
    "to": "Pruning",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Model Compression",
    "to": "Quantization",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Model Compression",
    "to": "Knowledge Distillation",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Knowledge Distillation",
    "to": "DistilBERT",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Knowledge Distillation",
    "to": "TinyBERT",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Knowledge Distillation",
    "to": "TinyCLIP",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Quantization",
    "to": "INT8",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Quantization",
    "to": "4-bit LLM",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Pruning",
    "to": "Sparse Neural Networks",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Sparse Neural Networks",
    "to": "Efficient Inference",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Flash Attention",
    "to": "Training Efficiency",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Flash Attention",
    "to": "Long Context",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Long Context",
    "to": "Retrieval Reduction",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Long Context",
    "to": "Document Understanding",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "LLM",
    "to": "Code Generation",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Code Generation",
    "to": "Code LLMs",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Code LLMs",
    "to": "Unit Test Generation",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Code LLMs",
    "to": "Bug Detection",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Code LLMs",
    "to": "Code Completion",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Instruction Following",
    "to": "Task Generalization",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Task Generalization",
    "to": "Zero-Shot Generalization",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Zero-Shot Generalization",
    "to": "Foundation Models",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Foundation Models",
    "to": "Adapting to Downstream",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Adapting to Downstream",
    "to": "Fine-tuning",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Adapting to Downstream",
    "to": "In-Context Learning",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Adapting to Downstream",
    "to": "Prompting",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Self-supervised Learning",
    "to": "SSL in NLP",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Self-supervised Learning",
    "to": "SSL in Vision",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "SSL in NLP",
    "to": "BERT",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "SSL in Vision",
    "to": "MAE",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "SSL in Vision",
    "to": "DINO",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "SSL in Vision",
    "to": "SimCLR",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Contrastive Self-supervised",
    "to": "MoCo",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Contrastive Self-supervised",
    "to": "SimCLR",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Contrastive Self-supervised",
    "to": "BYOL",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "BYOL",
    "to": "Bootstrap Your Own Latent",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "BYOL",
    "to": "No Negative Samples",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "BYOL",
    "to": "DINO (Inspired)",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "ViT",
    "to": "Multimodal (Image Side)",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "BERT",
    "to": "Multimodal (Text Side)",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "CLIP",
    "to": "Aligning Modalities",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Aligning Modalities",
    "to": "DALL-E",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Aligning Modalities",
    "to": "Flamingo",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Flamingo",
    "to": "In-Context Multimodal",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "In-Context Multimodal",
    "to": "OpenFlamingo",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "In-Context Multimodal",
    "to": "IDEFICS",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Diffusion Models",
    "to": "Creative Tools",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Creative Tools",
    "to": "Midjourney",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Creative Tools",
    "to": "Adobe Firefly",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "GANs",
    "to": "Synthetic Data Generation",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Synthetic Data Generation",
    "to": "Data Augmentation (Synthetic)",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Data Augmentation (Synthetic)",
    "to": "Low-Resource Learning",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "RL",
    "to": "Game Playing",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Game Playing",
    "to": "AlphaGo",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Game Playing",
    "to": "AlphaZero",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Game Playing",
    "to": "OpenAI Five",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "OpenAI Five",
    "to": "PPO (at Scale)",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "RL",
    "to": "Protein Folding",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Protein Folding",
    "to": "AlphaFold",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "AlphaFold",
    "to": "Structure Prediction",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Structure Prediction",
    "to": "MSA Transformer",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "MSA Transformer",
    "to": "Evolutionary Information",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "NLP",
    "to": "Code Understanding",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Code Understanding",
    "to": "CodeBERT",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "CodeBERT",
    "to": "Code Search",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Code Search",
    "to": "Developer Tools",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "LLM",
    "to": "Text-to-SQL",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Text-to-SQL",
    "to": "Natural Language Interfaces",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Natural Language Interfaces",
    "to": "Business Intelligence",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "LLM",
    "to": "Mathematical Reasoning",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Mathematical Reasoning",
    "to": "Chain-of-Thought Math",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Chain-of-Thought Math",
    "to": "Program Synthesis",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  },
  {
    "from": "Program Synthesis",
    "to": "Verified Code",
    "domain": "CROSS-DOMAIN CONNECTIONS",
    "section": ""
  }
];
