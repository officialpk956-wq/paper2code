"""
backend/scripts/generate_new_dojo_seeds.py

Fills the gap where 39 dojo problems exist in src/data/problems.ts (frontend)
but have no matching row in backend/scripts/json_dump/dojo_problems_seed.json
(the file problem_seed_service.py actually loads at startup) — so Run/Submit
has no judge harness to grade against for any of them.

For each new slug: a correct reference solution + an assert-based harness
(same "type": "harness" format as the existing 10 entries). Every harness is
self-verified here (reference + harness executed together, must print
'ALL TESTS PASSED') before being written out — a broken harness never reaches
the seed file.

Run:
  npx tsx backend/scripts/export_problems_to_json.mts   (regenerates the metadata cache)
  python backend/scripts/generate_new_dojo_seeds.py
Then redeploy — problem_seed_service.py picks up new rows on next startup.
"""

import json
import os
import subprocess
import sys

METADATA_PATH = os.path.join(os.path.dirname(__file__), "json_dump", "problems_from_ts.json")
SEED_PATH = os.path.join(os.path.dirname(__file__), "json_dump", "dojo_problems_seed.json")

DIFFICULTY_TIME = {"Easy": 10, "Medium": 15, "Hard": 25}

# slug -> (reference_solution_source, harness_code)
# Reference solutions are correct implementations used ONLY to generate and
# verify the harness locally — they are never shipped to the seed file.
PROBLEMS: dict[str, tuple[str, str]] = {}


def add(slug, ref, harness):
    PROBLEMS[slug] = (ref, harness)


add(
    "stats-mean-var-std",
    """import numpy as np
def stats_basics(x):
    n = len(x)
    mean = np.sum(x) / n
    var = np.sum((x - mean) ** 2) / n
    return (mean, var, float(np.sqrt(var)))
""",
    """import numpy as np
m, v, s = stats_basics(np.array([2, 4, 4, 4, 5, 5, 7, 9]))
assert abs(m - 5.0) < 1e-6, f'mean wrong: {m}'
assert abs(v - 4.0) < 1e-6, f'variance wrong (must use N not N-1): {v}'
assert abs(s - 2.0) < 1e-6, f'std wrong: {s}'
m2, v2, s2 = stats_basics(np.array([10.0]))
assert v2 == 0.0 and s2 == 0.0, 'single-element array must have 0 variance/std'
print('ALL TESTS PASSED')
""",
)

add(
    "stats-covariance-correlation",
    """import numpy as np
def cov_corr(x, y):
    mx, my = np.mean(x), np.mean(y)
    cov = np.sum((x - mx) * (y - my)) / len(x)
    corr = cov / (np.std(x) * np.std(y))
    return (cov, corr)
""",
    """import numpy as np
cov, corr = cov_corr(np.array([1, 2, 3]), np.array([3, 1, 2]))
assert abs(cov - (-1/3)) < 1e-4, f'covariance wrong: {cov}'
assert abs(corr - (-0.5)) < 1e-4, f'correlation wrong: {corr}'
cov2, corr2 = cov_corr(np.array([1, 2, 3, 4]), np.array([2, 4, 6, 8]))
assert abs(corr2 - 1.0) < 1e-4, f'perfectly correlated data must give corr=1: {corr2}'
print('ALL TESTS PASSED')
""",
)

add(
    "stats-z-score",
    """import numpy as np
def z_score(x):
    std = np.std(x)
    if std == 0:
        return np.zeros_like(x, dtype=float)
    return (x - np.mean(x)) / std
""",
    """import numpy as np
z = z_score(np.array([1, 2, 3]))
assert np.allclose(z, [-1.22474487, 0.0, 1.22474487], atol=1e-4), f'wrong z-scores: {z}'
assert abs(float(np.mean(z))) < 1e-9, 'z-scores must have mean 0'
z2 = z_score(np.array([5.0, 5.0, 5.0]))
assert np.allclose(z2, [0, 0, 0]), 'constant input must not divide by zero'
print('ALL TESTS PASSED')
""",
)

add(
    "stats-normal-pdf",
    """import numpy as np
def normal_pdf(x, mu=0.0, sigma=1.0):
    coef = 1.0 / (sigma * np.sqrt(2 * np.pi))
    return coef * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
""",
    """import numpy as np
p = normal_pdf(0.0, 0.0, 1.0)
assert abs(float(p) - 0.3989) < 1e-3, f'standard normal at 0 wrong: {p}'
arr = normal_pdf(np.array([-1, 0, 1]), 0, 1)
assert np.allclose(arr, [0.24197, 0.39894, 0.24197], atol=1e-3), f'wrong pdf array: {arr}'
print('ALL TESTS PASSED')
""",
)

add(
    "stats-bayes-theorem",
    """def bayes(p_a, p_b_given_a, p_b_given_not_a):
    p_not_a = 1 - p_a
    p_b = p_b_given_a * p_a + p_b_given_not_a * p_not_a
    return (p_b_given_a * p_a) / p_b
""",
    """r = bayes(0.01, 0.9, 0.05)
assert abs(r - 0.1538) < 1e-3, f'wrong posterior: {r}'
r2 = bayes(0.5, 1.0, 0.0)
assert abs(r2 - 1.0) < 1e-6, 'a perfect test on a 50 percent prior must give posterior 1.0'
print('ALL TESTS PASSED')
""",
)

add(
    "stats-bootstrap-ci",
    """import numpy as np
def bootstrap_ci(x, n_bootstraps=1000):
    np.random.seed(42)
    means = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        sample = np.random.choice(x, size=len(x), replace=True)
        means[i] = np.mean(sample)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (lo, hi)
""",
    """import numpy as np
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
lo, hi = bootstrap_ci(x, n_bootstraps=1000)
true_mean = np.mean(x)
assert lo < true_mean < hi, f'true mean {true_mean} must fall inside CI ({lo}, {hi})'
assert lo < hi, 'lower bound must be < upper bound'
assert x.min() <= lo and hi <= x.max(), 'CI must stay within the range of resampled data'
print('ALL TESTS PASSED')
""",
)

add(
    "stats-t-test",
    """import numpy as np
def t_statistic(x1, x2):
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / (sp * np.sqrt(1 / n1 + 1 / n2))
""",
    """import numpy as np
t = t_statistic(np.array([1, 2, 3]), np.array([4, 5, 6]))
assert abs(t - (-3.6742)) < 1e-3, f'wrong t-statistic: {t}'
t2 = t_statistic(np.array([1, 2, 3]), np.array([1, 2, 3]))
assert abs(t2) < 1e-9, 'identical samples must give t=0'
print('ALL TESTS PASSED')
""",
)

add(
    "stats-mle-gaussian",
    """import numpy as np
def mle_gaussian(x):
    return (float(np.mean(x)), float(np.var(x)))
""",
    """import numpy as np
mu, var = mle_gaussian(np.array([2, 4, 6, 8]))
assert abs(mu - 5.0) < 1e-6, f'mu wrong: {mu}'
assert abs(var - 5.0) < 1e-6, f'var wrong (must be population variance): {var}'
print('ALL TESTS PASSED')
""",
)

add(
    "stats-kl-divergence",
    """import numpy as np
def entropy_kl(P, Q):
    Pc = np.clip(P, 1e-9, 1.0)
    Qc = np.clip(Q, 1e-9, 1.0)
    entropy = -np.sum(P * np.log2(Pc))
    kl = np.sum(P * np.log2(Pc / Qc))
    return (float(entropy), float(kl))
""",
    """import numpy as np
h, kl = entropy_kl(np.array([0.5, 0.5]), np.array([0.1, 0.9]))
assert abs(h - 1.0) < 1e-3, f'entropy of a fair coin must be 1 bit: {h}'
assert abs(kl - 0.7369) < 1e-3, f'wrong KL divergence: {kl}'
h2, kl2 = entropy_kl(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
assert abs(kl2) < 1e-6, 'KL divergence of identical distributions must be 0'
print('ALL TESTS PASSED')
""",
)

add(
    "stats-log-sum-exp",
    """import numpy as np
def log_sum_exp(x):
    a = np.max(x)
    return float(a + np.log(np.sum(np.exp(x - a))))
""",
    """import numpy as np
r = log_sum_exp(np.array([1, 2, 3]))
assert abs(r - 3.4076) < 1e-3, f'wrong value: {r}'
r2 = log_sum_exp(np.array([1000.0, 1000.0]))
assert np.isfinite(r2), 'must not overflow on large inputs'
assert abs(r2 - (1000 + np.log(2))) < 1e-3, f'wrong large-input value: {r2}'
print('ALL TESTS PASSED')
""",
)

add(
    "ml-train-test-split",
    """import numpy as np
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    idx = np.random.permutation(len(X))
    n_test = int(len(X) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
""",
    """import numpy as np
X = np.arange(10)
y = np.arange(10) * 2
X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2, 42)
assert len(X_test) == 2, f'expected 2 test samples (20% of 10): got {len(X_test)}'
assert len(X_train) == 8, f'expected 8 train samples: got {len(X_train)}'
all_idx = np.concatenate([X_train, X_test])
assert set(all_idx.tolist()) == set(range(10)), 'train+test must cover every index exactly once'
assert len(set(X_train.tolist()) & set(X_test.tolist())) == 0, 'train and test must not overlap'
assert np.array_equal(y_test, X_test * 2), 'y_test must correspond to the same rows as X_test'
print('ALL TESTS PASSED')
""",
)

add(
    "ml-linear-regression-normal",
    """import numpy as np
def lin_reg_normal(X, y):
    Xb = np.c_[np.ones(len(X)), X]
    return np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y
""",
    """import numpy as np
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])
theta = lin_reg_normal(X, y)
assert np.allclose(theta, [0.0, 2.0], atol=1e-3), f'wrong weights for y=2x: {theta}'
print('ALL TESTS PASSED')
""",
)

add(
    "ml-linear-regression-gd",
    """import numpy as np
def lin_reg_gd(X, y, lr=0.01, epochs=1000):
    Xb = np.c_[np.ones(len(X)), X]
    theta = np.zeros(Xb.shape[1])
    n = len(y)
    for _ in range(epochs):
        pred = Xb @ theta
        grad = (2.0 / n) * Xb.T @ (pred - y)
        theta = theta - lr * grad
    return theta
""",
    """import numpy as np
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])
theta = lin_reg_gd(X, y, 0.01, 1000)
assert np.allclose(theta, [0.0, 2.0], atol=0.1), f'gradient descent did not converge near [0, 2]: {theta}'
print('ALL TESTS PASSED')
""",
)

add(
    "ml-logistic-regression-gd",
    """import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def log_reg_gd(X, y, lr=0.1, epochs=1000):
    Xb = np.c_[np.ones(len(X)), X]
    theta = np.zeros(Xb.shape[1])
    n = len(y)
    for _ in range(epochs):
        pred = sigmoid(Xb @ theta)
        grad = (1.0 / n) * Xb.T @ (pred - y)
        theta = theta - lr * grad
    return theta
""",
    """import numpy as np
X = np.array([[0.1], [0.5], [2.0], [3.0]])
y = np.array([0, 0, 1, 1])
theta = log_reg_gd(X, y, 0.5, 1000)
def sigmoid(z): return 1 / (1 + np.exp(-z))
Xb = np.c_[np.ones(len(X)), X]
preds = sigmoid(Xb @ theta)
assert np.all((preds > 0.5) == (y == 1)), f'trained classifier must separate the classes: {preds}'
print('ALL TESTS PASSED')
""",
)

add(
    "ml-classification-metrics",
    """import numpy as np
def calc_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return (acc, prec, rec, f1)
""",
    """import numpy as np
y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 0, 1, 1, 1])
acc, prec, rec, f1 = calc_metrics(y_true, y_pred)
assert abs(acc - 0.6667) < 1e-3, f'accuracy wrong: {acc}'
assert abs(prec - 0.75) < 1e-3, f'precision wrong: {prec}'
assert abs(rec - 0.75) < 1e-3, f'recall wrong: {rec}'
assert abs(f1 - 0.75) < 1e-3, f'f1 wrong: {f1}'
z_acc, z_prec, z_rec, z_f1 = calc_metrics(np.array([0, 0]), np.array([0, 0]))
assert z_prec == 0.0 and z_rec == 0.0 and z_f1 == 0.0, 'no positive predictions must not divide by zero'
print('ALL TESTS PASSED')
""",
)

add(
    "ml-roc-auc",
    """import numpy as np
def roc_auc(y_true, y_prob):
    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp, fp, auc, prev_fp = 0, 0, 0.0, 0
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    return auc / (n_pos * n_neg)
""",
    """import numpy as np
r = roc_auc(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.4]))
assert abs(r - 1.0) < 1e-6, f'perfectly separable classes must give AUC=1: {r}'
r2 = roc_auc(np.array([0, 1, 0, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
assert abs(r2 - 1.0) < 1e-6, f'wrong AUC (both positives outrank both negatives here too): {r2}'
r3 = roc_auc(np.array([1, 0]), np.array([0.3, 0.8]))
assert abs(r3 - 0.0) < 1e-6, f'a completely inverted ranking must give AUC=0: {r3}'
print('ALL TESTS PASSED')
""",
)

add(
    "ml-k-fold-cv",
    """import numpy as np
def k_fold(N, k):
    fold_sizes = np.full(k, N // k, dtype=int)
    fold_sizes[: N % k] += 1
    folds, current = [], 0
    for size in fold_sizes:
        test_idx = np.arange(current, current + size)
        train_idx = np.concatenate([np.arange(0, current), np.arange(current + size, N)])
        folds.append((train_idx, test_idx))
        current += size
    return folds
""",
    """folds = k_fold(5, 2)
assert len(folds) == 2, f'expected 2 folds: got {len(folds)}'
all_test = []
for train_idx, test_idx in folds:
    assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0, 'train/test must not overlap within a fold'
    all_test.extend(test_idx.tolist())
assert sorted(all_test) == list(range(5)), f'every index must appear in exactly one test fold: {sorted(all_test)}'
print('ALL TESTS PASSED')
""",
)

add(
    "ml-knn-classifier",
    """import numpy as np
def knn(X, y, x_test, k=3):
    dists = np.linalg.norm(X - x_test, axis=1)
    nearest = np.argsort(dists)[:k]
    labels = y[nearest]
    return int(np.bincount(labels).argmax())
""",
    """import numpy as np
X = np.array([[1,1], [1.2,1.2], [5,5], [5.1,5.2]])
y = np.array([0, 0, 1, 1])
r = knn(X, y, np.array([1.1, 1.1]), k=3)
assert r == 0, f'point near cluster 0 must be classified 0: got {r}'
r2 = knn(X, y, np.array([5.05, 5.1]), k=3)
assert r2 == 1, f'point near cluster 1 must be classified 1: got {r2}'
print('ALL TESTS PASSED')
""",
)

add(
    "ml-k-means",
    """import numpy as np
def kmeans(X, k, max_iters=100, random_state=42):
    np.random.seed(random_state)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iters):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids, labels
""",
    """import numpy as np
X = np.array([[1,1], [1.2,1.2], [5,5], [5.2,5.1]])
centroids, labels = kmeans(X, k=2)
assert len(set(labels.tolist())) == 2, f'must produce 2 distinct clusters: {labels}'
assert labels[0] == labels[1], 'the two nearby points must be in the same cluster'
assert labels[2] == labels[3], 'the other two nearby points must be in the same cluster'
assert labels[0] != labels[2], 'the two well-separated groups must be in different clusters'
assert centroids.shape == (2, 2), f'wrong centroids shape: {centroids.shape}'
print('ALL TESTS PASSED')
""",
)

add(
    "ml-pca",
    """import numpy as np
def pca(X, k):
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    top = vecs[:, order[:k]]
    return Xc @ top
""",
    """import numpy as np
X = np.array([[1, 2], [3, 4], [5, 6]])
proj = pca(X, k=1)
assert proj.shape == (3, 1), f'wrong output shape: {proj.shape}'
absvals = np.abs(np.round(proj, 4)).flatten()
assert np.allclose(sorted(absvals), sorted([2.8284, 0.0, 2.8284]), atol=1e-3), f'wrong projected magnitudes: {absvals}'
print('ALL TESTS PASSED')
""",
)

add(
    "ml-decision-tree-gini",
    """import numpy as np
def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.mean(y)
    return 1.0 - (p**2 + (1 - p) ** 2)
def best_split(X, y):
    n = len(y)
    base_cost = None
    best_feat, best_thresh, best_cost = None, None, np.inf
    for feat in range(X.shape[1]):
        for thresh in np.unique(X[:, feat]):
            left_mask = X[:, feat] <= thresh
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            cost = (left_mask.sum() / n) * gini(y[left_mask]) + (right_mask.sum() / n) * gini(y[right_mask])
            if cost < best_cost:
                best_cost, best_feat, best_thresh = cost, feat, thresh
    return (best_feat, best_thresh)
""",
    """import numpy as np
X = np.array([[1, 2], [2, 1], [4, 5], [5, 4]])
y = np.array([0, 0, 1, 1])
feat, thresh = best_split(X, y)
left_mask = X[:, feat] <= thresh
assert set(y[left_mask].tolist()) == {0}, f'best split must perfectly separate class 0: feat={feat} thresh={thresh}'
assert set(y[~left_mask].tolist()) == {1}, f'best split must perfectly separate class 1: feat={feat} thresh={thresh}'
print('ALL TESTS PASSED')
""",
)

add(
    "ml-gaussian-naive-bayes",
    """import numpy as np
def gnb_predict(x_test, priors, means, vars):
    best_c, best_score = None, -np.inf
    for c in priors:
        log_p = np.log(priors[c])
        log_lik = -0.5 * np.log(2 * np.pi * vars[c]) - ((x_test - means[c]) ** 2) / (2 * vars[c])
        score = log_p + np.sum(log_lik)
        if score > best_score:
            best_score, best_c = score, c
    return best_c
""",
    """import numpy as np
priors = {0: 0.5, 1: 0.5}
means = {0: np.array([0, 0]), 1: np.array([5, 5])}
vars = {0: np.array([1, 1]), 1: np.array([1, 1])}
r = gnb_predict(np.array([4, 4]), priors, means, vars)
assert r == 1, f'point near class-1 mean must predict 1: got {r}'
r2 = gnb_predict(np.array([0.1, -0.1]), priors, means, vars)
assert r2 == 0, f'point near class-0 mean must predict 0: got {r2}'
print('ALL TESTS PASSED')
""",
)

add(
    "dl-backprop-single-neuron",
    """import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def backprop_neuron(w, b, x, y):
    z = w * x + b
    a = sigmoid(z)
    dL_da = 2 * (a - y)
    da_dz = a * (1 - a)
    grad_w = dL_da * da_dz * x
    grad_b = dL_da * da_dz
    return (grad_w, grad_b)
""",
    """gw, gb = backprop_neuron(0.5, 0.1, 2.0, 1.0)
assert abs(gw - (-0.1872)) < 1e-3, f'grad_w wrong: {gw}'
assert abs(gb - (-0.0936)) < 1e-3, f'grad_b wrong: {gb}'
assert abs(gw - 2 * gb) < 1e-6, 'grad_w must equal x * grad_b since dz/dw = x = 2'
print('ALL TESTS PASSED')
""",
)

add(
    "dl-2-layer-mlp",
    """import numpy as np
def mlp_pass(X, y, W1, b1, W2, b2):
    N = X.shape[0]
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2
    y_hat = 1 / (1 + np.exp(-Z2))
    dZ2 = (y_hat - y) / N
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
""",
    """import numpy as np
np.random.seed(42)
X = np.random.randn(2, 3)
y = np.array([[1], [0]])
W1 = np.random.randn(3, 4)
b1 = np.zeros(4)
W2 = np.random.randn(4, 1)
b2 = np.zeros(1)
grads = mlp_pass(X, y, W1, b1, W2, b2)
for key in ("dW1", "db1", "dW2", "db2"):
    assert key in grads, f'missing gradient key: {key}'
assert grads["dW1"].shape == (3, 4), f'dW1 wrong shape: {grads["dW1"].shape}'
assert grads["dW2"].shape == (4, 1), f'dW2 wrong shape: {grads["dW2"].shape}'
assert np.allclose(grads["dW2"].flatten(), [0.0107, -0.0180, 0.0, -0.2058], atol=1e-3), f'dW2 values wrong: {grads["dW2"].flatten()}'
print('ALL TESTS PASSED')
""",
)

add(
    "dl-batch-norm",
    """import numpy as np
def batch_norm(X, gamma, beta, eps=1e-5):
    mu = X.mean(axis=0, keepdims=True)
    var = X.var(axis=0, keepdims=True)
    Xhat = (X - mu) / np.sqrt(var + eps)
    return gamma * Xhat + beta
""",
    """import numpy as np
X = np.array([[1.0, 2.0], [3.0, 4.0]])
gamma = np.array([1.0, 1.0])
beta = np.array([0.0, 0.0])
out = batch_norm(X, gamma, beta)
assert np.allclose(np.mean(out, axis=0), [0, 0], atol=1e-3), 'output must have ~zero mean per feature'
assert out.shape == X.shape
assert np.allclose(out, [[-1.0, -1.0], [1.0, 1.0]], atol=1e-2), f'wrong normalized values: {out}'
print('ALL TESTS PASSED')
""",
)

add(
    "dl-layer-norm",
    """import numpy as np
def layer_norm(X, gamma, beta, eps=1e-5):
    mu = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    Xhat = (X - mu) / np.sqrt(var + eps)
    return gamma * Xhat + beta
""",
    """import numpy as np
X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
gamma = np.ones(3)
beta = np.zeros(3)
out = layer_norm(X, gamma, beta)
assert np.allclose(np.mean(out, axis=-1), [0, 0], atol=1e-3), 'each row must have ~zero mean'
assert np.allclose(out[0], out[1], atol=1e-3), 'both rows are arithmetic sequences so must normalize identically'
print('ALL TESTS PASSED')
""",
)

add(
    "dl-dropout",
    """import numpy as np
def dropout(X, p, is_training=True):
    np.random.seed(42)
    if not is_training:
        return X
    mask = np.random.rand(*X.shape) > p
    return (X * mask) / (1.0 - p)
""",
    """import numpy as np
out = dropout(np.ones(10), p=0.5, is_training=True)
vals = set(np.round(out, 4).tolist())
assert vals <= {0.0, 2.0}, f'with p=0.5 on ones, output must only contain 0 or 2: {vals}'
assert 0.0 in vals and 2.0 in vals, 'expect a mix of zeroed and scaled elements'
out2 = dropout(np.ones(10), p=0.5, is_training=False)
assert np.array_equal(out2, np.ones(10)), 'eval mode must return input unchanged'
print('ALL TESTS PASSED')
""",
)

add(
    "dl-conv2d",
    """import numpy as np
def conv2d(image, kernel):
    H, W = image.shape
    F = kernel.shape[0]
    out = np.zeros((H - F + 1, W - F + 1))
    for i in range(H - F + 1):
        for j in range(W - F + 1):
            patch = image[i:i+F, j:j+F]
            out[i, j] = np.sum(patch * kernel)
    return out
""",
    """import numpy as np
img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
kernel = np.array([[1, 0], [0, -1]])
out = conv2d(img, kernel)
assert out.shape == (2, 2), f'wrong output shape: {out.shape}'
assert np.array_equal(out, [[-4, -4], [-4, -4]]), f'wrong conv output: {out}'
print('ALL TESTS PASSED')
""",
)

add(
    "dl-max-pooling",
    """import numpy as np
def max_pool2d(image, pool_size):
    H, W = image.shape
    out_H, out_W = H // pool_size, W // pool_size
    out = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            patch = image[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
            out[i, j] = np.max(patch)
    return out
""",
    """import numpy as np
img = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
out = max_pool2d(img, 2)
assert np.array_equal(out, [[6, 8], [14, 16]]), f'wrong pooled output: {out}'
print('ALL TESTS PASSED')
""",
)

add(
    "dl-sgd-momentum",
    """import numpy as np
def sgd_momentum(params, grads, velocity, lr=0.1, beta=0.9):
    v_new = beta * velocity + (1 - beta) * grads
    p_new = params - lr * v_new
    return (p_new, v_new)
""",
    """import numpy as np
w = np.array([1.0, 2.0])
dw = np.array([0.5, -0.2])
v = np.array([0.1, -0.1])
w_new, v_new = sgd_momentum(w, dw, v, 0.1, 0.9)
assert np.allclose(w_new, [0.986, 2.011], atol=1e-3), f'wrong updated params: {w_new}'
assert np.allclose(v_new, [0.14, -0.11], atol=1e-3), f'wrong updated velocity: {v_new}'
print('ALL TESTS PASSED')
""",
)

add(
    "dl-adam-optimizer",
    """import numpy as np
def adam_step(params, grads, m, v, t, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    m_new = b1 * m + (1 - b1) * grads
    v_new = b2 * v + (1 - b2) * (grads ** 2)
    m_hat = m_new / (1 - b1 ** t)
    v_hat = v_new / (1 - b2 ** t)
    p_new = params - lr * m_hat / (np.sqrt(v_hat) + eps)
    return (p_new, m_new, v_new)
""",
    """import numpy as np
w = np.array([1.0, 2.0])
dw = np.array([0.5, -0.2])
m = np.zeros(2)
v = np.zeros(2)
w_new, m_new, v_new = adam_step(w, dw, m, v, t=1)
assert np.allclose(w_new, [0.9990, 2.0010], atol=1e-3), f'wrong updated params: {w_new}'
assert np.allclose(m_new, [0.05, -0.02], atol=1e-4), f'wrong first moment: {m_new}'
print('ALL TESTS PASSED')
""",
)

add(
    "dl-embedding-lookup",
    """import numpy as np
def embedding_layer(indices, embeddings, dOut):
    output = embeddings[indices]
    dE = np.zeros_like(embeddings, dtype=float)
    np.add.at(dE, indices, dOut)
    return (output, dE)
""",
    """import numpy as np
emb = np.array([[0,0], [1,1], [2,2], [3,3]])
idx = np.array([1, 3, 1])
dOut = np.array([[0.1, 0.1], [0.3, 0.3], [0.5, 0.5]])
out, dE = embedding_layer(idx, emb, dOut)
assert np.array_equal(out, [[1,1],[3,3],[1,1]]), f'wrong forward lookup: {out}'
assert np.allclose(dE[1], [0.6, 0.6], atol=1e-6), f'repeated index 1 must accumulate gradients: {dE[1]}'
assert np.allclose(dE[3], [0.3, 0.3], atol=1e-6), f'index 3 gradient wrong: {dE[3]}'
assert np.allclose(dE[0], [0, 0]) and np.allclose(dE[2], [0, 0]), 'unused rows must have zero gradient'
print('ALL TESTS PASSED')
""",
)

add(
    "dl-multi-head-attention",
    """import numpy as np
def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)
def mha(X, W_q, W_k, W_v, W_o, h):
    N, d_model = X.shape
    d_head = d_model // h
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    Q = Q.reshape(N, h, d_head).transpose(1, 0, 2)
    K = K.reshape(N, h, d_head).transpose(1, 0, 2)
    V = V.reshape(N, h, d_head).transpose(1, 0, 2)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_head)
    weights = softmax(scores, axis=-1)
    out = weights @ V
    out = out.transpose(1, 0, 2).reshape(N, d_model)
    return out @ W_o
""",
    """import numpy as np
np.random.seed(42)
X = np.random.randn(3, 4)
Wq, Wk, Wv, Wo = np.random.randn(4,4), np.random.randn(4,4), np.random.randn(4,4), np.random.randn(4,4)
out = mha(X, Wq, Wk, Wv, Wo, 2)
assert out.shape == (3, 4), f'wrong output shape: {out.shape}'
assert np.all(np.isfinite(out)), 'output must not contain nan/inf'
print('ALL TESTS PASSED')
""",
)

add(
    "dl-transformer-encoder-block",
    """import numpy as np
def transformer_block(X, W1, b1, W2, b2, mha_fn, ln_fn):
    out1 = mha_fn(X)
    X1 = ln_fn(X + out1)
    ffn_out = np.maximum(0, X1 @ W1 + b1) @ W2 + b2
    out2 = ln_fn(X1 + ffn_out)
    return out2
""",
    """import numpy as np
np.random.seed(42)
X = np.random.randn(2, 4)
W1, b1 = np.random.randn(4, 8), np.random.randn(8)
W2, b2 = np.random.randn(8, 4), np.random.randn(4)
mha_fn = lambda x: x * 0.1
ln_fn = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-5)
out = transformer_block(X, W1, b1, W2, b2, mha_fn, ln_fn)
assert out.shape == (2, 4), f'wrong output shape: {out.shape}'
assert np.all(np.isfinite(out)), 'output must not contain nan/inf'
assert np.allclose(np.mean(out, axis=-1), [0, 0], atol=1e-3), 'post-LN output rows must be ~zero-mean'
print('ALL TESTS PASSED')
""",
)

add(
    "dl-resnet-block",
    """import numpy as np
def resnet_block(X, W1, W2, conv2d_fn, bn_fn):
    out1 = bn_fn(conv2d_fn(X, W1))
    out1 = np.maximum(0, out1)
    out2 = bn_fn(conv2d_fn(out1, W2))
    return np.maximum(0, out2 + X)
""",
    """import numpy as np
np.random.seed(42)
X = np.random.randn(3, 3)
W1, W2 = np.random.randn(2, 2), np.random.randn(2, 2)
conv2d_fn = lambda x, w: x * np.sum(w)
bn_fn = lambda x: x * 0.5
out = resnet_block(X, W1, W2, conv2d_fn, bn_fn)
assert out.shape == (3, 3), f'wrong output shape: {out.shape}'
assert np.all(out >= 0), 'final ReLU must make all outputs non-negative'
print('ALL TESTS PASSED')
""",
)

add(
    "dl-bert-mlm-loss",
    """import numpy as np
def bert_mlm_loss(logits, targets, mask_indices):
    mask = mask_indices == 1
    if not np.any(mask):
        return 0.0
    l = logits[mask]
    t = targets[mask]
    e = np.exp(l - np.max(l, axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    target_probs = probs[np.arange(len(t)), t]
    return float(-np.mean(np.log(target_probs)))
""",
    """import numpy as np
logits = np.array([[2.0, 0.5, -1.0], [0.1, 3.0, 0.1], [-0.5, 0.0, 2.5]])
targets = np.array([0, 2, 2])
mask = np.array([1, 0, 1])
loss = bert_mlm_loss(logits, targets, mask)
assert loss > 0, f'loss must be positive: {loss}'
loss_none = bert_mlm_loss(logits, targets, np.array([0, 0, 0]))
assert loss_none == 0.0, 'no masked tokens must return 0.0'
loss_easy = bert_mlm_loss(np.array([[10.0, -10.0]]), np.array([0]), np.array([1]))
assert loss_easy < 0.01, f'a confident correct prediction must have near-zero loss: {loss_easy}'
print('ALL TESTS PASSED')
""",
)

add(
    "dl-vit-patch-embed",
    """import numpy as np
def vit_patch_embed(image, patch_size, W):
    H, W_dim, C = image.shape
    P = patch_size
    patches = image.reshape(H // P, P, W_dim // P, P, C).transpose(0, 2, 1, 3, 4).reshape(-1, P * P * C)
    return patches @ W
""",
    """import numpy as np
img = np.arange(16).reshape(4, 4, 1)
W = np.ones((4, 2))
out = vit_patch_embed(img, 2, W)
assert out.shape == (4, 2), f'expected 4 patches of dim 2: got {out.shape}'
assert np.array_equal(out, [[10,10],[18,18],[42,42],[50,50]]), f'wrong patch embeddings: {out}'
print('ALL TESTS PASSED')
""",
)

add(
    "dl-lora-forward",
    """import numpy as np
def lora_forward(X, W0, A, B, alpha, r):
    out_base = X @ W0.T
    lora_out = (X @ A.T) @ B.T
    return out_base + (alpha / r) * lora_out
""",
    """import numpy as np
X = np.ones((2, 4))
W0 = np.ones((3, 4)) * 2
A = np.ones((2, 4)) * 0.5
B = np.ones((3, 2)) * 0.5
out = lora_forward(X, W0, A, B, 4, 2)
assert out.shape == (2, 3), f'wrong output shape: {out.shape}'
assert np.allclose(out, 12.0, atol=1e-6), f'wrong lora forward values: {out}'
print('ALL TESTS PASSED')
""",
)

add(
    "dl-flash-attention-conceptual",
    """import numpy as np
def flash_attn_single_query(q, K, V, block_size):
    N, d = K.shape
    m = -np.inf
    l = 0.0
    O = np.zeros((1, d))
    for j in range(0, N, block_size):
        K_j = K[j:j+block_size]
        V_j = V[j:j+block_size]
        s_j = q @ K_j.T
        m_new = max(m, np.max(s_j))
        p_j = np.exp(s_j - m_new)
        l_new = l * np.exp(m - m_new) + np.sum(p_j)
        O = (O * l * np.exp(m - m_new) + p_j @ V_j) / l_new
        m, l = m_new, l_new
    return O
""",
    """import numpy as np
q = np.array([[1.0, 0.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
V = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
out = flash_attn_single_query(q, K, V, 2)
# reference: standard (unscaled, per the problem's own formulas) softmax attention
scores = q @ K.T
w = np.exp(scores - np.max(scores)); w = w / w.sum()
expected = w @ V
assert np.allclose(out, expected, atol=1e-4), f'tiled result must match standard softmax attention: {out} vs {expected}'
print('ALL TESTS PASSED')
""",
)


def main():
    with open(METADATA_PATH, encoding="utf-8") as f:
        ts_problems = {p["slug"]: p for p in json.load(f)}

    with open(SEED_PATH, encoding="utf-8") as f:
        existing = json.load(f)
    existing_slugs = {e["slug"] for e in existing}

    new_entries = []
    failures = []

    for slug, (ref, harness) in PROBLEMS.items():
        if slug in existing_slugs:
            continue
        meta = ts_problems.get(slug)
        if meta is None:
            failures.append((slug, "no matching entry in problems.ts export"))
            continue

        # Self-verify: reference + harness must execute cleanly and print the pass marker.
        script = ref + "\n" + harness
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
        )
        if proc.returncode != 0 or "ALL TESTS PASSED" not in proc.stdout:
            failures.append((slug, proc.stderr.strip() or proc.stdout.strip()))
            continue

        new_entries.append(
            {
                "id": slug,
                "slug": slug,
                "title": meta["title"],
                "category": (meta.get("topics") or ["General"])[0],
                "difficulty": meta["difficulty"],
                "estimated_time": DIFFICULTY_TIME.get(meta["difficulty"], 15),
                "description": meta["description"],
                "tags": meta.get("topics", []),
                "python_template": meta["starter_code"],
                "test_cases": [{"type": "harness", "code": harness.strip()}],
            }
        )

    if failures:
        print(f"FAILED self-verification for {len(failures)} problem(s):")
        for slug, err in failures:
            print(f"  - {slug}: {err[:300]}")
        sys.exit(1)

    print(f"Self-verified {len(new_entries)} new problems.")
    all_entries = existing + new_entries
    with open(SEED_PATH, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2)
    print(f"Wrote {len(all_entries)} total entries to {SEED_PATH}")


if __name__ == "__main__":
    main()
