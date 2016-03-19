# Machine Learning

## K-Means

### 1. model

**input**:
$$
\{x^{(1)},\ldots,x^{(m)}\},\;x^{(i)} \in \mathbb{R}^n
$$

**initialize**:
$$
\text{cluster centroid: }\mu_1,\mu_2,\ldots,\mu_k \in \mathbb{R}^n
$$

**repeat until convergence**:
$$
c^{(i)} = \mathop{\text{argmin}}_{j}||x^{(i)} - \mu_j||^2 \\
\mu_j = \frac{\sum_{i = 1}^m1\{c^{(i)}=j\}x^{(i)}}{\sum_{i = 1}^m1\{c^{(i)}=j\}}
$$

**output**:

$$
y^{(i)} = \mathop{\text{argmin}}_{j}||x^{(i)} - \mu_j||^2
$$

### 2. convergence