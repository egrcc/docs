# Numerical Linear Algebra

## Projectors

### 1. Projectors

A <i>projector</i> is a square matrix $P$ that satisfies

$$
P^2 = P.
$$

### 2. Complementary Projectors

If $P$ is a projector, $I - P$ is also a projector,
$$
(I-P)^2 = I - P.
$$

The matrix $I - P$ is called the <i>complementary projector</i> to $P$.

**Properties**:

$$
\text{range}(I - P) = \text{null}(P).
$$

$$
\text{null}(I - P) = \text{range}(P).
$$

$$
\text{range}(P) \cap \text{null}(P) = \{0\}.
$$

### 3. Orthogonal Projectors

**Theorem 1.** A projector $P$ is orthogonal if and only if $P = P^T$.

### 4. Projection with an Orthogonal Basis

$$
P = QQ^T
$$

where the columns of $Q$ are orthonormal.

### 5. Projection with an Arbitrary Basis

$A \in \mathbb{R}^{m \times n}$ whose $j$th column is $a\_j$. $\{a\_1,\ldots,a\_n\}$ are linearly indendent vectors.

$$
P = A(A^TA)^{-1}A^T.
$$













