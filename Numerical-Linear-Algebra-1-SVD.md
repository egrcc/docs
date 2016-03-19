# Numerical Linear Algebra

## Singular Value Decomposition

### 1. Geometric Observation

The SVD is motivated by the following geometric fact:
 
<p style="text-align:center">**The image of the unit sphere under any $m \times n$ matrix is a hyperellipse.**

<p style="text-align:center"><img src="./svd1.png"/> 

We define the $n$ <i>singular values</i> of $A$: $\sigma_1, \sigma_2,\ldots,\sigma_n$ are the lengths of the $n$ principal semiaxes of $AS$, where $\sigma_1 \ge \sigma_2\ge \cdots \ge \sigma_n$. Define the $n$ <i>left singular vectors</i> of $A$ are the unit vectors $\{u_1, u_2, \ldots, u_n\}$ oriented in the directions of the principal semiaxes of $AS$. Define the $n$ <i>right singular vectors</i> of $A$ are the unit vectors $\{v_1, v_2, \ldots, v_n\} \in S$ that are the preimages of the principal semiaxes of $AS$, so $Av_j = \sigma_ju_j$.

### 2. Reduced SVD

$A \in \mathbb{R}^{m \times n} (m > n)$ and we assume $A$ has full rank $n$:

$$
A = \hat{U}\hat{\Sigma}V^T.
$$

<p style="text-align:center"><img src="./svd2.png"/> 

### 3. Full SVD

$$
A = U\Sigma V^T.
$$

<p style="text-align:center"><img src="./svd3.png"/> 

### 4. Formal Definition

Given $A \in \mathbb{R}^{m \times n}$, a <i>singular value decomposition</i> (SVD) of $A$ is a factorization
$$A = U\Sigma V^T$$
where 
$$
\begin{aligned}
&U \in \mathbb{R}^{m \times m} \quad \text{is unitary},\\\
&V \in \mathbb{R}^{n \times n} \quad \text{is unitary},\\\
&\Sigma \in \mathbb{R}^{m \times n} \quad \text{is diagonal}.
\end{aligned} 
$$
</br>

**Theorem 1.** Every matrix $A \in \mathbb{R}^{m \times n}$ has a singular value decomposition. Furthermore, the singular values $\{\sigma\_j\}$ are uniquely determined, and, if $A$ is square and the $\sigma\_j$ are distinct, the left and right singular vectors $\{u\_j\}$ and $\{v\_j\}$ are uniquely determined up to complex signs.


### Matrix properties via the SVD

**Theorem 2.** The rank of $A$ is $r$, the number of nonzero singular values.

**Theorem 3.** $\text{range}(A) = \langle u\_1,\ldots, u\_r \rangle$ and $\text{null}(A) = \langle v_{r + 1}, \ldots, v_n \rangle$.

**Theorem 4.** $||A||\_2 = \sigma\_1$ and $||A||\_F = \sqrt{\sigma\_1^2 + \sigma\_2^2 + \cdots + \sigma\_r^2}$.

**Theorem 5.** The nonzero singular values of $A$ are the square roots of the nonzero eigenvalues of $A^TA$ or $AA^T$.

**Theorem 6.** For $A \in \mathbb{R}^{m \times m}$, $|\text{det}(A)| = \prod\_{i = 1}^m \sigma\_i$.
 

### Low-Rank Approximations

**Theorem 7.** $A$ is the sum of $r$ rank-one matrices:
$$
A = \sum\_{j = 1}^r \sigma\_ju\_jv_j^T.
$$

**Theorem 8.** For any $\nu$ with $0 \le \nu \le r$, define
$$
A\_{\nu} = \sum\_{j = 1}^{\nu} \sigma\_ju\_jv\_j^T;
$$
if $\nu = p = \min\{m, n\}$, define $\sigma_{\nu + 1} = 1$. Then
$$
||A - A_{\nu}||\_2 = \inf\_{B \in \mathbb{R}^{m \times n}\\\ \text{rank}(B)\le \nu} ||A - B||\_2 = \sigma\_{\nu + 1}.
$$
 
**Theorem 9.** For any $\nu$ with $0 \le \nu \le r$, the matrix $A\_{\nu}$ also satisfies 
$$
||A - A\_{\nu}||\_F = \inf\_{B \in \mathbb{R}^{m \times n}\\\ \text{rank}(B)\le \nu} ||A - B||\_F = \sqrt{\sigma\_{\nu + 1}^2 + \cdots + \sigma\_{r}^2}.
$$

















