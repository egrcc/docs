# Machine Learning

## Principal Components Analysis

### 1. Basics

**variance**:
$$
D\xi := E(\xi - E\xi)^2 = E\xi^2 - (E\xi)^2
$$

**normalization**:
$$
\frac{\xi - E\xi}{\sqrt{D\xi}}
$$

**covariance**:
$$
\begin{aligned}
\text{cov}(\xi,\eta) :&= E[(\xi - E\xi)(\eta - E\eta)]\\
&= E\xi\eta - E\xi \cdot E\eta
\end{aligned}
$$

**correlation**:
$$
\rho(\xi,\eta) := \frac{\text{cov}(\xi,\eta)}{\sqrt{D\xi \cdot D\eta}}
$$

**SVD**:
$$
M = U \Sigma V^{\top} \; \text{where} \; U^{\top}U = I, V^{\top}V = I\\
M^{\top}MV = V\Sigma^{2}\\
MM^{\top}U = U\Sigma^2
$$

### 2. Model

**input**:
$$
\{x^{(i)};i = 1,\ldots,m\},x^{(i)} \in \mathbb{R}^n
$$

**normalization**:
$$
x^{(i)} = x^{(i)} - \frac{1}{m}\sum_{i = 1}^m x^{(i)}\\
x^{(i)}_j = \frac{x^{(i)}_j}{\sqrt{\frac{1}{m}\sum_{i = 1}^m (x^{(i)}_j)^2}}
$$

**objective function**:

$$
\begin{aligned}
J(u) &= \frac{1}{m}\sum_{i = 1}^m((x^{(i)})^{\top}u)^2 = \frac{1}{m}\sum_{i = 1}^mu^{\top}x^{(i)}(x^{(i)})^{\top}u\\
&= u^{\top}\left( \frac{1}{m}\sum_{i = 1}^mx^{(i)}(x^{(i)})^{\top} \right)u = u^{\top}\Sigma u
\end{aligned}
$$

**optimization**:

$$
\max J(u) \quad \text{s.t} \quad u^{\top}u = 1
$$

**solution**:

$u$ is the principal eigenvector of $\Sigma = \frac{1}{m}\sum_{i = 1}^mx^{(i)}(x^{(i)})^{\top} = \frac{1}{m}X^{\top}X$.

**The k-th component**:

$$
\hat{X}_k = X - \sum_{s = 1}^{k - 1}Xu_{(s)}u_{(s)}^{\top}\\
u_{(k)} = \text{argmax}_{||u||=1}\frac{1}{m}||\hat{X}_ku||^2
$$

**the eigenvectors of $\Sigma$**:

$$X^{\top} = 
\begin{bmatrix} 
| &amp; | &amp; &amp; |  \\
x^{(1)} &amp; x^{(2)} &amp; \cdots &amp; x^{(n)}  \\
| &amp; | &amp; &amp; | 
\end{bmatrix} \\
\,\\
\text{SVD}:X^{\top} = USV^{\top}\\
\,\\
U = 
\begin{bmatrix} 
| &amp; | &amp; &amp; |  \\
u_{(1)} &amp; u_{(2)} &amp; \cdots &amp; u_{(n)}  \\
| &amp; | &amp; &amp; | 
\end{bmatrix} 
$$

**output**:

$$
y^{(i)} = 
\begin{bmatrix} 
u_{(1)}^{\top}x^{(i)} \\
u_{(2)}^{\top}x^{(i)} \\
\vdots \\ 
u_{(k)}^{\top}x^{(i)} \\ 
\end{bmatrix} \in \mathbb{R}^k 
$$

**recover**:

$$
\hat{x}  = U \begin{bmatrix} \tilde{y}_1 \\ \vdots \\ \tilde{y}_k \\ 0 \\ \vdots \\ 0 \end{bmatrix}  
= \sum_{i=1}^k u_{(i)} \tilde{y}_i. 
$$

### 3. TODO
1. Optimization
2. Calculating the SVD