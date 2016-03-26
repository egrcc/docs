# Numerical Linear Algebra

## Norms

### 1. Vector Norms

A <i>norm</i> is a function $||\cdot||: \mathbb{R}^m \to \mathbb{R}$ that assigns a real-valued length to each vector that satisfy the following three conditions.
$$
\begin{aligned}
&(1)\, ||x|| \ge 0, \text{and } ||x|| = 0 \text{ only if } x = 0,\\\
&(2)\, ||x + y|| \le ||x|| + ||y||,\\\
&(3)\, ||\alpha x|| = |\alpha|\, ||x||.
\end{aligned}
$$

**Examples**:

<p style="text-align:center"><img src="./norms.png"/> 

$$
\begin{aligned}
&||x||\_1 = \sum\_{i = 1}^m |x\_i|,\\\
&||x||\_2 = \left(\sum\_{i = 1}^m |x\_i|^2\right)^{1/2} = \sqrt{x^Tx},\\\
&||x||\_{\infty} = \max\_{1\le i \le m} |x\_i|,\\\
&||x||\_p = \left(\sum\_{i = 1}^m |x\_i|^p\right)^{1/p} \quad (1 \le p < \infty).
\end{aligned}
$$

### 2. Matrix Norms Induced by Vector Norms

$A \in \mathbb{R}^{m \times n}$, the matrix norm can be defined:

$$
||A||\_{(m,n)} = \sup\_{x \in \mathbb{R}^n\\\ x\ne 0}\frac{||Ax||\_{(m)}}{||x||\_{(n)}} = \sup\_{x \in \mathbb{R}^n\\\ ||x||\_{(n)} = 1}||Ax||_{(m)}.
$$

**The $p$-Norm of a diagonal Matrix**:

$$
D = \begin{bmatrix} 
d_1 &amp; &amp; &amp; \\\
&amp; d_2 &amp; &amp; \\\
&amp; &amp; \ddots &amp; \\\
&amp; &amp; &amp; d_m 
\end{bmatrix}
$$

then $||D||\_p = \max\_{1\le i \le m}|d_i|$.

**The $1$-Norm of a Matrix**:

$A \in \mathbb{R}^{m \times n}$, then $||A||_1$ id equal to the "maximum column sum" of $A$.

$$
||Ax||\_1 = ||\sum\_{j = 1}^n x\_ja\_j||\_1 \le \sum\_{j = 1}^n |x\_j|\, ||a\_j||\_1 \le \max\_{1\le j \le n} ||a\_j||\_1.
$$

$$||A||\_1 = \max\_{1\le j \le n} ||a\_j||\_1.$$

**The $\infty$-Norm of a Matrix**:

$$||A||\_{\infty} = \max\_{1\le j \le m} ||a\_j^T||\_1,$$

where $a\_j^T$ denotes the $j$th row of $A$.

### 3. Cauchy-Schwarz and Holder Inequalities

Let $p$ and $q$ satisfy $1/p + 1/q = 1,$ with $1\le p,q \le \infty$. Then the <i>Holder inequality</i> states that, for any vectors $x$ and $y$,
$$
|x^Ty| \le ||x||\_p||y||\_q.
$$

### 4. Bounding $||AB||$ in an Induced Matrix Norm

$$
||AB||\_{(l,n)} \le ||A||\_{(l,m)}||B||\_{(m,n)}.
$$

### 5. General Matrix Norms

**Frobenius norm**:

$$
||A||\_F = \left(\sum\_{i=1}^m\sum\_{j=1}^n |a\_{ij}|^2\right)^{1/2}.
$$

$$
||AB||\_F^2 = ||A||\_F^2 ||B||\_F^2
$$

**Theorem 1.** For any $A \in \mathbb{R}^{m \times n}$ and unitary $Q \in \mathbb{R}^{m \times m}$, we have
$$
||QA||\_2 = ||A||\_2, \quad ||QA||\_F = ||A||\_F.
$$






















