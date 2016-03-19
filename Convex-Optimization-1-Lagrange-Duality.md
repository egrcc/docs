# Convex Optimization

## Lagrange Duality

### 1. Lagrange Dual Function

**Standard form problem** (not necessarily convex):

$$
\begin{aligned}
\text{minimize}\quad &f_0(x) \\\
\text{subject to}\quad &f_i(x)\le 0, \quad i = 1, \ldots, m \\\
& h_i(x) = 0, \quad i = 1, \ldots, p
\end{aligned}
$$
 
variable $x \in \mathbb{R}^n$, domain $\mathcal{D}$, optimal value $p^*$.

**Lagrangian**:

$L: \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$, with $\text{dom } L = \mathcal{D} \times \mathbb{R}^m \times \mathbb{R}^p$,
$$
L(x, \lambda, \nu) = f\_0(x) + \sum\_{i = 1}^m \lambda\_if\_i(x) + \sum\_{i = 1}^p\nu\_ih\_i(x)
$$

**Lagrange dual function**:
$g: \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$,
$$
\begin{aligned}
g(\lambda, \nu) &= \inf\_{x \in \mathcal{D}} L(x, \lambda, \nu) \\\
&= \inf\_{x \in \mathcal{D}} \left( f\_0(x) + \sum\_{i = 1}^m \lambda\_if\_i(x) + \sum\_{i = 1}^p\nu\_ih\_i(x)\right)
\end{aligned}
$$

$g$ is concave, can be $-\infty$ for some $\lambda, \nu$.

**Lower bound property**:

If $\lambda \succeq 0$, then $g(\lambda, \nu) \le p^*$.

### 2. The Dual Problem

**Lagrange dual problem**:

$$
\begin{aligned}
\text{maxmize}\quad &g(\lambda, \nu) \\\
\text{subject to}\quad &\lambda \succeq 0
\end{aligned}
$$
 
A convex optimization problem, optimal value denoted $d^{\*}$.

**Weak duality**:
$$
d^{\*} \le p^*
$$

**Strong duality**:
$$
d^{\*} = p^*
$$

**Slater's constraint qualification**:

Strong duality holds for a convex problem
$$
\begin{aligned}
\text{minimize}\quad &f_0(x) \\\
\text{subject to}\quad &f_i(x)\le 0, \quad i = 1, \ldots, m \\\
&Ax = b
\end{aligned}
$$

if it is strictly feasible, 
$$
\exists x \in \text{int } \mathcal{D}: \quad f_i(x)\le 0, \quad i = 1, \ldots, m, \quad Ax = b
$$

### 3. Karush-Kuhn-Tucker (KKT) Conditions

**Complementary slackness**:

Assume strong duality holds, $x^{\*}$ is primal optimal, $(\lambda^{\*}, \nu^*)$ is dual optimal

$$
\begin{aligned}
f\_0(x^{\*}) = g(\lambda^{\*}, \nu^{\*}) &= \inf\_x \left( f\_0(x) + \sum\_{i = 1}^m \lambda\_i^{\*}f\_i(x) + \sum\_{i = 1}^p\nu^{\*}\_ih\_i(x)\right) \\\
&\le f\_0(x^{\*}) + \sum\_{i = 1}^m \lambda\_i^{\*}f\_i(x^{\*}) + \sum\_{i = 1}^p\nu^{\*}\_ih\_i(x^{\*})\\\
&\le f_0(x^{\*})
\end{aligned}
$$

hence, the two inequalities hold with equality:

- $x^{\*}$ minimizes $L(x, \lambda^{\*}, \nu^*)$
- $\lambda\_i^{\*}f\_i(x^{\*}) = 0$ for $i = 1, \ldots, m$ (known as complementary slackness):
$$
\lambda\_i^{\*} > 0 \Rightarrow f\_i(x^{\*}) = 0, \quad f\_i(x^{\*}) < 0 \Rightarrow \lambda\_i^{\*} = 0
$$

**KKT conditions**:

1. primal constraints: $f\_i(x) \le 0, i = 1, \ldots, m, h_i(x) = 0, i = 1, \ldots, p$
2. dual constraints: $\lambda \succeq 0$
3. complementary slackness: $\lambda\_if\_i(x) = 0, i = 1, \ldots, m$
4. gradient of Lagrangian with respect to $x$ vanishes:
$$
\nabla f\_0(x) + \sum\_{i = 1}^m \lambda\_i \nabla f\_i(x) + \sum\_{i = 1}^p\nu\_i \nabla h\_i(x) = 0
$$

if strong duality holds and $x, \lambda, \nu$ are optimal, then they must satisfy the KKT conditions.

**KKT conditions for convex problem**:

If $\tilde{x}, \tilde{\lambda}, \tilde{\nu}$ satisfy KKT for a convex problem, then they are optimal:

- from complementary slackness: $f_0(\tilde{x}) = L(\tilde{x}, \tilde{\lambda}, \tilde{\nu})$
- from 4th condition (and convexity): $g(\tilde{\lambda}, \tilde{\nu}) = L(\tilde{x}, \tilde{\lambda}, \tilde{\nu})$

hence, $f_0(\tilde{x}) = g(\tilde{\lambda}, \tilde{\nu})$.

















