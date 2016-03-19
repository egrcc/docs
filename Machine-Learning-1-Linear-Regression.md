# Machine Learning

## Linear Regression

### 1. Some equations
$$
\begin{aligned}
\nabla_{A}\text{tr}(AB) &= \nabla_{A}\text{tr}(BA) = B^T\\
\nabla_{A^T}f(A) &= (\nabla_{A}f(A))^T\\
\nabla_{A}\text{tr}(ABA^TC) &= CAB + C^{T}AB^{T}\\
\nabla_{A}|A| &= |A|(A^{-1})^T
\end{aligned}
$$

### 2. Model

**input**:
$$(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}),\cdots, (x^{(m)}, y^{(m)})$$

**hypothesis**:
$$
\begin{aligned}
h_{\theta}(x) &= \theta_0 + \theta_{1}x_1 + \theta_{2}x_2 + \cdots + \theta_{n}x_n\\
&= \sum_{i = 0}^{n}\theta_{i}x_i = \theta^Tx
\end{aligned}
$$

**cost function**:
$$J(\theta) = \frac{1}{2}\sum_{i = 1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)})^2 = \frac{1}{2}(X\theta - y)^T(X\theta - y)$$

**derivative**:
$$
\begin{aligned}
\nabla_{\theta}J(\theta) &= \nabla_{\theta}(\frac{1}{2}\theta^TX^TX\theta -y^TX\theta - \frac{1}{2}y^Ty)\\
&= X^TX\theta - X^Ty
\end{aligned}
$$

**solution**:
$$\nabla_{\theta}J(\theta) = 0 \Longrightarrow X^TX\theta = X^Ty \Longrightarrow \theta = (X^TX)^{-1}X^Ty$$

### 3. Locally weighted linear regression

**weights**:
$$\omega_i = \text{exp}(-\frac{(x^{(i)} - x)^2}{2k^2})$$

**cost function**:
$$
\begin{aligned}
J(\theta) &= \frac{1}{2}\sum_{i = 1}^{m}\omega_i(h_{\theta}(x^{(i)}) - y^{(i)})^2 \\
&= \frac{1}{2}(WX\theta - Wy)^T(X\theta - y)
\end{aligned}$$
 
**derivative**:
$$
\nabla_{\theta}J(\theta) = X^TWX\theta - X^TWy
$$

**solution**:
$$
\begin{aligned}
\nabla_{\theta}J(\theta) = 0 &\Longrightarrow X^TWX\theta = X^TWy \\
&\Longrightarrow \theta = (X^TWX)^{-1}X^TWy
\end{aligned}$$

### 4. Ridge regression

**cost function**:
$$
\begin{aligned}
J(\theta) &= \frac{1}{2}\sum_{i = 1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)})^2 + \frac{1}{2}\lambda\sum_{i = 0}^{n}\theta_i^2 \\
&= \frac{1}{2}(X\theta - y)^T(X\theta - y) + \frac{1}{2}\lambda\theta^T\theta
\end{aligned}$$

**derivative**:
$$
\nabla_{\theta}J(\theta) = (X^TX + \lambda I)\theta - X^Ty
$$

**solution**:
$$
\begin{aligned}
\nabla_{\theta}J(\theta) = 0 &\Longrightarrow (X^TX + \lambda I)\theta = X^Ty \\
&\Longrightarrow \theta = (X^TX + \lambda I)^{-1}X^Ty
\end{aligned}$$