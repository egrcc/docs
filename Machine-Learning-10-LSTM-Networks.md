# Machine Learning

## LSTM Networks

<p style="text-align:center"><img src="./LSTM.png" height="244.1" width="650"/>

### 1. Basics

**sigmoid function**:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

$$
\sigma^{'}(x) = \sigma(x)\cdot[1 - \sigma(x)]
$$

**hyperbolic function**:

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

$$
\tanh^{'}(x) = 1 - \tanh^2(x)
$$

**softmax function**:

$$
\mathbf{y} = \text{softmax}(\mathbf{x})
$$

$$
y_i = \frac{e^{x_i}}{\sum^n_{j = 1}e^{x_j}}
$$

$$
\frac{\partial y_i}{\partial x_j} =
\begin{cases}
-y_i\cdot y_j, & i \ne j\\\\
y_i\cdot (1 - y_i), & i = j
\end{cases}
$$

**standard RNN**:

<p style="text-align:center"><img src="./LSTM-SimpleRNN.png" height="244.1" width="650"/>


### 2. Model

**input**:

$$
\mathbf{x} = (\mathbf{x}\_1,\mathbf{x}\_2,\ldots,\mathbf{x}\_T) \quad \mathbf{x}\_t \in \mathbb{R}^n
$$

**initialize hidden state**:

$$
\mathbf{h}\_0 \in \mathbb{R}^k
$$

**forget gate**:

$$
\mathbf{f}\_t = \sigma(\mathbf{W}\_{xf}\mathbf{x}\_t + \mathbf{W}\_{hf}\mathbf{h}\_{t-1} + \mathbf{b}\_f)
$$

<p style="text-align:center"><img src="./LSTM-focus-f.png" height="280.8" width="450"/>


**input gate**:

$$
\mathbf{i}\_t = \sigma(\mathbf{W}\_{xi}\mathbf{x}\_t + \mathbf{W}\_{hi}\mathbf{h}\_{t-1} + \mathbf{b}\_i)
$$

**cell update transformation**:

$$
\mathbf{\tilde{c}}\_t = \tanh(\mathbf{W}\_{xc}\mathbf{x}\_t + \mathbf{W}\_{hc}\mathbf{h}\_{t-1} + \mathbf{b}\_c)
$$

<p style="text-align:center"><img src="./LSTM-focus-i.png" height="280.8" width="450"/>

**cell state update**:

$$
\mathbf{c}\_t = \mathbf{f}\_t \odot \mathbf{c}\_{t - 1} + \mathbf{i}\_t \odot \mathbf{\tilde{c}}\_t
$$

<p style="text-align:center"><img src="./LSTM-focus-C.png" height="280.8" width="450"/>

**output gate**:

$$
\mathbf{o}\_t = \sigma(\mathbf{W}\_{xo}\mathbf{x}\_t + \mathbf{W}\_{ho}\mathbf{h}\_{t-1} + \mathbf{b}\_o)
$$

**hidden update**:

$$
\mathbf{h}\_t = \mathbf{o}\_t \odot \tanh(\mathbf{c}\_{t})
$$

<p style="text-align:center"><img src="./LSTM-focus-o.png" height="280.8" width="450"/>

### 3. Variants on LSTM

**extending LSTM with peephole connections**:

$$
\begin{aligned}
&\mathbf{i}\_t = \sigma(\mathbf{W}\_{xi}\mathbf{x}\_t + \mathbf{W}\_{hi}\mathbf{h}\_{t-1} + \mathbf{W}\_{ci}\mathbf{c}\_{t-1} + \mathbf{b}\_i)\\\
&\mathbf{f}\_t = \sigma(\mathbf{W}\_{xf}\mathbf{x}\_t + \mathbf{W}\_{hf}\mathbf{h}\_{t-1} +  \mathbf{W}\_{cf}\mathbf{c}\_{t-1} + \mathbf{b}\_f)\\\
&\mathbf{o}\_t = \sigma(\mathbf{W}\_{xo}\mathbf{x}\_t + \mathbf{W}\_{ho}\mathbf{h}\_{t-1} + \mathbf{W}\_{c0}\mathbf{c}\_{t} + \mathbf{b}\_o)
\end{aligned}
$$
 
<p style="text-align:center"><img src="./LSTM-var-peepholes.png" height="280.8" width="450"/>

**use coupled forget and input gates**:

$$
\mathbf{c}\_t = \mathbf{f}\_t \odot \mathbf{c}\_{t - 1} + (1 - \mathbf{f}\_t) \odot \mathbf{\tilde{c}}\_t
$$

<p style="text-align:center"><img src="./LSTM-var-tied.png" height="280.8" width="450"/>

### 4. Gated Recurrent Unit(GRU)

**update gate**:

$$
\mathbf{z}\_t = \sigma(\mathbf{W}\_{xz}\mathbf{x}\_t + \mathbf{W}\_{hz}\mathbf{h}\_{t-1} + \mathbf{b}\_z)
$$

**reset gate**:

$$
\mathbf{r}\_t = \sigma(\mathbf{W}\_{xr}\mathbf{x}\_t + \mathbf{W}\_{hr}\mathbf{h}\_{t-1} + \mathbf{b}\_r)
$$

**hidden update**:

$$
\begin{aligned}
&\mathbf{\tilde{h}}\_t = \tanh(\mathbf{W}\_{xh}\mathbf{x}\_t + \mathbf{W}\_{hh}(\mathbf{r}\_t \odot \mathbf{h}\_{t-1}) + \mathbf{b}\_h) \\\ 
&\mathbf{h}\_t = (1 - \mathbf{z}\_t) \odot 
\mathbf{\tilde{h}}\_t + \mathbf{z}\_t \odot \mathbf{h}\_{t -1}
\end{aligned}
$$

<p style="text-align:center"><img src="./LSTM-var-GRU.png" height="280.8" width="450"/>

### Reference

1. **Understanding LSTM Networks**:
http://colah.github.io/posts/2015-08-Understanding-LSTMs/