# Notation

This page defines the notation. Every guide, docstring and notebook in this
repository follows it.

## Symbols

Random variables are capital Latin letters, for example $Y$. Their
realizations — observed or sampled values — are lowercase Latin letters, for
example $y$. The cumulative distribution function of $Y$ is $F_Y$ and the
density is $f_Y$. A hat marks an estimate: $\hat{F}$ is the fitted
distribution, $F$ the true and often unknown one.

Sets and vectors are bold, for example
$\mathbf{X} = [X_1, \dots, X_J]$.

A single optimized parameter is a lowercase Greek letter, for example
$\vartheta$ or $\beta$. A vector of parameters is bold lowercase, for example
$\boldsymbol{\vartheta} = (\vartheta_1, \ldots, \vartheta_M)^\top$. A tuple
that collects several parameter groups — for example all weight matrices of a
network — is bold capital, for example $\boldsymbol{\Theta}$.

A function that a parameter vector defines carries it as a subscript:
$h_{\boldsymbol{\vartheta}}$ is the monotone transformation with coefficients
$\boldsymbol{\vartheta}$.

## The model's symbols

| Symbol | Meaning | In code |
|---|---|---|
| $X_j$, $Y$ | the variables of the DAG | node names in the spec |
| $\mathrm{pa}(X_j)$ | the causal parents of $X_j$ | `node_parents` |
| $U_j$ | the standard-logistic latent of node $j$ | `u = flow.abduct(df)` |
| $h_{\boldsymbol{\vartheta}}$ | the monotone transformation of a node | the `I` term; `theta` in docstrings |
| $\vartheta_k$ | the ordinal cutpoints, elements of $\boldsymbol{\vartheta}$ | `ordinal_cutpoints` |
| $\beta$ | a linear-shift coefficient; $e^\beta$ is an odds ratio | `LS`; `flow.ls_coefficients()` |
| $g_j$ | a complex shift, an MLP of one term's parents | `CS` |
| $\beta_0$ | the constant treatment effect of a `VC` term | `beta0` |
| $b_{\boldsymbol{\Theta}}$ | the penalized effect-modification network | the `VC` net; `units=` sets its layers |
| $\lambda$ | the L2 weight on $b_{\boldsymbol{\Theta}}$ | `penalty=` |
| $\sigma$ | the logistic function | `torch.sigmoid` |

Every node's model is one additive formula on the latent scale:

$$
u \;=\; h(x \mid \mathrm{pa}(x)) \;=\;
h_{\boldsymbol{\vartheta}(\cdot)}(x)
\;+\; \sum_j \beta_j\, x_j
\;+\; \sum_k g_k(x_k)
\;+\; (\beta_0 + b_{\boldsymbol{\Theta}}(x_{\text{mod}}))\, x_t .
$$

For a continuous node the shifts are added, as written. For an ordinal node
the shift is subtracted inside the sigmoid:
$P(Y \le k \mid \mathrm{pa}) = \sigma(\vartheta_k - \text{shift})$.

## Plain-text fallback

Docstrings and terminal output cannot render Greek subscripts. There,
`theta` stands for $\boldsymbol{\vartheta}$, `h_theta` for
$h_{\boldsymbol{\vartheta}}$, `b_theta` for $b_{\boldsymbol{\Theta}}$ and
`beta0` for $\beta_0$.
