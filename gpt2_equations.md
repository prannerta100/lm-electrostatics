# GPT-2 Forward Pass: Complete Equations

From $\text{input\_ids}$ to $P(x_{t+1} | x_{1:t})$.

---

## 1. Notation

| Symbol | Description | Shape / Value |
|--------|-------------|---------------|
| $S$ | Sequence length | variable, $\leq S_{\max}$ |
| $S_{\max}$ | Maximum sequence length | 1024 |
| $B$ | Batch size (assumed 1, omitted) | 1 |
| $H$ | Hidden dimension / model dimension | 768 |
| $N$ | Number of attention heads | 12 |
| $d_k$ | Head dimension $= H / N$ | 64 |
| $L$ | Number of layers | 12 |
| $V$ | Vocabulary size | 50257 |
| $X_l$ | Output of layer $l$ | $(S, H)$ |
| $\gamma, \beta$ | LayerNorm scale and shift parameters | $(H,)$ |
| $\epsilon$ | LayerNorm epsilon | $10^{-5}$ |
| $M$ | Causal attention mask | $(S, S)$ |

---

## 2. Token + Positional Embedding

$$X_0 = W_e[\text{input\_ids}] + W_p[\text{positions}]$$

where $\text{positions} = [0, 1, \ldots, S-1]$.

| Parameter | Shape | Description |
|-----------|-------|-------------|
| $W_e$ | $(V, H)$ | Token embedding matrix |
| $W_p$ | $(S_{\max}, H)$ | Positional embedding matrix |

$X_0$ has shape $(S, H)$. Each row is the sum of a token embedding and a positional embedding.

---

## 3. Transformer Layers (Pre-Norm)

For each layer $l = 1, \ldots, L$:

### 3a. LayerNorm before Attention

$$H_l = \text{LayerNorm}(X_{l-1};\; \gamma^{\text{attn}}_l,\; \beta^{\text{attn}}_l)$$

where LayerNorm is defined as:

$$\text{LayerNorm}(x;\; \gamma,\; \beta) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

with $\mu$ and $\sigma^2$ computed over the last dimension ($H$):

$$\mu = \frac{1}{H} \sum_{i=1}^{H} x_i, \qquad \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2$$

Parameters: $\gamma^{\text{attn}}_l, \beta^{\text{attn}}_l \in \mathbb{R}^{H}$.

### 3b. Multi-Head Attention

**Linear projections:**

$$Q = H_l W^Q_l + b^Q_l, \qquad K = H_l W^K_l + b^K_l, \qquad V = H_l W^V_l + b^V_l$$

| Parameter | Shape |
|-----------|-------|
| $W^Q_l, W^K_l, W^V_l$ | $(H, H)$ |
| $b^Q_l, b^K_l, b^V_l$ | $(H,)$ |

**Split into $N$ heads.** For head $i = 1, \ldots, N$:

$$Q_i, K_i, V_i \in \mathbb{R}^{S \times d_k}$$

obtained by slicing columns $[(i-1) d_k : i \, d_k]$ from $Q$, $K$, $V$ respectively.

**Scaled dot-product attention with causal mask:**

$$\text{Attention}_i = \text{softmax}\!\left(\frac{Q_i K_i^T}{\sqrt{d_k}} + M\right) V_i$$

where $M \in \mathbb{R}^{S \times S}$ is the causal mask:

$$M_{st} = \begin{cases} 0 & \text{if } s \geq t \\ -\infty & \text{if } s < t \end{cases}$$

Each $\text{Attention}_i$ has shape $(S, d_k)$.

**Concatenate and project:**

$$\text{MHA}(H_l) = \text{Concat}(\text{Attention}_1, \ldots, \text{Attention}_N) \, W^O_l + b^O_l$$

| Parameter | Shape |
|-----------|-------|
| $W^O_l$ | $(H, H)$ |
| $b^O_l$ | $(H,)$ |

Output shape: $(S, H)$.

### 3c. First Residual Connection

$$A_l = X_{l-1} + \text{MHA}(\text{LayerNorm}(X_{l-1}))$$

$A_l$ has shape $(S, H)$.

### 3d. LayerNorm before FFN

$$F_l = \text{LayerNorm}(A_l;\; \gamma^{\text{ffn}}_l,\; \beta^{\text{ffn}}_l)$$

Parameters: $\gamma^{\text{ffn}}_l, \beta^{\text{ffn}}_l \in \mathbb{R}^{H}$.

### 3e. Feed-Forward Network

$$\text{FFN}(F_l) = \text{GELU}(F_l \, W^1_l + b^1_l) \, W^2_l + b^2_l$$

| Parameter | Shape |
|-----------|-------|
| $W^1_l$ | $(H, 4H)$ |
| $b^1_l$ | $(4H,)$ |
| $W^2_l$ | $(4H, H)$ |
| $b^2_l$ | $(H,)$ |

### 3f. Second Residual Connection

$$X_l = A_l + \text{FFN}(\text{LayerNorm}(A_l))$$

$X_l$ has shape $(S, H)$.

---

## 4. Final Output

**Final LayerNorm:**

$$H_{\text{out}} = \text{LayerNorm}(X_L;\; \gamma_f,\; \beta_f)$$

Parameters: $\gamma_f, \beta_f \in \mathbb{R}^{H}$.

**Logits (weight tying with embedding matrix):**

$$Z = H_{\text{out}} \, W_u, \qquad W_u = W_e^T$$

$W_u$ has shape $(H, V)$, so $Z$ has shape $(S, V)$.

**Next-token probability:**

$$P(x_{t+1} | x_{1:t}) = \text{softmax}(Z[t, :])$$

where $\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$.

---

## 5. Complete Forward Pass Summary

$$P(x_{t+1} | x_{1:t}) = \text{softmax}\!\Big(W_e^T \cdot \text{LayerNorm}\big(\text{Layer}_L(\text{Layer}_{L-1}(\cdots \text{Layer}_1(W_e[\text{input\_ids}] + W_p[\text{pos}])\cdots))\big)\Big)\bigg|_{t}$$

where each $\text{Layer}_l$ is:

$$\text{Layer}_l(X) = A + \text{FFN}(\text{LayerNorm}(A)), \qquad A = X + \text{MHA}(\text{LayerNorm}(X))$$

---

## 6. GELU Approximation

GPT-2 uses the tanh approximation of the Gaussian Error Linear Unit:

$$\text{GELU}(x) = 0.5 \, x \left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715 \, x^3\right)\right)\right)$$

---

## 7. Parameter Count

### Per-layer parameters ($l = 1, \ldots, L$)

| Component | Parameter | Shape | Count |
|-----------|-----------|-------|-------|
| Attention LayerNorm | $\gamma^{\text{attn}}_l$ | $(H,)$ | 768 |
| | $\beta^{\text{attn}}_l$ | $(H,)$ | 768 |
| Attention $Q$ | $W^Q_l$ | $(H, H)$ | 589,824 |
| | $b^Q_l$ | $(H,)$ | 768 |
| Attention $K$ | $W^K_l$ | $(H, H)$ | 589,824 |
| | $b^K_l$ | $(H,)$ | 768 |
| Attention $V$ | $W^V_l$ | $(H, H)$ | 589,824 |
| | $b^V_l$ | $(H,)$ | 768 |
| Attention output | $W^O_l$ | $(H, H)$ | 589,824 |
| | $b^O_l$ | $(H,)$ | 768 |
| FFN LayerNorm | $\gamma^{\text{ffn}}_l$ | $(H,)$ | 768 |
| | $\beta^{\text{ffn}}_l$ | $(H,)$ | 768 |
| FFN up-projection | $W^1_l$ | $(H, 4H)$ | 2,359,296 |
| | $b^1_l$ | $(4H,)$ | 3,072 |
| FFN down-projection | $W^2_l$ | $(4H, H)$ | 2,359,296 |
| | $b^2_l$ | $(H,)$ | 768 |
| **Per-layer total** | | | **7,087,872** |

### Non-layer parameters

| Component | Parameter | Shape | Count |
|-----------|-----------|-------|-------|
| Token embedding | $W_e$ | $(V, H)$ | 38,597,376 |
| Positional embedding | $W_p$ | $(S_{\max}, H)$ | 786,432 |
| Final LayerNorm | $\gamma_f$ | $(H,)$ | 768 |
| | $\beta_f$ | $(H,)$ | 768 |
| **Non-layer total** | | | **39,385,344** |

Note: The unembedding matrix $W_u = W_e^T$ is tied to $W_e$ and contributes no additional parameters.

### Total

$$\text{Total} = 39{,}385{,}344 + 12 \times 7{,}087{,}872 = 124{,}439{,}808$$

Approximately **124M parameters**.

---

## 8. Vector Field Definition (for this project)

For electrostatics analysis, we define the vector field:

$$f_l : \mathbb{R}^{S \cdot H} \to \mathbb{R}^{S \cdot H}$$

$$f_l(\text{flatten}(X_0)) = \text{flatten}(X_l)$$

where $X_0$ is the initial embedding and $X_l$ is the output of layer $l$, both flattened from $(S, H)$ to $(S \cdot H,)$.

All model parameters $\{W_e, W_p, W^Q_l, W^K_l, W^V_l, W^O_l, W^1_l, W^2_l, \gamma, \beta, b\}$ are **frozen**. The only variable is $X_0$.

We measure:
- **Divergence**: $\nabla \cdot f_l = \text{Tr}(J_l)$ where $J_l = \frac{\partial f_l}{\partial X_0}$
- **Jacobian asymmetry**: $\frac{\|J_l - J_l^T\|}{\|J_l + J_l^T\|}$ measuring how non-conservative (rotational) the vector field is
