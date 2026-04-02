# Conservative vs Rotational: Dissipation in Transformer Layers

## The Jacobian Decomposition

For a transformer layer $f_l: \mathbb{R}^d \to \mathbb{R}^d$ where $d = S \cdot H$, the Jacobian is:

$$J_l = \frac{\partial f_l(X_0)}{\partial X_0}$$

Any Jacobian can be uniquely decomposed as:

$$J = S + \Omega$$

where:
- $S = \frac{1}{2}(J + J^T)$ is the **symmetric** part (conservative)
- $\Omega = \frac{1}{2}(J - J^T)$ is the **antisymmetric** part (rotational)

Our conservative ratio measures:

$$\text{cons\_ratio} = \frac{\|S\|^2_F}{\|S\|^2_F + \|\Omega\|^2_F} \in [0, 1]$$

- **1.0** = purely conservative (J = J^T)
- **0.5** = equal symmetric/antisymmetric (random)
- **0.0** = purely rotational (J = -J^T)

---

## Conservative = Gradient Field = CAN Dissipate

**Definition**: A vector field is **conservative** if it's the gradient of a scalar potential:

$$f(x) = \nabla \phi(x)$$

**Key property**: The Jacobian is symmetric:

$$J_{ij} = \frac{\partial f_i}{\partial x_j} = \frac{\partial^2 \phi}{\partial x_j \partial x_i} = \frac{\partial^2 \phi}{\partial x_i \partial x_j} = J_{ji}$$

(by equality of mixed partials)

**Dissipation**: Conservative fields can **change the norm** of vectors:

$$\|\Delta x\|^2 \approx \Delta x^T J^T J \Delta x = \Delta x^T S^2 \Delta x + \Delta x^T \Omega^2 \Delta x$$

For purely conservative ($\Omega = 0$), this simplifies to eigenvalue scaling:

- **Positive eigenvalues** of $S$: expansion (amplification)
- **Negative eigenvalues** of $S$: contraction (dissipation)
- **Mixed eigenvalues**: some directions grow, others shrink

### Example: LayerNorm (highly conservative)

From `gpt2_equations.md` line 51:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

This normalizes to zero mean and unit variance, then scales by $\gamma$. The Jacobian for the normalization part is approximately:

$$J \approx \frac{1}{\sigma}\left(I - \frac{1}{H}\mathbf{1}\mathbf{1}^T - \frac{1}{H}(x-\mu)(x-\mu)^T\right)$$

This is **symmetric** (conservative) and **contracts** all dimensions toward the mean. It's dissipative: $\|x_{\text{out}}\| < \|x_{\text{in}}\|$ (before the $\gamma$ scaling).

**Physical analogy**: Water flowing downhill — follows gradients of elevation, loses energy (dissipates).

---

## Rotational = Curl Field = PRESERVES Norm

**Definition**: A vector field is **rotational** (or solenoidal) if it's the curl of a vector potential:

$$f(x) = \nabla \times A(x)$$

**Key property**: The Jacobian is antisymmetric:

$$J_{ij} = -J_{ji}$$

**Norm preservation**: Antisymmetric matrices preserve inner products:

$$\langle x, \Omega x \rangle = x^T \Omega x = -x^T \Omega^T x = -x^T \Omega x = 0$$

So for purely rotational flow ($S = 0$), the rate of change of $\|x\|^2$ is:

$$\frac{d}{dt}\|x\|^2 = 2x^T \frac{dx}{dt} = 2x^T \Omega x = 0$$

The norm is **constant** — the flow rotates vectors without changing their magnitude.

### Example: Attention mixing (has rotational component)

From `gpt2_equations.md` lines 78-88, attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

The softmax creates a **weighted average** (conservative bias), but the $QK^T$ interaction creates **position mixing** (rotational component).

Consider two positions $s, t$ exchanging information via attention weights $\alpha_{st}$. The Jacobian block is:

$$J_{st} = \alpha_{st} W^V W^O$$

while

$$J_{ts} = \alpha_{ts} W^V W^O$$

If $\alpha_{st} \neq \alpha_{ts}$ (asymmetric attention), the cross-terms don't cancel → $J \neq J^T$ → rotational component.

**Physical analogy**: Fluid vortex — spins water around without pumping it in/out. Energy conserved (Hamiltonian dynamics).

---

## Why Middle Layers Are ~0.5 (Equal Mix)

From `gpt2_equations.md` lines 99-122, each transformer layer is:

$$X_l = X_{l-1} + \text{MHA}(\text{LayerNorm}(X_{l-1})) + \text{FFN}(\text{LayerNorm}(\cdots))$$

The **residual connection** $X_l = X_{l-1} + \Delta X$ makes the Jacobian:

$$J_l = I + J_{\text{MHA}} + J_{\text{FFN}}$$

where $J_{\text{MHA}}$ and $J_{\text{FFN}}$ are the Jacobians of the residual branches.

### Conservative components (symmetric):
1. **LayerNorm**: Explicitly symmetric, dissipative (contracts to unit norm)
2. **FFN**: $\text{GELU}(x W^1) W^2$ — for small updates, GELU is nearly linear → $J_{\text{FFN}} \approx W^2 \text{diag}(\text{GELU}'(x W^1)) W^1$ — not exactly symmetric but has diagonal structure → conservative bias

### Rotational components (antisymmetric):
1. **Attention cross-terms**: $QK^T$ creates asymmetric position mixing
2. **Non-linearity**: GELU introduces asymmetry when activations differ across dimensions

### The balance:

- **Early layers**: Input embeddings → features. Need contraction (dissipation) to reduce input variability. Conservative bias (0.6).
- **Middle layers**: Computation requires mixing/rotation. Attention dominates. Balanced (0.5).
- **Late layers**: Preparing for output projection. Less transformation needed. Conservative bias returns (0.55).

---

## Concrete Example: Linear Layer

To see dissipation vs rotation clearly, consider a simple linear layer: $f(x) = Wx$.

The Jacobian is just $J = W$.

### Case 1: Symmetric W (conservative)

$$W = \begin{pmatrix} 2 & 1 \\ 1 & 0.5 \end{pmatrix}$$

Eigenvalues: $\lambda_1 \approx 2.28$, $\lambda_2 \approx 0.22$

Effect on norm: $\|Wx\|^2 = x^T W^T W x$ — depends on direction
- Along eigenvector $v_1$: expands by $2.28^2 \approx 5.2\times$
- Along eigenvector $v_2$: contracts by $0.22^2 \approx 0.05\times$

**This is dissipative**: some information is amplified, some is suppressed. Norm can decrease (information loss).

### Case 2: Antisymmetric W (rotational)

$$W = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$$

This is a 90° rotation matrix.

Effect on norm:
$$\|Wx\|^2 = x^T W^T W x = x^T W^T W x = x^T I x = \|x\|^2$$

**This preserves the norm exactly**: no dissipation, pure rotation. All information is preserved.

---

## Why Transformers Need Both

**Conservative (dissipation)**:
- Regularizes representations (LayerNorm, dropout)
- Creates information bottlenecks (forces compression)
- Enables gradient flow (symmetric → better conditioned)

**Rotational (computation)**:
- Mixes information across positions (attention)
- Creates non-linear transformations (complex functions need rotation)
- Preserves information capacity (reversibility)

The **~0.5 ratio** is the optimal trade-off:
- Too conservative (>0.8): Information bottleneck, can't compute complex functions
- Too rotational (<0.2): Training instability, exploding/vanishing gradients
- Balanced (~0.5): Maximum expressiveness + trainability

---

## Connection to Literature

1. **Neural ODEs**: View $X_l = X_{l-1} + f(X_{l-1})$ as discretized ODE $\frac{dX}{dt} = f(X)$
   - Symmetric J → dissipative dynamics → convergence to attractors
   - Antisymmetric J → Hamiltonian dynamics → energy preservation

2. **Lyapunov Stability**: Stable training requires some dissipation (conservative structure) to prevent divergence

3. **Reversible Architectures**: RevNets use $\det(J) = 1$ (invertible) but not necessarily antisymmetric. They achieve reversibility through architectural constraints, not pure Hamiltonian flow.

4. **Information Bottleneck**: Conservative layers create compression; rotational layers preserve information. The balance determines the network's information flow.

---

## Testable Predictions

1. **Regularization**: More dropout/weight decay → higher conservativeness (more dissipation to prevent overfitting)

2. **Model size**: Larger models may have *lower* conservativeness (more capacity → less need for compression)

3. **Task type**:
   - Classification (compress to class label) → higher conservativeness
   - Generation (preserve details) → higher rotational component

4. **Training dynamics**: Conservativeness should increase during training as the model learns to compress/regularize
