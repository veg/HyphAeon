# Theoretical Framework: Deconvolving Bacterial Recombination Signatures, Chromosomal Architecture, and Natural Selection

This document provides the formal mathematical foundations for decomposing bacterial recombination into biophysical delivery vehicles, integration mechanisms, and evolutionary selective sieves.

---

## 1. Input Space and Dual-Manifold Representation

Let the bacterial population cohort consist of $N$ sequenced isolates with a circular reference chromosome of length $L_{\text{chr}}$ (typically $1.8\text{--}5.5\text{ Mb}$):
$$s \in [1, L_{\text{chr}}]$$

Rather than discarding the accessory genome to construct an artificially gapless alignment, RhizAeon models the bacterial chromosome as a **dual-manifold system**:

1. **The Core Sequence Tensor $\mathbf{X} \in \{A, C, G, T, -\}^{N \times L}$:**
   The base-by-base homologous alignment spanning both conserved core genes and polymorphic intervals.
2. **The Structural Gap State Matrix $\mathbf{G} \in [0, 1]^{N \times L}$:**
   $$G_{i}(s) = \frac{1}{2w + 1} \sum_{u = s - w}^{s + w} \mathbb{I}(X_{iu} = \text{'-'})$$
   representing the localized deletion or non-homologous insertion status of isolate $i$ in window $w$.

---

## 2. The 5-Tuple Recombination Signature Vector

For any detected recombination tract $e$ with chromosomal boundaries $[a_e, b_e]$ in isolate $R$, we extract the **5-tuple signature vector**:
$$\mathbf{\Sigma}(e) = \langle L_e, \quad \nabla g_e, \quad \mathcal{P}_{\perp}(e), \quad \mathcal{M}_{\text{flank}}(e), \quad \Delta \mathbf{S}_e \rangle$$

### 2.1 Tract Length $L_e$
$$L_e = b_e - a_e + 1$$
* **Transformation:** Follows a continuous exponential distribution $f(L) = \frac{1}{\delta} e^{-L/\delta}$, where $\delta \approx 2\text{--}8\text{ kb}$.
* **Generalized Transduction:** Strictly upper-bounded by phage capsid headful volume: $L \le L_{\text{capsid}} \approx 40\text{--}100\text{ kb}$.
* **Conjugative ICE / Pathogenicity Islands:** Modular, multi-gene blocks: $L \ge 20\text{ kb}$ (often $40\text{--}120\text{ kb}$).
* **Transposons / Insertion Sequences:** Compact discrete sizes: $L \approx 800\text{--}3{,}500\text{ bp}$.

### 2.2 Synteny Gap Gradient $\nabla g_e$
$$\nabla g_e = \max\left( |G_R(a_e + \delta) - G_R(a_e - \delta)|, \; |G_R(b_e + \delta) - G_R(b_e - \delta)| \right)$$
* **Allelic Swaps (Homologous Recombination):** $\nabla g_e \approx 0.00$. Length and chromosomal synteny are preserved.
* **Accessory Additions / Deletions (Non-Homologous Recombination):** $\nabla g_e \ge 0.50$ (typically $\to 1.00$).

### 2.3 Orthogonal Subspace Departure $\mathcal{P}_{\perp}(e)$
Let $\mathbf{V}_k \in \mathbb{R}^{N \times k}$ represent the top $k$ spectral eigenvectors of the centered core Gram matrix $B_{\text{core}}$. For the tract $e$, the isolate's low-pass embedding is $\mathbf{y}_R(e) \in \mathbb{R}^k$.
$$\mathcal{P}_{\perp}(e) = \|(I - \mathbf{V}_k \mathbf{V}_k^T) \mathbf{y}_R(e)\|_2$$
* **Core Allelic Replacement:** $\mathcal{P}_{\perp}(e) \approx 0$. The sequence migrates within the existing core phylogenetic manifold.
* **Novel Accessory Introgression:** $\mathcal{P}_{\perp}(e) \gg 0$. The sequence departs from the linear subspace spanned by core clades into an orthogonal mobilome dimension.

### 2.4 Flanking Motif Alignment $\mathcal{M}_{\text{flank}}(e)$
Evaluates whether the physical breakpoint boundaries coincide with known site-specific recombinase or RecBCD motifs:
$$\mathcal{M}_{\text{flank}}(e) = \max\left( \mathcal{S}_{\chi}(a_e, b_e), \; \mathcal{S}_{\text{att}}(a_e, b_e), \; \mathcal{S}_{\text{TIR}}(a_e, b_e) \right)$$
1. **Chi ($\chi$) Octamer Score:**
   $$\mathcal{S}_{\chi}(s) = \sum_{m \in \text{Chi motifs}} \mathbb{I}(X[s - 50 : s + 50] \text{ matches } m)$$
2. **Attachment Site Direct Repeat Score (*attL* / *attR*):**
   $$\mathcal{S}_{\text{att}}(s) = \max_{k \in [10, 45]} \operatorname{Score}(\text{Local direct repeat matching 3' tRNA or att consensus})$$
3. **Terminal Inverted Repeat (TIR) Score:**
   $$\mathcal{S}_{\text{TIR}}(s) = \max_{k \in [12, 35]} \operatorname{Score}(\text{Inverted repeat characteristic of transposons})$$

### 2.5 Directional Attention Drift $\Delta \mathbf{S}_e$
Derived from Tier 2 PhyloAxialTransformer:
$$\Delta \mathbf{S}_e = \|\mathbf{S}_{a_e - w}[R, :] - \mathbf{S}_{a_e + w}[R, :]\|_2$$
Isolates the specific recombinant lineage from non-recombinant background genomes with single-base/codon precision.

---

## 3. Mathematical Classification Engine

The 5-way signature classification function maps each event $e$ to its biological mechanism:

$$\operatorname{Class}(e) = \begin{cases}
\textbf{Homologous Conversion (Transformation)} & \text{if } \nabla g_e < 0.15 \;\land\; \mathcal{P}_{\perp}(e) < \tau_{\perp} \;\land\; L_e \le 15\text{ kb} \;\land\; \text{L-PIR}_e \ge 0.50 \\
\textbf{Micro-Conversion} & \text{if } \nabla g_e < 0.15 \;\land\; L_e < 200\text{ bp} \;\land\; \Delta \mathbf{S}_e \ge 0.12 \\
\textbf{Specialized Transduction (Prophage)} & \text{if } \nabla g_e \ge 0.50 \;\land\; \mathcal{S}_{\text{att}}(e) \ge \tau_{\text{att}} \;\land\; 25\text{ kb} \le L_e \le 55\text{ kb} \\
\textbf{Generalized Transduction} & \text{if } \nabla g_e < 0.15 \;\land\; \mathcal{P}_{\perp}(e) < \tau_{\perp} \;\land\; 15\text{ kb} < L_e \le L_{\text{capsid}} \\
\textbf{Conjugative Mega-Island (ICE)} & \text{if } \nabla g_e \ge 0.50 \;\land\; L_e \ge 20\text{ kb} \;\land\; \mathcal{P}_{\perp}(e) \ge \tau_{\perp} \\
\textbf{Transposition / IS Insertion} & \text{if } \nabla g_e \ge 0.50 \;\land\; L_e < 5\text{ kb} \;\land\; \mathcal{S}_{\text{TIR}}(e) \ge \tau_{\text{TIR}}
\end{cases}$$

---

## 4. Chromosomal Polar Coordinate System and Spatial Mapping

Bacterial chromosomes are circular, organized around two replication forks initiating at $oriC$ and terminating at $ter$. Every chromosomal coordinate $s$ is mapped onto biophysical features:

### 4.1 Replication Polar Coordinates
Let $s_{oriC}$ and $s_{ter}$ denote the coordinates of the replication origin and terminus:
$$\theta_{\text{rep}}(s) = 2\pi \cdot \frac{s - s_{oriC}}{L_{\text{chr}}} \in [0, 2\pi)$$
$$d_{\text{ori}}(s) = \min(|s - s_{oriC}|, \; L_{\text{chr}} - |s - s_{oriC}|)$$
$$d_{\text{ter}}(s) = \min(|s - s_{ter}|, \; L_{\text{chr}} - |s - s_{ter}|)$$

### 4.2 Replication Fork Polarity and GC Skew
$$\operatorname{Skew}(s) = \frac{G(s \pm w) - C(s \pm w)}{G(s \pm w) + C(s \pm w)}$$
* $\operatorname{Skew}(s) > 0$: Leading replication strand.
* $\operatorname{Skew}(s) < 0$: Lagging replication strand.
Head-on collisions between RNA polymerase and the DNA replication fork occur preferentially on the lagging strand, triggering fork arrest and break-induced homologous recombination.

### 4.3 Continuous Chi ($\chi$) Density Field
Let $M_{\chi}$ denote the set of species-specific Chi motif instances on both forward and reverse strands:
$$\mathcal{D}_{\chi}(s) = \frac{1}{\sigma_{\chi} \sqrt{2\pi}} \sum_{m \in M_{\chi}} \exp\left( -\frac{(s - s_m)^2}{2\sigma_{\chi}^2} \right)$$
Chi sites modulate the exonuclease activity of RecBCD, dictating where RecA filaments nucleate.

---

## 5. Deconvolving Mechanism from Selection: The "Why" Formulation

The observed rate of recombination along the chromosome is modeled as a non-homogeneous Poisson point process:
$$\lambda_{\text{observed}}(s) = \lambda_{\text{mechanistic}}(s) \times \mathcal{S}_{\text{selective}}(s)$$

### 5.1 The Biophysical Mechanistic Delivery Prior $\lambda_{\text{mech}}(s)$
Estimated using generalized additive models (GAM) or Poisson GLM from purely sequence-based biophysical features:
$$\ln \lambda_{\text{mech}}(s) = \beta_0 + f_1(d_{\text{ori}}(s)) + \beta_2 \mathcal{D}_{\chi}(s) + \beta_3 \mathbb{I}(s \in \text{tRNA/att}) + \beta_4 |\operatorname{Skew}(s)|$$

### 5.2 The Evolutionary Selective Sieve $\mathcal{S}_{\text{sel}}(s)$
$$\mathcal{S}_{\text{sel}}(s) = \frac{\lambda_{\text{observed}}(s)}{\lambda_{\text{mech}}(s)}$$

$$\log \mathcal{S}_{\text{sel}}(s) \begin{cases}
< -\tau_{\text{sel}} & \textbf{Recombination Desert (Strong Purifying Selection):} \\
& \text{Essential stoichiometric core complexes (ribosomes, ATP synthase). Recombinants are lethal.} \\
\approx 0 & \textbf{Neutral Recombination:} \\
& \text{Recombination matches biophysical integration expectations under neutral genetic drift.} \\
> +\tau_{\text{sel}} & \textbf{Adaptive Hotspot (Positive / Diversifying Selection):} \\
& \text{Recombinant alleles are actively enriched and fixed (AMR, capsule switches, antiphage defense).}
\end{cases}$$

### 5.3 Decoupling Convergent Adaptation via Tier 2 $dN/dS$ Concordance
To guarantee that high $\mathcal{S}_{\text{sel}}(s)$ reflects authentic chromosomal sequence transfer rather than convergent point mutations, Tier 2 calculates the Reticulation Concordance Index:
$$\rho_{\text{retic}}(s) = \frac{2 \langle \Delta_{dS}(s), \; \Delta_{dN}(s) \rangle}{\|\Delta_{dS}(s)\|_2^2 + \|\Delta_{dN}(s)\|_2^2 + \epsilon}$$
* **Authentic Adaptive Recombination:** $\rho_{\text{retic}}(s) \ge 0.70$ (both neutral clock and coding mutations jump in lockstep).
* **Convergent Point Mutation Mimicry:** $\rho_{\text{retic}}(s) < 0.25$ (non-synonymous mutations spike without synonymous clock support $\implies$ rejected).
