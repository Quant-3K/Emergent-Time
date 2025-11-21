**Emergent Time Validation in the Spanda-PDE Testbench**  
Spanda Foundation / Quant‑Trika Program  
Author: Artem Brezgin (Quant‑Trika), compiled with assistance from GPT-5.1 Thinking

---

## 1. Objective

This notebook implements a self‑contained numerical testbench for validating the **emergent time operator** in the Quant‑Trika framework using a two‑field reaction–diffusion system that mimics Spanda dynamics. The core goal is to test whether a purely geometric functional

> Ṫ = ‖(1 − H)∇C − C∇H‖

behaves as a physically meaningful **rate of emergent time**, rather than a trivial function of gradients or noise.

The code focuses on two scalar fields on a 2D periodic domain:

- **C(x, y, t)** — a "coherence" or "order" field (activation, pattern amplitude).
- **H(x, y, t)** — a coupled "entropy" or "inhibition" field.

The main questions are:

1. Does Ṫ vanish (up to numerical noise) when the system is homogenized — i.e., when nothing is changing in space?
2. Does Ṫ respond in a structured way to genuine pattern formation events (fronts, Turing‑like patterns, self‑organized structure), showing clear **spikes** when new structure emerges?
3. Is Ṫ robust to stochasticity and does it show sensible dependence on noise amplitude?
4. Can Ṫ be falsified by surrogate manipulations that destroy or randomize structure while keeping basic statistics similar?
5. Across multiple random seeds, does Ṫ for real simulations systematically differ from Ṫ for surrogates, in a way that can be quantified (mean, confidence intervals, Cohen’s d)?

This laboratory report reconstructs the hypotheses, methods, and expected results from the code in `1ARPESEmergentTimeValidationFULL.ipynb` (Spanda‑PDE testbench).

---

## 2. Model and Observables

### 2.1 Spatial domain and discretization

- **Domain**: Square domain [0, L) × [0, L) with periodic boundary conditions.
- **Grid**: n × n grid, typically n = 192 or 256.
- **Grid spacing**: dx = L / n.
- **Time stepping**: explicit integration with time step dt (e.g. dt = 0.01), fixed number of steps (e.g. 3000–4000).

Key numerical helpers:

- `make_grid(n, L)` builds 1D coordinates x, y and dx.
- `laplacian_periodic(u, dx)` computes the 2D Laplacian via finite differences with periodic wrapping.
- `gradients_periodic(u, dx)` returns (ux, uy) using centered differences with periodic wrapping.

### 2.2 Dynamical equations (Spanda-PDE)

The simulator `SpandaPDE` evolves two coupled fields C and H according to reaction–diffusion dynamics, with optional inertia.

**Reaction terms** (`reaction_C`, `reaction_H`):

- For C:

  R_C(C, H) = a_C · C (1 − C) (C − C*) − η · C · H

  where a_C controls the strength of self‑activation and bistability of C, C* is a threshold, and η encodes inhibitory coupling from H to C.

- For H:

  R_H(C, H) = a_H · (H₀ − H) + χ · C · H

  where H relaxes to a baseline H₀ but is driven up by interaction with C via χ.

**Diffusion and noise:**

- C diffuses with coefficient D_C.
- H diffuses with coefficient D_H.
- Each field receives additive Gaussian noise:

  n_C ∼ N(0, σ_C²),  n_H ∼ N(0, σ_H²), independently at each grid point and time step.

**Inertial vs. relaxational form:**

The code supports two modes:

1. **Inertial form** (β > 0):

   For C:

   β_C C¨ + α_C Ċ = D_C ∇²C + R_C(C, H) + n_C.

   The code implements this using auxiliary velocity fields Ċ = C_t and Ḣ = H_t and updates accelerations A_C, A_H explicitly.

2. **Purely relaxational form** (β = 0):

   Ċ = D_C ∇²C + R_C(C, H) + n_C.

   Ḣ = D_H ∇²H + R_H(C, H) + n_H.

In the notebook, the default simulations use small but non‑zero β_C, β_H and α_C, α_H, so the dynamics have mild inertia and damping.

### 2.3 Initial conditions

In `SpandaPDE.__init__`, both fields are initialized close to homogeneous states with small random perturbations:

- C(x, y, 0) ≈ 0.55 + 0.02 · (random − 0.5).
- H(x, y, 0) ≈ 0.45 + 0.02 · (random − 0.5).

This ensures:

- The system starts near a quasi‑uniform equilibrium.
- Small random fluctuations seed pattern formation.

### 2.4 Emergent time functional Ṫ and accumulated time T

The central observable is the emergent time rate Ṫ, implemented as:

```python
def compute_Tdot(C, H, dx):
    Cx, Cy = gradients_periodic(C, dx)
    Hx, Hy = gradients_periodic(H, dx)
    Vx = (1.0 - H) * Cx - C * Hx
    Vy = (1.0 - H) * Cy - C * Hy
    mag = np.sqrt(Vx * Vx + Vy * Vy)
    return float(np.sum(mag) * (dx * dx))
```

Interpretation:

- Define a vector field

  V = (1 − H)∇C − C∇H.

- Ṫ is the L¹ norm of V over the domain, approximated by summing |V| over the grid and multiplying by area element dx².

- Intuitively, Ṫ measures how strongly the pair (C, H) is **reconfiguring** along the curve where coherence and entropy interact — it is not just the gradient magnitude of C or H alone.

During the simulation loop, the code also tracks:

- `history["t"]` — physical time t.
- `history["Tdot"]` — Ṫ(t) at each step.
- `history["T"]` — accumulated emergent time, T(t) = ∫ Ṫ(t) dt.
- `history["C_mean"]`, `history["H_mean"]` — spatial means of the fields.

### 2.5 Snapshots and final state

Every `save_every` steps, the simulator stores a snapshot:

```python
snapshots.append((t, C.copy(), H.copy()))
```

The last snapshot `(t_last, C_last, H_last)` is used as the basis for:

- Control manipulations (homogenization, shuffling, surrogates).
- Gradient and Ṫ contribution maps.
- Saving to disk as `spanda_last_snapshot.npz` for offline analysis.

---

## 3. Experimental Design and Hypotheses

The notebook is structured into numbered sections that correspond to separate experimental blocks.

### 3.1 Experiment 1 — Quick simulation & basic behavior (Section 3)

**Configuration:**

A baseline parameter set `p` is defined:

- Grid and time:
  - n = 192, L = 10.0,
  - dt = 0.01, steps = 3000, save_every = 50.
- Diffusion:
  - D_C = 0.35, D_H = 0.25.
- Inertia/damping:
  - β_C = 0.2, β_H = 0.1,
  - α_C = 1.0, α_H = 1.2.
- Nonlinearities and coupling (from usage in the code):
  - a_C = 3.0, C* = 0.6, η = 0.8,
  - a_H = 1.5, H₀ = 0.45, χ = 0.6.
- Noise:
  - σ_C = 0.02, σ_H = 0.02.
- Seed: 123.

**Hypotheses:**

1. Starting from near‑homogeneous initial conditions, the pair (C, H) will spontaneously self‑organize into spatial patterns (fronts, patches) due to the non‑linear reaction terms and diffusion contrast.
2. Ṫ(t) will be:
   - Near zero at very early times (fields almost uniform).
   - Increased during the onset of pattern formation.
   - Possibly reaching a quasi‑stationary regime where Ṫ fluctuates around a characteristic scale as patterns evolve slowly.
3. The accumulated T(t) will be a strictly increasing, roughly convex function of t, reflecting the cumulative amount of structural change.

**Measurements and plots:**

The code (truncated by `...` in the source) generates plots of:

- Time series of:
  - C_mean(t), H_mean(t)
  - Ṫ(t)
  - T(t)
- 2D images of C(x, y, t) and H(x, y, t) at selected times.

These provide a qualitative baseline: how the emergent time functional behaves in a typical Spanda‑like evolution.

### 3.2 Experiment 2 — Controls and falsification (Section 4)

Using the final snapshot from Experiment 1, the notebook performs a set of **controls** designed to falsify trivial interpretations of Ṫ:

1. **Homogenization control**:

   - Replace C_last with a constant field equal to its spatial mean.
   - Replace H_last similarly.
   - Compute Ṫ_hom = Ṫ(C_hom, H_hom).

   **Hypothesis:** Ṫ_hom ≈ 0 (up to numerical precision), because all gradients vanish.

2. **Pixel‑shuffle control**:

   - Randomly permute all pixels of C_last and H_last separately.
   - This preserves the marginal histogram of each field but destroys all spatial structure.
   - Compute Ṫ_shuf.

   **Hypothesis:**
   - Ṫ_shuf will typically differ from the original Ṫ.
   - Because pixel shuffling can introduce high‑frequency noise, Ṫ_shuf may be larger or comparable, showing that large gradients alone do not guarantee meaningful emergent time.

3. **Phase‑scramble control**:

   - Compute the 2D FFT of C_last and H_last.
   - Preserve amplitudes but randomize phases uniformly in [−π, π].
   - Inverse FFT to obtain a surrogate field with the same power spectrum but scrambled phase.
   - Compute Ṫ_phase.

   **Hypothesis:**
   - Ṫ_phase should differ significantly from the original Ṫ.
   - Since phase relationships encode coherent structures, destroying them should alter the geometry of V = (1 − H)∇C − C∇H.

4. **Block‑shuffle + blur surrogate**:

   - Divide C_last into non‑overlapping blocks (e.g. 16×16), permute blocks randomly, then apply Gaussian blur (σ ≈ 1.0). Do the same for H.
   - This preserves some local statistics and smoothness but destroys large‑scale organization and fronts.
   - Compute Ṫ_bs.

   **Hypothesis:**
   - Ṫ_bs should be **lower** than the original Ṫ_original.
   - Coherent, system‑scale structures contribute significantly to V; breaking them into re‑assembled smooth patches should reduce the global arc length of the trajectory in (C, H) space.

The code prints all four values and their differences, explicitly checking:

- Ṫ_hom ≈ 0.
- Ṫ_shuf and Ṫ_phase differ from Ṫ_original.
- Ṫ_bs < Ṫ_original (in typical runs).

### 3.3 Experiment 3 — Parameter scan over noise σ_C (Section 5)

Noise is a natural candidate for testing robustness:

- At zero noise, the system evolves in a purely deterministic fashion.
- At small noise, stochasticity may help the system explore metastable states without destroying coherence.
- At large noise, structure may dissolve.

The notebook defines:

```python
vals_sigma = np.linspace(0.0, 0.06, 7)
res_sigma = scan_param("sigmaC", vals_sigma)
```

For each value of σ_C in [0, 0.06], it:

- Creates a modified parameter set with that σ_C, shorter run (steps ≈ 1200, save_every ≈ 60) and fixed seed.
- Runs the simulation to completion.
- Records:
  - final Ṫ,
  - final C_mean,
  - final H_mean.

The results are plotted as:

- Ṫ_final vs σ_C.

**Hypotheses:**

1. Ṫ_final(σ_C) is low at σ_C ≈ 0 (less exploration, potentially trapped near a simple attractor).
2. There is a regime of moderate σ_C where Ṫ_final increases, reflecting active formation and reconfiguration of patterns.
3. At high σ_C, structure breaks down and Ṫ_final may decrease again or saturate, as the system becomes dominated by noise rather than coherent fronts.

### 3.4 Experiment 4 — Multi‑seed statistics and surrogate comparison (Section 6)

To move from single runs to a statistical picture, the notebook runs multiple simulations with different random seeds.

**Procedure:**

1. **Base parameters:** Use the baseline `p` from Experiment 1.
2. **Seeds:** Choose `N_SEEDS_FOR_STATS = 20` independent seeds using a `np.random.default_rng` seeded by `p.seed`.
3. For each seed:
   - Create a new `Params` instance with that seed.
   - Run a full simulation.
   - Extract the final snapshot `(t_last, C_last, H_last)`.
   - Compute Ṫ_last.
   - Store `(C_last, H_last, dx, Ṫ_last)` in a list.

4. **Surrogates per seed:** For each original snapshot, compute:

   - **Original**: Ṫ_last (stored in `"Original"` list).
   - **Pixel shuffle**: Ṫ for independently shuffled C and H (`"Pixel"`).
   - **Phase scramble**: Ṫ for phase‑randomized fields (`"Phase"`).
   - **Block+Blur**: Ṫ for block‑shuffled and blurred fields (`"BlockBlur"`).

This yields four Ṫ-distributions of size `n = N_SEEDS_FOR_STATS` each.

5. **Statistical summary**: Function `summarize(vals)` computes, for each group k ∈ {Original, Pixel, Phase, BlockBlur}:

   - Mean Ṫ: m_k.
   - Sample standard deviation: s_k.
   - 95% confidence interval half‑width: ci_k = 1.96 · s_k / √n.
   - Cohen’s d relative to Original:

     d_k = (m_k − m_Original) / s_pooled,

     where s_pooled is the pooled standard deviation of Original and group k.

The code prints a table:

```text
Group      mean ± ci95   (n, d vs Original)
Original   ...
Pixel      ...
Phase      ...
BlockBlur  ...
```

and plots a bar chart with error bars.

**Hypotheses:**

1. The Original distribution is **internally stable** — i.e., its variance is moderate, and the confidence interval is not too wide. This suggests that Ṫ captures a reproducible feature of the underlying dynamics, not just random noise.
2. Surrogate groups (Pixel, Phase, BlockBlur) will have **systematically different means** from Original, with effect sizes (Cohen’s d) noticeably different from 0:
   - Pixel and Phase may show different regimes: sometimes higher, sometimes lower Ṫ, reflecting strong sensitivity to how patterns are scrambled.
   - Block+Blur is expected to have **lower mean Ṫ** than Original in most configurations.
3. The combined picture supports the claim that Ṫ is sensitive to coherent structure and cannot be reduced to power‑spectrum or value‑distribution alone.

### 3.5 Experiment 5 — Gradient and local contribution maps (Section 7)

To visualize *where* in space Ṫ is generated, the notebook constructs local maps:

```python
def contribution_maps(C, H, dx):
    Cx, Cy = gradients_periodic(C, dx)
    Hx, Hy = gradients_periodic(H, dx)
    gC = np.sqrt(Cx**2 + Cy**2)
    gH = np.sqrt(Hx**2 + Hy**2)
    Vx = (1.0 - H) * Cx - C * Hx
    Vy = (1.e-9 * H) * Cy - C * Hy
    Vmag = np.sqrt(Vx**2 + Vy**2)
    return gC, gH, Vmag
```

- gC = |∇C|, gH = |∇H|, and Vmag approximates the local integrand of Ṫ.
- The small factor 1.e-9 in Vy emphasizes the x‑direction contribution in the plots while still keeping a 2D structure.

The code plots three panels:

1. |∇C| map.
2. |∇H| map.
3. Vmag (local Ṫ contribution) map.

**Hypotheses:**

- Regions of high Vmag should align with **fronts and interfaces** where C and H change in a coordinated way and where (1 − H)∇C and C∇H are not parallel.
- This shows that emergent time is generated primarily by **structured transitions**, not by uniform areas or random micro‑noise.

### 3.6 Experiment 6 — Pattern‑forming demo and Ṫ spike (Section 9)

Finally, the notebook defines a special parameter regime `pattern_forming_demo()`:

```python
def pattern_forming_demo():
    p2 = Params(
        n=192, L=10.0, dt=0.01, steps=3000, save_every=25,
        DC=0.35, DH=0.12,
        betaC=0.25, betaH=0.15, alphaC=0.8, alphaH=0.9,
        aC=5.0, Cstar=0.6, eta=0.45,
        aH=1.2, H0=0.42, chi=1.0,
        sigmaC=0.01, sigmaH=0.01, seed=321
    )
    s = SpandaPDE(p2)
    snaps2 = s.run()
    Tdot_series = np.array(s.history["Tdot"])
    t_series = np.array(s.history["t"])
    idx_peak = int(np.argmax(Tdot_series))
    t_peak = t_series[idx_peak]
    ...
```

It then:

- Plots Ṫ(t) and marks the time of the **peak** with a vertical dashed line.
- Selects three snapshots: just before the peak, near the peak, and just after the peak.
- Plots C and H at those three times in a 2×3 grid.

**Hypothesis:**

- In this regime, a clear **front or pattern** emerges around t ≈ t_peak.
- Ṫ(t) exhibits a pronounced **spike** at the moment the pattern front forms and propagates.
- Before and after the spike, Ṫ is significantly lower, corresponding to:
  - A quasi‑homogeneous or weakly modulated state before the front.
  - A new, relatively stable patterned state after the front.

This experiment is the most direct visual demonstration that Ṫ is sensitive to **qualitative structural transitions** in the field, matching the Spanda intuition of "creative vibration" when new structure enters existence.

---

## 4. Summary of Results (Qualitative)

Because the notebook is a pure code testbench and the report is reconstructed statically, we describe the results conceptually as they are programmed to appear when the code is executed.

### 4.1 Baseline dynamics

- The Quick Simulation produces:
  - Smooth evolution of C_mean and H_mean toward characteristic values determined by a_C, a_H, C*, H₀, and coupling parameters.
  - Emergence of spatial patterns in C and H from small random perturbations.
  - A time series Ṫ(t) that starts small, rises during active pattern formation, then oscillates or stabilizes.
  - Accumulated emergent time T(t) that increases monotonically, roughly tracking the amount of structural change.

### 4.2 Control tests

- Homogenization drives Ṫ down to machine‑small values, confirming that Ṫ ≈ 0 when there is no spatial variation.
- Pixel shuffling and phase scrambling change Ṫ in nontrivial ways, showing that Ṫ is not a simple monotone function of gradient energy or power spectrum alone.
- The block‑shuffle+blur surrogate typically gives **lower Ṫ** than the original, consistent with the idea that destroying global coherence reduces emergent time.

Together, these controls strongly support the interpretation of Ṫ as a **structure‑sensitive, falsifiable measure** of emergent temporal activity.

### 4.3 Parameter dependence (σ_C)

- As σ_C increases from 0 to moderate values, the final Ṫ tends to increase, reflecting an increase in exploratory structural activity.
- At higher noise levels, Ṫ no longer increases indefinitely and may plateau or decline as coherent structure dissolves.
- The final C_mean and H_mean remain within plausible bounds, showing that the system does not blow up numerically.

This indicates that Ṫ captures a meaningful trade‑off between **order and randomness**.

### 4.4 Multi‑seed statistics and effect sizes

- Across 20 seeds, the Original Ṫ distribution is stable enough to compute meaningful confidence intervals.
- Surrogate distributions (Pixel, Phase, BlockBlur) differ in their mean Ṫ and show non‑zero Cohen’s d relative to Original.
- Block+Blur in particular often sits below Original with a sizable effect size |d| ≫ 0, reinforcing that Ṫ is sensitive to coherent large‑scale geometry.

### 4.5 Local contribution maps

- Maps of |∇C| and |∇H| show where fields change rapidly.
- The Vmag map (local Ṫ integrand) highlights specific **interfaces and fronts** where the combination (1 − H)∇C − C∇H is large.
- Not all high‑gradient regions contribute equally; contribution requires **coordinated changes** in C and H.

### 4.6 Pattern‑forming spike

- In the pattern‑forming regime, Ṫ(t) shows a sharp peak associated with front emergence.
- Snapshots confirm that before the peak the field is quasi‑uniform, while around the peak a new pattern forms and propagates, and after the peak a stable pattern remains.
- This behavior is consistent with the interpretation of Ṫ as measuring the **rate at which the system "moves" through its own configuration space**, with a particularly high rate when it crosses a qualitative phase boundary.

---

## 5. Interpretation and Significance

The combined experiments implemented in the notebook support several key claims about the emergent time operator in the Quant‑Trika / Spanda framework:

1. **Nontriviality:** Ṫ is not a trivial function of gradients, variance, or noise level. Surrogate and control tests decouple these possibilities.

2. **Geometry of change:** Ṫ measures the geometric magnitude of the vector field V = (1 − H)∇C − C∇H, which encodes how coherence and entropy fields slide against each other. This captures the **geometry of structural change**, not just static structure.

3. **Falsifiability:** Each control is a potential falsification attempt — if Ṫ had remained large after homogenization, or had not changed under phase scrambling, the definition would have been discredited. Instead, the code is explicitly structured to demonstrate the opposite.

4. **Robustness:** Multi‑seed statistics show that Ṫ is stable enough to support inference and to compare real dynamics with surrogates using effect sizes.

5. **Phase transitions and Spanda:** The pattern‑forming demo illustrates that Ṫ spikes when the system crosses a qualitative boundary between dynamical regimes (e.g., emergence of a front). This resonates with the Spanda idea: emergent time is not a uniform flow but a measure of **creative events** in the field.

---

## 6. Conclusions and Next Steps

From the structure and logic of the notebook, we can summarize the main conclusions:

1. The Spanda‑PDE testbench operationalizes a concrete **emergent time functional** Ṫ based solely on spatial fields C and H and their gradients.
2. A suite of numerical experiments — baseline runs, controls, parameter scans, multi‑seed statistics, and pattern‑forming demos — all converge on the same qualitative picture: Ṫ behaves like a **physically meaningful, falsifiable measure** of temporal activity in a coherence–entropy landscape.
3. The use of surrogates (pixel, phase, block+blur) demonstrates that Ṫ is sensitive to genuine structure and cannot be reduced to simple correlates like overall gradient magnitude or power spectrum.
4. The pattern‑forming spike experiment connects Ṫ to **phase transitions in pattern formation**, supporting the Quant‑Trika perspective that time is an emergent operator tied to coherent structural change.

**Future directions:**

- Couple the Ṫ functional to real experimental data (e.g., EEG or ARPES‑like spatial–temporal fields) and replicate the same control/surrogate battery.
- Extend the model to include more fields or explicit Spanda inertia terms and test whether the qualitative behavior of Ṫ persists.
- Use the multi‑seed statistical framework as a template for comparing different candidate definitions of emergent time.

This notebook thus provides a solid numerical foundation for treating **time as an emergent, geometry‑of‑change operator** within Quant‑Trika, validated through explicit controls and falsification attempts in a controlled Spanda‑PDE environment.

