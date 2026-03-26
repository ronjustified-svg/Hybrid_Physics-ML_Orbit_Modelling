# Hybrid Physics-ML Orbit Modelling

**Can a small neural network, guided by physics, outperform either physics or data alone?**

This project answers that question by modelling Earth's orbit using NASA ephemeris data, progressing through three increasingly principled approaches, each building on the limitations of the last.

---

## Background & Motivation

This work was developed as research part of my Master's thesis at Budapest University of Technology and Economics (BME-VIK), under the theme of *combining model-based signal processing with AI methods*. The core question is: **how do you most effectively integrate domain knowledge with data-driven learning?**

Earth's orbit is an ideal test case. The physics is well-understood (Newtonian gravity), but a real-world trajectory from NASA contains perturbations from other planets, the Moon, solar oblateness, etc.: things a pure physics model quietly gets wrong. The challenge is to handle that gap intelligently.

The project started in Python for early experimentation, then moved entirely to MATLAB for the physics simulations, neural network training, and final results.

---

## Data

Daily position and velocity data for Earth (2023–2024) fetched from [NASA JPL Horizons](https://ssd.jpl.nasa.gov/horizons/).

```
File: horizons_results.txt
Columns: Julian Date | Date String | X [km] | Y [km] | Z [km] | VX [km/s] | VY [km/s] | VZ [km/s]
~365 observations | Heliocentric ecliptic J2000 frame
```

All models work in AU and AU/year, using only the 2D projection (X, Y) of the orbit.

---

## Approach Evolution

### Stage 1 — Pure Neural Network (`NN_earth_2324.m`)

A simple feedforward network mapping normalized time directly to position.

```
t ∈ [0,1]  →  [FC(60) + tanh]  →  [FC(60) + tanh]  →  (x, y)
```

Training uses MSE on normalized positions, optimized with Adam (lr = 1e-3, 2000 epochs).

**What it reveals:** The network fits the training data well but is entirely unconstrained — it has no concept of gravity, periodicity, or conservation laws. It treats orbital mechanics as an arbitrary curve-fitting problem. This is the baseline that motivates everything that follows.

<p align="center">
  <img src="figs/YOUR_NN_ORBIT_FIGURE.png" alt="Pure NN full orbit" width="700"/>
</p>
<p align="center"><em>Full orbit: NASA data vs pure NN prediction. The fit looks reasonable at a glance but breaks down at the boundaries.</em></p>

<p align="center">
  <img src="figs/YOUR_NN_DIVERGENCE_FIGURE.png" alt="Pure NN endpoint divergence" width="700"/>
</p>
<p align="center"><em>Zoomed view at the orbit endpoints. The predicted trajectory diverges sharply — shooting outward tangentially instead of closing the orbit. This is endpoint divergence: the network maps t ∈ [0,1] to position with no knowledge that the orbit is periodic, i.e. that t = 0 and t = 1 must be the same point in space. At the boundaries, the network only has neighbours on one side, so it extrapolates along the local tangent of the orbit rather than curving back to close it. This is a direct consequence of having no physics — the network simply does not know orbits are closed.</em></p>

---

### Stage 2 — Physics-Informed Neural Network (`pinn_earth_2.m`, `PINN_earth_2324.m`)

Same architecture, but the loss function now penalises violations of two physical laws, computed via **automatic differentiation** through the network:

```
Loss = L_data  +  λ₁ · L_Newton  +  λ₂ · L_AngularMomentum
```

**Newton's law residual** — the network's trajectory must satisfy `d²r/dt² = −GM·r / |r|³`. Second derivatives are computed symbolically using `dlgradient` with `EnableHigherDerivatives`, with careful unit handling via the chain rule (normalized → AU/year²).

**Angular momentum conservation** — `L = x·vy − y·vx` should be constant. Velocities are recovered from the position output via autodiff.

A key design detail: the physics weights **λ₁ and λ₂ are ramped up linearly** between epochs 2000–3500. Starting with λ = 0 lets the network first build a reasonable data fit; only then are the physics constraints gradually tightened. Without this, the physics loss destabilises early training.

**What it reveals:** Physics constraints improve trajectory smoothness and orbital shape. But asking one network to simultaneously fit data, satisfy ODEs, and conserve quantities means competing objectives that are difficult to balance and sensitive to the weight schedule.

**Why physics losses fail here:** The Newton residual operates on a scale of ~10²–10³ while DataLoss is ~10⁻⁴ — a difference of six orders of magnitude. Even a small λ causes the physics gradient to overwhelm the data signal. More fundamentally, the physics loss enforces pure two-body Newtonian gravity, but the NASA data includes perturbations from other planets and the Moon — so the network is simultaneously penalised for correctly fitting the data. This tension cannot be resolved with a soft penalty.

<p align="center">
  <img src="figs/YOUR_PINN_ORBIT_FIGURE.png" alt="PINN full orbit" width="700"/>
</p>
<p align="center"><em>Full orbit: NASA data vs PINN prediction. The physics constraints are present in the loss function but the same endpoint divergence persists — soft penalties cannot enforce periodicity or orbital closure.</em></p>

<p align="center">
  <img src="figs/YOUR_PINN_DIVERGENCE_FIGURE.png" alt="PINN endpoint divergence" width="700"/>
</p>
<p align="center"><em>Zoomed view at the orbit endpoints. The endpoint divergence is still visible, showing that adding physics as a soft loss term is not sufficient to fix the structural limitation of mapping time → position without any notion of periodicity. The network still treats the orbit as an open curve.</em></p>

---

### Stage 3 — Discrepancy Modelling (`Dmodel.m`) ← **Main result**

Instead of making one network do everything, the problem is **decomposed by responsibility**:

```
x_predicted  =  x_Kepler(t)  +  D( state_Kepler )
                     ↑                    ↑
             physics model        small correction NN
           (does the heavy lifting)   (learns what physics misses)
```

**Step 1 — Keplerian baseline.** Integrate Newton's two-body ODE with Earth's actual initial conditions using `ode45` (RelTol=1e-10). This is analytically correct for a pure Sun-Earth system.

**Step 2 — Compute the discrepancy.** `Δ = x_NASA − x_Kepler`. This small residual is what Kepler gets wrong — primarily due to planetary perturbations. Directly differentiating positions to recover accelerations was intentionally avoided: with dt = 1/365 years, second differencing amplifies noise by ~10⁵. Learning position residuals directly is far more numerically stable.

**Step 3 — Train the correction network D.** A compact network learns the mapping:

```
[x_kep, y_kep, vx_kep, vy_kep]  →  [Δx, Δy]
         ↑
   state, not time
```

The input is the Keplerian **state**, not time. This means the correction depends on *where* Earth is in its orbit, not *when* — making it physically meaningful and more generalisable across orbital positions. Both inputs and outputs are normalized to zero mean / unit std before training.

**Step 4 — Final prediction.** `x_full = x_Kepler + D(state_Kepler)`

![Discrepancy Modelling Results](figs/Discrepancy_Modelling.png)
*Top: full orbit comparison (NASA data vs Kepler baseline vs Kepler + D hybrid). Bottom: X and Y position discrepancies — true residual vs learned correction.*

**Why this works better:** Kepler already explains ~99.9% of the variance. The neural network only needs to learn a small, structured residual — a far easier task. The physics model handles the dominant dynamics; the network fills in precisely what physics alone cannot capture. The result is a smaller network (4→32→32→2 vs 1→60→60→2), faster training, and better RMSE.

---

## Results Summary

| Model | Input | Network | Physics | RMSE |
|---|---|---|---|---|
| Pure NN | time | 1→60→60→2 | none | baseline |
| PINN | time | 1→60→60→2 | Newton + Ang. mom. (soft penalty) | improved |
| **Discrepancy Model** | **Kepler state** | **4→32→32→2** | **Kepler ODE (hard constraint)** | **best** |

The discrepancy model achieves the best accuracy with the smallest network — demonstrating that embedding physics as a hard structural constraint (an ODE solver) is more powerful than encoding it as a soft loss penalty.

---

## Key Takeaways

**Physics as structure beats physics as penalty.** The PINN encodes Newton's law as a loss term — it can still be violated. The discrepancy model encodes it as the actual ODE solver — it cannot. Hard constraints are more powerful when the physics is known and trusted.

**Decompose the problem.** Asking one network to explain everything is harder than asking a physics model to explain most of it and a network to explain the rest. The residual is smaller, smoother, and easier to learn.

**Input choice matters.** Using Keplerian state as input (rather than time) makes the correction physically meaningful — it ties the network's output to *where* in the orbit Earth is, not just *when*, which is more robust and interpretable.

**Numerical stability is non-trivial.** Second-differencing positions to recover accelerations amplifies noise by ~1/dt² ≈ 10⁵. Learning position residuals directly, rather than acceleration residuals, was critical to obtaining clean training signal.

---

## Repo Structure

```
├── NN_earth_2324.m       # Stage 1: Pure NN baseline (time → position)
├── pinn_earth_2.m        # Stage 2a: PINN with physics monitoring, λ = 0
├── PINN_earth_2324.m     # Stage 2b: PINN with adaptive physics loss ramp
├── Dmodel.m              # Stage 3: Discrepancy model (Kepler + NN) ← main result
├── horizons_results.txt  # NASA Horizons ephemeris data (Earth, 2023–2024)
└── README.md
```

---

## Requirements

MATLAB R2022b or later with the **Deep Learning Toolbox** (for `dlnetwork`, `dlarray`, `dlfeval`, `adamupdate`, `dlgradient`).

No additional toolboxes required. The Keplerian integration uses the built-in `ode45`.

---

## Running

Ensure `horizons_results.txt` is in your working directory and run any script directly in MATLAB. Start with `Dmodel.m` for the main result.

Diagnostic output (initial conditions check, Kepler RMSE sanity check, discrepancy statistics, final RMSE comparison) prints to the command window. Training progress appears in MATLAB's `trainingProgressMonitor`.
