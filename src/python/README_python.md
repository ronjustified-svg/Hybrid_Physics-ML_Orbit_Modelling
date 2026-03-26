# Python / PyTorch Implementation — Earth Orbit PINN

This folder contains the Python prototype that preceded the MATLAB work. It was built first to explore the PINN approach using PyTorch, and its failure on this problem directly motivated the discrepancy modelling approach in MATLAB.

---

## Why Python First, Then MATLAB?

Python and PyTorch are the standard environment for neural network research — fast prototyping, clean autograd, easy loss composition. The PINN was built here first because it was the natural tool for the job.

The switch to MATLAB happened for one specific reason: the discrepancy model requires a high-accuracy ODE solver (`ode45`) tightly coupled with the neural network training loop. MATLAB's `ode45` with `RelTol=1e-10` gives Keplerian integration precise enough that the residual the network needs to learn is genuinely small and structured. Replicating that in Python (with `scipy.integrate.solve_ivp`) is possible but MATLAB's implementation is more battle-tested for this kind of stiff physical simulation.

The Python code remains as the PINN baseline and proof-of-concept. The MATLAB code is the main result.

---

## Files

```
├── pinn.py           # PINN / pure NN — main script
├── horizons_io.py    # data loader for NASA Horizons format
├── horizons_results.txt  # NASA ephemeris data (Earth 2023–2024)
└── figs/             # saved output figures
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

pip install torch numpy pandas matplotlib
python pinn.py
```

---

## What the Code Does

### `horizons_io.py` — Data Pipeline

Parses the raw NASA Horizons vector export (a messy semi-CSV format with header/footer markers `$$SOE` / `$$EOE`) into a clean `OrbitData2D` dataclass.

**Key concepts used:**
- `@dataclass` for structured data storage — cleaner than a dict, typed, no boilerplate
- `pandas.read_csv` with `io.StringIO` to parse in-memory rather than writing a temp file
- `torch.tensor` with explicit `dtype=torch.float32` — important for GPU compatibility and memory
- Everything is converted to AU and AU/day internally, keeping units consistent throughout

```python
@dataclass
class OrbitData2D:
    t_days:     np.ndarray      # time since first observation (days)
    x_au:       np.ndarray      # X position in AU
    y_au:       np.ndarray      # Y position in AU
    t_tensor:   torch.Tensor    # ready for network input
    xy_tensor:  torch.Tensor    # ready as training target
    ...
```

---

### `pinn.py` — Network, Physics Losses, Training

#### Network Architecture — `OrbitNet`

```python
class OrbitNet(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2)        # output: (x, y)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)
```

**Why `nn.Module`?** Subclassing `nn.Module` gives you `.parameters()` for the optimiser, automatic device management (`.to(device)`), and clean separation of architecture from training logic.

**Why `Tanh` not `ReLU`?** For physics problems, `Tanh` is preferred — it's smooth everywhere (ReLU has a non-differentiable kink at 0), bounded, and its derivatives are well-behaved. The physics losses require 2nd derivatives of the network output — `Tanh` makes this numerically stable; `ReLU` would give zero 2nd derivatives almost everywhere.

**Why normalise time to `[-1, 1]`?** `Tanh` saturates outside `[-2, 2]`. Feeding raw Julian dates (in the thousands) into the network would push all neurons into saturation, killing gradients. Normalising keeps activations in the sensitive linear region of `Tanh`.

---

#### Automatic Differentiation for Physics

This is the core PyTorch concept the PINN depends on. The network outputs position `(x, y)` as a function of time `t`. To enforce Newton's law, you need `d²x/dt²` — the second derivative of the network's output with respect to its input.

PyTorch's autograd computes this exactly (not numerically) by backpropagating through the computation graph:

```python
def d_dt(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y, t,
        grad_outputs=torch.ones_like(y),
        create_graph=True          # keep graph alive for 2nd pass
    )[0]

def d2_dt2(y, t):
    return d_dt(d_dt(y, t), t)    # two passes = 2nd derivative
```

`create_graph=True` is critical — without it, PyTorch discards the computation graph after the first `.grad()` call, making the second derivative impossible to compute.

This is different from the gradients used in training (which are w.r.t. network *weights*). Here we differentiate w.r.t. the network *input* — a less common but powerful use of autograd.

---

#### Physics Loss — Newton's Law

```python
def loss_newton(t_col):
    t = t_col.clone().detach().requires_grad_(True)
    xy = model(t)
    x, y = xy[:, 0:1], xy[:, 1:2]

    x_tt = d2_dt2(x, t)
    y_tt = d2_dt2(y, t)

    r   = torch.sqrt(x**2 + y**2 + 1e-12)
    res = (x_tt + MU * x / r**3)**2 + (y_tt + MU * y / r**3)**2
    return torch.mean(res)
```

**Why `.clone().detach().requires_grad_(True)`?** Collocation points come from a numpy array converted to a tensor — they have no gradient history. We need `t` to be a leaf variable with `requires_grad=True` so autograd can differentiate through it. `.detach()` severs any existing graph; `.requires_grad_(True)` starts tracking from here.

**Why `+1e-12` in `r`?** Prevents division by zero if the network predicts a position near the Sun. Small numerical safeguard.

---

#### Physics Loss — Angular Momentum

```python
def loss_ang_mom(t_col):
    t = t_col.clone().detach().requires_grad_(True)
    xy = model(t)
    x, y = xy[:, 0:1], xy[:, 1:2]
    vx = d_dt(x, t)
    vy = d_dt(y, t)
    h  = x * vy - y * vx          # specific angular momentum
    return torch.mean((h - h0)**2)
```

Velocities `vx`, `vy` are not predicted by the network directly — they are recovered by differentiating the position output with respect to time. This means the network only needs to learn positions; velocities come for free via autograd.

---

#### Collocation Points

```python
def sample_collocation(n: int) -> torch.Tensor:
    t_raw = np.random.uniform(t_min, t_max, size=(n, 1))
    t_n   = 2.0 * (t_raw - t_min) / (t_max - t_min) - 1.0
    return torch.tensor(t_n, dtype=torch.float32)
```

Physics losses are evaluated at **randomly sampled collocation points** — not just at the 365 observation times. This forces the physics constraint to hold everywhere in the time domain, not just where data exists. 800 points are sampled fresh each epoch, giving stochastic coverage of the full trajectory.

---

#### Adaptive Loss Weighting

```python
def ramp(epoch: int, target: float) -> float:
    if epoch < RAMP_START:
        return 0.0
    progress = (epoch - RAMP_START) / max(1, RAMP_END - RAMP_START)
    return target * min(1.0, max(0.0, progress))
```

Physics weights start at 0 and ramp up linearly. The idea: let the network build a good data fit first, then gradually introduce the physics constraints. Starting with physics from epoch 1 destabilises training because the random initialisation violates Newton's law badly, producing enormous gradients.

---

## The PINN Failure — What the Figure Shows

<p align="center">
  <img src="fig/pinnpy.png" alt="PINN failure" width="800"/>
</p>

**Orbit plot (left):** The predicted trajectory visibly diverges from the data — RMSE = 3.86e-02 AU, which is an order of magnitude *worse* than the pure Kepler model (3.34e-03 AU). Adding physics made the result worse.

**Loss history (right):** This explains why. As soon as the physics ramp begins (~epoch 250 here), the Newton residual (green) shoots to ~10² and plateaus there — it never decreases meaningfully. The angular momentum loss (red) similarly plateaus at ~10¹. Total loss (blue) is dominated by these physics terms and rises sharply, while DataLoss (orange) stops improving.

**Root cause — two fundamental tensions:**

1. **Scale mismatch.** Newton residual (~10²) and DataLoss (~10⁻⁴) differ by six orders of magnitude. Even a small `λ` means the physics gradient dominates and prevents data loss from converging.

2. **Contradictory objectives.** The Newton loss enforces pure two-body gravity. But NASA data includes perturbations from Jupiter, the Moon, and other bodies. The network is simultaneously penalised for correctly fitting the data and for violating an idealised physics model that the data doesn't actually satisfy. These two objectives cannot both be minimised.

This is not a bug or a tuning failure — it is a structural limitation of the approach. The PINN encodes physics as a *soft penalty* that can be violated and that conflicts with the data. The discrepancy model in MATLAB sidesteps this entirely by running the physics model first and asking the network to learn only the residual.

---

## PyTorch Concepts Used — Reference

| Concept | Where used | Why it matters |
|---|---|---|
| `nn.Module` | `OrbitNet` | Standard way to define models — gives `.parameters()`, device management, clean API |
| `nn.Sequential` | Network definition | Composing layers without writing a forward pass manually |
| `torch.autograd.grad` | `d_dt`, `d2_dt2` | Differentiating network output w.r.t. input — not weights. Core PINN technique |
| `create_graph=True` | First derivative | Keeps graph alive so 2nd derivative can be computed |
| `requires_grad_(True)` | Collocation points | Makes leaf tensor differentiable — needed for input-space autograd |
| `.clone().detach()` | Physics losses | Severs gradient history before re-attaching — avoids graph contamination |
| `torch.optim.Adam` | Training loop | Adaptive learning rate optimiser — handles varying gradient scales better than SGD |
| `torch.no_grad()` | Evaluation | Disables gradient tracking during inference — saves memory, faster |
| `dtype=torch.float32` | Data loading | Explicit precision — float64 is default in numpy but float32 is standard for neural nets |
| `@dataclass` | `horizons_io.py` | Clean structured data container — typed, no boilerplate, readable |
