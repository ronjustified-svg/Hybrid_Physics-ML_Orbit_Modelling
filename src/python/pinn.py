# =============================================================================
#  Physics-Informed Neural Network (PINN) — Earth Orbit  [PyTorch]
# =============================================================================
#
#  PURPOSE:
#    Model Earth's orbital trajectory using a PINN that penalises violations
#    of Newtonian gravity and angular momentum conservation, in addition to
#    fitting the NASA Horizons position data.
#
#  APPROACH:
#    - Input:  time t, normalised to [-1, 1]
#    - Output: 2D position (x, y) in AU
#    - Loss:   L_data  +  λ₁·L_Newton  +  λ₂·L_AngularMomentum
#    - Optimiser: Adam (lr = 1e-3)
#
#  ARCHITECTURE:
#    t (1)  >  Linear(64) + Tanh  →  Linear(64) + Tanh  >  (x, y) (2)
#
#  PHYSICS LOSSES (via torch.autograd):
#    L_Newton     — enforces d²r/dt² = -μ·r / |r|³
#                   2nd derivatives computed via autograd.grad with
#                   create_graph=True to keep the computation graph alive
#    L_AngularMom — enforces h = x·vy - y·vx = h0 (conserved scalar)
#                   velocities recovered from position output via autograd
#
#  ADAPTIVE LOSS WEIGHTING:
#    λ₁ and λ₂ are ramped linearly from 0 to target values between
#    epochs RAMP_START and RAMP_END. The network first builds a data fit
#    without physics interference, then constraints are gradually introduced.
#    Set LAMBDA_PHYS = LAMBDA_ANG = 0 to run as a pure NN baseline.
#
#    LIMITATIONS:
#    Even with careful weighting, the PINN struggles on this problem because:
#    (1) The Newton loss enforces pure two-body gravity, but NASA data
#        includes perturbations from other planets — the two objectives
#        are fundamentally in tension.
#    (2) PhysRes (~10²–10³) and DataLoss (~10⁻⁴) operate on vastly
#        different scales, making gradient balancing difficult.
#    These limitations motivated the discrepancy modelling approach in
#    MATLAB (Dmodel.m), which uses physics as a structural prior rather
#    than a soft penalty — sidestepping this tension entirely.
#
#  DATA:
#    NASA JPL Horizons — Earth ephemeris, 2023–2024
#    ~365 daily observations, heliocentric ecliptic J2000 frame
#
#  REQUIREMENTS:
#    Python 3.9+, torch, numpy, matplotlib
#    horizons_results.txt and horizons_io.py must be in the same directory
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from horizons_io import load_horizons_vectors_2d

# =============================================================================
#  CONFIG
# =============================================================================
HIDDEN       = 64      # neurons per hidden layer
NUM_EPOCHS   = 2000    # total training epochs
LEARN_RATE   = 1e-3    # Adam learning rate
COLLOC_PTS   = 800     # collocation points sampled per epoch for physics loss
PRINT_EVERY  = 500     # logging interval (epochs)

LAMBDA_PHYS  = 1e-4    # target weight for Newton loss      (set 0 to disable)
LAMBDA_ANG   = 1e-4    # target weight for ang. mom. loss   (set 0 to disable)
RAMP_START   = 1    # epoch at which physics weights begin ramping up
RAMP_END     = 2000    # epoch at which physics weights reach target value
# =============================================================================

# ========================== DATA =============================================
data = load_horizons_vectors_2d("horizons_results.txt")

DAY = 86400.0
AU  = 149597870.7
MU  = 1.32712440018e11 * (DAY**2) / (AU**3)   # GM in AU³/day²

t_min = float(data.t_days.min())
t_max = float(data.t_days.max())

# Normalise time to [-1, 1] — keeps tanh activations in their sensitive range
t_norm    = 2.0 * (data.t_days - t_min) / (t_max - t_min) - 1.0
t_tensor  = torch.tensor(t_norm, dtype=torch.float32).view(-1, 1)
xy_tensor = torch.tensor(
    np.stack([data.x_au, data.y_au], axis=1), dtype=torch.float32
)

# Initial conditions
x0  = float(data.x_au[0])
y0  = float(data.y_au[0])
vx0 = float(data.vx0_au_per_day)
vy0 = float(data.vy0_au_per_day)
h0  = x0 * vy0 - y0 * vx0   # specific angular momentum (AU²/day) — conserved

print("=" * 60)
print(f"  Observations : {len(data.t_days)}")
print(f"  t range      : {t_min:.1f} – {t_max:.1f} days")
print(f"  x0, y0 (AU)  : ({x0:.4f}, {y0:.4f})")
print(f"  h0 (AU²/day) : {h0:.6f}")
print(f"  λ_Newton     : {LAMBDA_PHYS}  (ramp {RAMP_START}→{RAMP_END})")
print(f"  λ_AngMom     : {LAMBDA_ANG}   (ramp {RAMP_START}→{RAMP_END})")
print("=" * 60)

# ========================== NETWORK ==========================================
class OrbitNet(nn.Module):
    """
    Feedforward MLP: t → (x, y)
    Tanh activations: smooth, bounded, suited to continuous periodic signals.
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)

model = OrbitNet(hidden=HIDDEN)

# ========================== AUTOGRAD HELPERS =================================
def d_dt(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """First derivative dy/dt via autograd."""
    return torch.autograd.grad(
        y, t,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]

def d2_dt2(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Second derivative d²y/dt² — two autograd passes."""
    return d_dt(d_dt(y, t), t)

# ========================== LOSS FUNCTIONS ===================================
def loss_data(t: torch.Tensor, xy_true: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and observed positions."""
    return torch.mean((model(t) - xy_true) ** 2)


def loss_newton(t_col: torch.Tensor) -> torch.Tensor:
    """
    Newton's law residual at collocation points.
    Enforces: d²r/dt² = -μ·r / |r|³
    t_col does not need grad — clone is made internally.
    """
    t = t_col.clone().detach().requires_grad_(True)
    xy   = model(t)
    x, y = xy[:, 0:1], xy[:, 1:2]

    x_tt = d2_dt2(x, t)
    y_tt = d2_dt2(y, t)

    r   = torch.sqrt(x**2 + y**2 + 1e-12)    # +eps avoids divide-by-zero
    res = (x_tt + MU * x / r**3)**2 + (y_tt + MU * y / r**3)**2
    return torch.mean(res)


def loss_ang_mom(t_col: torch.Tensor) -> torch.Tensor:
    """
    Angular momentum conservation at collocation points.
    Enforces: h = x·vy - y·vx = h0
    """
    t = t_col.clone().detach().requires_grad_(True)
    xy   = model(t)
    x, y = xy[:, 0:1], xy[:, 1:2]

    vx = d_dt(x, t)
    vy = d_dt(y, t)
    h  = x * vy - y * vx

    return torch.mean((h - h0) ** 2)


def sample_collocation(n: int) -> torch.Tensor:
    """Randomly sample n collocation points, normalised to [-1, 1]."""
    t_raw = np.random.uniform(t_min, t_max, size=(n, 1))
    t_n   = 2.0 * (t_raw - t_min) / (t_max - t_min) - 1.0
    return torch.tensor(t_n, dtype=torch.float32)


def ramp(epoch: int, target: float) -> float:
    """Linearly ramp a weight from 0 → target between RAMP_START and RAMP_END."""
    if epoch < RAMP_START:
        return 0.0
    progress = (epoch - RAMP_START) / max(1, RAMP_END - RAMP_START)
    return target * min(1.0, max(0.0, progress))

# ========================== TRAINING =========================================
optimiser = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

history = {"total": [], "data": [], "newton": [], "ang": []}

print("\nTraining...\n")
for epoch in range(1, NUM_EPOCHS + 1):
    optimiser.zero_grad()

    lam_phys = ramp(epoch, LAMBDA_PHYS)
    lam_ang  = ramp(epoch, LAMBDA_ANG)

    L_data = loss_data(t_tensor, xy_tensor)

    # Only compute expensive physics losses when weights are non-zero
    t_col    = sample_collocation(COLLOC_PTS)
    L_newton = loss_newton(t_col)  if lam_phys > 0 else torch.tensor(0.0)
    L_ang    = loss_ang_mom(t_col) if lam_ang  > 0 else torch.tensor(0.0)

    loss = L_data + lam_phys * L_newton + lam_ang * L_ang
    loss.backward()
    optimiser.step()

    history["total"].append(loss.item())
    history["data"].append(L_data.item())
    history["newton"].append(L_newton.item())
    history["ang"].append(L_ang.item())

    if epoch % PRINT_EVERY == 0:
        print(f"Epoch {epoch:5d}/{NUM_EPOCHS} | "
              f"Total {loss.item():.3e} | "
              f"Data {L_data.item():.3e} | "
              f"Newton {L_newton.item():.3e} | "
              f"AngMom {L_ang.item():.3e} | "
              f"λ={lam_phys:.1e}")

# ========================== EVALUATION =======================================
model.eval()
with torch.no_grad():
    pred = model(t_tensor).numpy()

x_pred = pred[:, 0]
y_pred = pred[:, 1]

rmse = np.sqrt(np.mean((x_pred - data.x_au)**2 + (y_pred - data.y_au)**2))
print(f"\nRMSE: {rmse:.4e} AU")

# ========================== PLOTS ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Orbit comparison
ax = axes[0]
ax.plot(data.x_au, data.y_au, 'bx', markersize=3,   label='NASA Data')
ax.plot(x_pred,    y_pred,    'r--', linewidth=1.5,  label='PINN Predicted')
ax.plot(0, 0,                 'y*',  markersize=12,   label='Sun')
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
ax.set_xlabel('X [AU]')
ax.set_ylabel('Y [AU]')
ax.set_title(f'Earth Orbit: PINN  (RMSE = {rmse:.2e} AU)')

# Loss history — log scale reveals the ramp effect clearly
ax = axes[1]
epochs = range(1, NUM_EPOCHS + 1)
ax.semilogy(epochs, history["total"],  label='Total Loss',       linewidth=1.5)
ax.semilogy(epochs, history["data"],   label='Data Loss',        linewidth=1.5)
ax.semilogy(epochs, history["newton"], label='Newton Residual',  linewidth=1.0, alpha=0.7)
ax.semilogy(epochs, history["ang"],    label='Ang. Momentum',    linewidth=1.0, alpha=0.7)
ax.axvline(RAMP_START, color='gray', linestyle='--', alpha=0.6, label=f'Ramp start (ep. {RAMP_START})')
ax.axvline(RAMP_END,   color='gray', linestyle=':',  alpha=0.6, label=f'Ramp end (ep. {RAMP_END})')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log scale)')
ax.set_title('Training Loss History')
ax.legend(fontsize=8)
ax.grid(True)

plt.tight_layout()
plt.show()
