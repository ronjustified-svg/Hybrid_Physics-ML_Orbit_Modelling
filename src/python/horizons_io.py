# horizons_io.py
from __future__ import annotations
import matplotlib.pyplot as plt
import io
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import torch

AU_KM: float = 149_597_870.7
DAY_S: float = 86_400.0

@dataclass
class OrbitData2D:
    # numpy arrays (scaled)
    t_days: np.ndarray     # shape (N,) (days since first sample)
    x_au: np.ndarray       # shape (N,) (X position in AU)
    y_au: np.ndarray       # shape (N,) (Y position in AU)

    # initial velocity (scaled)
    vx0_au_per_day: float
    vy0_au_per_day: float

    # tensors 
    t_tensor: torch.Tensor     # shape (N,1)
    xy_tensor: torch.Tensor    # shape (N,2)

    # raw 
    jd: np.ndarray         # shape (N,)
    x_km: np.ndarray       # shape (N,)
    y_km: np.ndarray       # shape (N,)
    vx_km_s: np.ndarray    # shape (N,)
    vy_km_s: np.ndarray    # shape (N,)


def _extract_horizons_data_lines(lines: list[str]) -> list[str]:
    """Return only the CSV data lines between $$SOE and $$EOE."""
    try:
        start = lines.index("$$SOE\n") + 1
        end = lines.index("$$EOE\n")
    except ValueError as e:
        raise ValueError("Could not find $$SOE/$$EOE markers in file.") from e

    data_lines = lines[start:end]
    if len(data_lines) == 0:
        raise ValueError("No data lines found between $$SOE and $$EOE.")
    return data_lines


def load_horizons_vectors_2d(
    filepath: str,
    *,
    au_km: float = AU_KM,
    day_s: float = DAY_S,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> OrbitData2D:
    """
    Parse a Horizons VECTORS export (.txt but CSV format) and return 2D orbit data:
    t_days, x_au, y_au (+ initial velocity in AU/day), and torch tensors.

    Expects columns like:
      JD, DATE, X, Y, Z, VX, VY, VZ, ...
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_lines = _extract_horizons_data_lines(lines)

    # Read CSV rows (Horizons sometimes ends each row with a trailing comma => extra empty col)
    df = pd.read_csv(io.StringIO("".join(data_lines)), header=None)
    df = df.dropna(axis=1, how="all")  # remove completely empty columns

    # Most common Horizons vector format here is 14 cols after dropping empty:
    # JD, DATE, X, Y, Z, VX, VY, VZ, X_s, Y_s, Z_s, VX_s, VY_s, VZ_s
    # We only need the first 8 to get X,Y,VX,VY.
    if df.shape[1] < 8:
        raise ValueError(f"Unexpected column count ({df.shape[1]}). Need at least 8 columns.")

    df = df.iloc[:, :8].copy()
    df.columns = ["JD", "DATE", "X", "Y", "Z", "VX", "VY", "VZ"]

    # Convert numeric columns
    for c in ["JD", "X", "Y", "VX", "VY"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["JD", "X", "Y", "VX", "VY"])
    if len(df) < 2:
        raise ValueError("Not enough valid numeric rows after parsing.")

    jd = df["JD"].to_numpy()
    x_km = df["X"].to_numpy()
    y_km = df["Y"].to_numpy()
    vx_km_s = df["VX"].to_numpy()
    vy_km_s = df["VY"].to_numpy()

    jd0 = jd[0]
    t_days = jd - jd0

    x_au = x_km / au_km
    y_au = y_km / au_km

    vx0_au_per_day = float(vx_km_s[0] * day_s / au_km)
    vy0_au_per_day = float(vy_km_s[0] * day_s / au_km)

    t_tensor = torch.tensor(t_days, dtype=dtype).view(-1, 1)
    xy_tensor = torch.tensor(np.stack([x_au, y_au], axis=1), dtype=dtype)

    if device is not None:
        t_tensor = t_tensor.to(device)
        xy_tensor = xy_tensor.to(device)

    return OrbitData2D(
        t_days=t_days,
        x_au=x_au,
        y_au=y_au,
        vx0_au_per_day=vx0_au_per_day,
        vy0_au_per_day=vy0_au_per_day,
        t_tensor=t_tensor,
        xy_tensor=xy_tensor,
        jd=jd,
        x_km=x_km,
        y_km=y_km,
        vx_km_s=vx_km_s,
        vy_km_s=vy_km_s,
    )
    
