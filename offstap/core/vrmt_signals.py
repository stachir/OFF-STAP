"""
src/vrmt_signals.py

Derives Vive motion signals (finite diff velocities, tracking loss intervals).
"""
import numpy as np
import pandas as pd
from typing import Tuple

def derive_vrmt_signals(df: pd.DataFrame, gap_threshold_s: float = 0.5) -> pd.DataFrame:
    df_out = df.copy()

    t = df_out["relative_time_s"].values
    dt = np.diff(t)

    # 3D Speed (Finite differences)
    if all(c in df_out.columns for c in ["x", "y", "z"]):
        dx = np.diff(df_out["x"].values)
        dy = np.diff(df_out["y"].values)
        dz = np.diff(df_out["z"].values)

        speed = np.zeros(len(df_out))
        valid_dt = dt > 0
        if np.any(valid_dt):
            dist = np.sqrt(dx[valid_dt]**2 + dy[valid_dt]**2 + dz[valid_dt]**2)
            speed[1:][valid_dt] = dist / dt[valid_dt]

        df_out["speed_magnitude"] = speed

    # Tracking loss candidates
    tracking_loss = np.zeros(len(df_out), dtype=bool)
    if np.any(dt > gap_threshold_s):
        # Mark the point after the gap as a potential relocalization/loss point
        tracking_loss[1:][dt > gap_threshold_s] = True
    df_out["tracking_loss_candidate"] = tracking_loss

    return df_out
