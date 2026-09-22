"""
src/degeneracy.py

Checks if the calibration interval contains sufficient excitation
(linear distance and angular change).
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from offstap.core.config import Config

def check_degeneracy(df: pd.DataFrame, cfg: Config, is_ev3: bool = True) -> Tuple[bool, Dict[str, float]]:
    """
    Returns (is_degenerate, metrics_dict)
    """
    if df.empty:
        return True, {"distance": 0.0, "angular_change": 0.0}

    metrics = {}

    if is_ev3:
        # Distance (integral of speed)
        if "speed_magnitude" in df.columns:
            time = pd.to_numeric(df["local_time_s"], errors="coerce").to_numpy(dtype=float)
            speed_all = pd.to_numeric(df["speed_magnitude"], errors="coerce").to_numpy(dtype=float)
            dt = np.diff(time)
            speed = speed_all[:-1]
            valid = np.isfinite(dt) & (dt > 0) & np.isfinite(speed)
            dist = np.sum(speed[valid] * dt[valid])
            metrics["distance"] = float(dist)
        else:
            metrics["distance"] = 0.0

        # Angular change
        if "unwrapped_heading" in df.columns:
            h = pd.to_numeric(df["unwrapped_heading"], errors="coerce").to_numpy(dtype=float)
            dh = np.diff(h)
            ang_change = np.sum(np.abs(dh[np.isfinite(dh)]))
            metrics["angular_change"] = float(ang_change)
        else:
            metrics["angular_change"] = 0.0

    else:
        # VRMT Distance
        if all(c in df.columns for c in ["x", "y"]):
            x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
            dx = np.diff(x)
            dy = np.diff(y)
            valid = np.isfinite(dx) & np.isfinite(dy)
            dist = np.sum(np.sqrt(dx[valid]**2 + dy[valid]**2))
            metrics["distance"] = float(dist)
        else:
            metrics["distance"] = 0.0

        # Angular change for VRMT (if yaw is available, else 0)
        metrics["angular_change"] = 0.0

    min_dist = cfg.calibration.get("min_linear_distance_m", 0.1)
    min_ang = cfg.calibration.get("min_angular_change_deg", 10.0)

    if not np.isfinite(metrics["distance"]):
        metrics["distance"] = 0.0
    if not np.isfinite(metrics["angular_change"]):
        metrics["angular_change"] = 0.0

    if is_ev3:
        # Translation and rotation are alternative sources of excitation. A
        # straight calibration trajectory is not degenerate merely because it
        # has little angular change.
        is_degenerate = (
            metrics["distance"] < min_dist
            and metrics["angular_change"] < min_ang
        )
    else:
        is_degenerate = metrics["distance"] < min_dist

    return is_degenerate, metrics
