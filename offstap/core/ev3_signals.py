"""
src/ev3_signals.py

Derives EV3 motion signals (speed proxies, turn rates, unwrapped heading)
without overwriting raw telemetry.
"""
import numpy as np
import pandas as pd
from typing import Optional

def unwrap_heading(angles_deg: np.ndarray) -> np.ndarray:
    """Unwraps angular data in degrees to avoid discontinuities at 360."""
    angles_rad = np.deg2rad(angles_deg)
    unwrapped_rad = np.unwrap(angles_rad)
    return np.rad2deg(unwrapped_rad)

def derive_ev3_signals(df: pd.DataFrame, speed_col: str = "motor_speed", heading_col: Optional[str] = None) -> pd.DataFrame:
    df_out = df.copy()

    # Speed magnitude
    if speed_col in df_out.columns:
        df_out["speed_magnitude"] = df_out[speed_col].abs()

    # Unwrapped heading & turn rate proxy
    if heading_col and heading_col in df_out.columns:
        df_out["unwrapped_heading"] = unwrap_heading(df_out[heading_col].values)

        # Turn rate proxy (derivative of heading)
        dt = np.diff(df_out["local_time_s"].values)
        dheading = np.diff(df_out["unwrapped_heading"].values)

        turn_rate = np.zeros(len(df_out))
        valid_dt = dt > 0
        if np.any(valid_dt):
            turn_rate[1:][valid_dt] = dheading[valid_dt] / dt[valid_dt]
        df_out["turn_rate_proxy"] = turn_rate

        # Phase labels
        df_out["movement_phase"] = "straight"
        df_out.loc[np.abs(df_out["turn_rate_proxy"]) > 5.0, "movement_phase"] = "turning"

    if "speed_magnitude" in df_out.columns:
        df_out.loc[df_out["speed_magnitude"] < 1.0, "movement_phase"] = "stopped"

    return df_out
