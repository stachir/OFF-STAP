"""
src/clock_model.py

Evaluates constant-offset vs affine clock models.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy.optimize import least_squares
from offstap.core.config import Config


def _finite_time_signal(time: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    keep = np.isfinite(time) & np.isfinite(signal)
    time = time[keep]
    signal = signal[keep]
    if len(time) == 0:
        return time, signal
    order = np.argsort(time, kind="stable")
    time = time[order]
    signal = signal[order]
    _, unique = np.unique(time, return_index=True)
    return time[unique], signal[unique]

def fit_clock_model(ev3_calib: pd.DataFrame, vrmt_calib: pd.DataFrame, temporal_params: Dict[str, float], cfg: Config) -> Dict[str, Any]:
    """
    Fits and compares constant offset and affine clock models.
    """
    b_initial = temporal_params.get("b_final", temporal_params.get("b_refined", temporal_params.get("b_onset", 0.0)))

    ev3_proxy = cfg.temporal.get("ev3_onset_proxy", "speed_magnitude")
    vrmt_proxy = cfg.temporal.get("vrmt_onset_proxy", "speed_magnitude")

    if ev3_proxy not in ev3_calib.columns or vrmt_proxy not in vrmt_calib.columns:
        return {
            "selected_model": "constant_offset",
            "offset_b": float(b_initial),
            "scale_a": 1.0,
            "temporal_refinement_delta_s": 0.0,
            "affine_improvement_ratio": 0.0,
            "reason": "Missing signals for clock fitting."
        }

    t_e, s_e = _finite_time_signal(
        ev3_calib["local_time_s"].values, ev3_calib[ev3_proxy].values
    )
    t_v, s_v = _finite_time_signal(
        vrmt_calib["relative_time_s"].values, vrmt_calib[vrmt_proxy].values
    )
    if len(t_e) < 4 or len(t_v) < 4:
        return {
            "selected_model": "constant_offset", "offset_b": float(b_initial),
            "scale_a": 1.0, "candidate_scale_a": np.nan,
            "candidate_offset_b": np.nan, "temporal_refinement_delta_s": 0.0,
            "affine_improvement_ratio": 0.0, "constant_mse": np.nan,
            "affine_mse": np.nan, "candidate_hit_bound": False,
            "reason": "Insufficient finite signals for clock fitting.",
        }

    # Normalize for fitting
    s_e_norm = s_e - np.mean(s_e)
    if np.std(s_e_norm) > 0: s_e_norm /= np.std(s_e_norm)
    s_v_norm = s_v - np.mean(s_v)
    if np.std(s_v_norm) > 0: s_v_norm /= np.std(s_v_norm)

    # We map t_e -> t_v: tau = a * t_e + b
    # So we interpolate s_e at (t_v - b) / a
    def residual(params):
        a, b = params
        # To find what EV3 time corresponds to each VRMT time: t_e_query = (t_v - b) / a
        t_e_query = (t_v - b) / a
        inside = (t_e_query >= t_e[0]) & (t_e_query <= t_e[-1])
        s_e_interp = np.interp(t_e_query, t_e, s_e_norm)
        result = s_v_norm - s_e_interp
        result[~inside] = 2.0
        return result

    # Constant offset baseline
    res_const = residual([1.0, b_initial])
    mse_const = np.mean(res_const**2)

    scale_min = cfg.clock_model.get("scale_min", 0.95)
    scale_max = cfg.clock_model.get("scale_max", 1.05)

    # Affine fit
    try:
        opt = least_squares(residual, [1.0, b_initial], bounds=([scale_min, -np.inf], [scale_max, np.inf]))
        a_opt, b_opt = opt.x
        res_affine = opt.fun
        mse_affine = np.mean(res_affine**2)
    except Exception:
        a_opt, b_opt = 1.0, b_initial
        res_affine = res_const
        mse_affine = mse_const

    # We will use a heuristic: if MSE improves by more than threshold, accept affine
    # threshold name in config is affine_improvement_threshold_m (though it acts as a ratio)
    threshold = cfg.clock_model.get("affine_improvement_threshold_m", 0.05)

    candidate_hit_bound = bool(
        np.isclose(a_opt, scale_min, rtol=0.0, atol=1e-6)
        or np.isclose(a_opt, scale_max, rtol=0.0, atol=1e-6)
    )
    raw_improvement_ratio = (mse_const - mse_affine) / mse_const if mse_const > 0 else 0.0

    if mse_const > 0 and raw_improvement_ratio > threshold and not candidate_hit_bound:
        selected_model = "affine"
        b_final = b_opt
        a_final = a_opt
        improvement_ratio = (mse_const - mse_affine) / mse_const
        reason = f"Affine improved fit by {improvement_ratio:.2%} (rel MSE)"
    else:
        selected_model = "constant_offset"
        b_final = b_initial
        a_final = 1.0
        improvement_ratio = 0.0
        reason = "Affine improvement below threshold or hit bounds."

    return {
        "selected_model": selected_model,
        "offset_b": float(b_final),
        "scale_a": float(a_final),
        "candidate_scale_a": float(a_opt),
        "candidate_offset_b": float(b_opt),
        "temporal_refinement_delta_s": float(b_final - b_initial),
        "affine_improvement_ratio": float(improvement_ratio),
        "constant_mse": float(mse_const),
        "affine_mse": float(mse_affine),
        "candidate_hit_bound": candidate_hit_bound,
        "reason": reason,
        "plot_data": {
            "t_v": t_v.tolist() if 't_v' in locals() else [],
            "res_const": res_const.tolist() if 'res_const' in locals() else [],
            "res_affine": res_affine.tolist() if 'res_affine' in locals() else []
        }
    }
