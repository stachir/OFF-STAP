"""
src/metrics.py

Manually implements Absolute Trajectory Error (ATE) and Relative Pose Error (RPE)
for 2D planar trajectories.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Mapping, Optional

def compute_ate(p_est: np.ndarray, p_gt: np.ndarray) -> np.ndarray:
    """
    Computes Absolute Trajectory Error.
    p_est: (N, 2) array of estimated positions (e.g. EV3 odometry)
    p_gt: (N, 2) array of ground truth positions (e.g. Vive reference)
    Returns array of errors per point.
    """
    diff = p_est - p_gt
    return np.linalg.norm(diff, axis=1)

def compute_robust_metrics(err: np.ndarray, huber_delta: float = 1.0, trim_ratio: float = 0.05, outlier_threshold: float = 1.0) -> Dict[str, float]:
    """Computes robust statistics on an error array."""
    err = np.asarray(err, dtype=float)
    err = err[np.isfinite(err)]
    if len(err) == 0:
        return {
            "mae": np.nan, "trimmed_rmse": np.nan, "huber_loss": np.nan, "outlier_count": 0,
            "mean": np.nan, "p75": np.nan, "p95": np.nan, "max": np.nan, "min": np.nan
        }

    abs_err = np.abs(err)
    mae = float(np.median(abs_err))

    # Huber loss
    quadratic = np.minimum(abs_err, huber_delta)
    linear = abs_err - quadratic
    huber = np.mean(0.5 * quadratic**2 + huber_delta * linear)

    # Trimmed RMSE
    n_trim = int(len(err) * trim_ratio)
    if n_trim > 0 and len(err) > 2 * n_trim:
        sorted_err = np.sort(err)
        trimmed_err = sorted_err[n_trim:-n_trim]
        t_rmse = np.sqrt(np.mean(trimmed_err**2))
    else:
        t_rmse = np.sqrt(np.mean(err**2))

    outliers = int(np.sum(abs_err > outlier_threshold))

    return {
        "mae": float(mae),
        "trimmed_rmse": float(t_rmse),
        "huber_loss": float(huber),
        "outlier_count": outliers,
        "mean": float(np.mean(abs_err)),
        "p75": float(np.percentile(abs_err, 75)),
        "p95": float(np.percentile(abs_err, 95)),
        "max": float(np.max(abs_err)),
        "min": float(np.min(abs_err))
    }


def _metric_value(cfg: Any, *keys: str, default: float) -> float:
    """Read a metric setting from either ``Config`` or a mapping.

    ``Config`` is intentionally a very small wrapper around a dictionary, while
    tests and callers often provide plain dictionaries.  Supporting both here
    keeps the metric implementation independent of the configuration loader.
    """
    section: Any = None
    if cfg is not None:
        if isinstance(cfg, Mapping):
            section = cfg.get("metrics", cfg)
        else:
            section = getattr(cfg, "metrics", None)
            if section is None and hasattr(cfg, "_config"):
                section = cfg._config.get("metrics", {})
    if not isinstance(section, Mapping):
        section = {}
    for key in keys:
        if key in section and section[key] is not None:
            try:
                return float(section[key])
            except (TypeError, ValueError):
                break
    return float(default)


def _time_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first supported elapsed-time column present in ``df``."""
    for name in (
        "_metric_time_s",
        "t_common_s",
        "common_time_s",
        "timestamp_s",
        "elapsed_time_s",
        "elapsed_s",
        "time_s",
        "relative_time_s",
        "local_time_s",
        "timestamp",
        "time",
        "source_timestamp_ns",
    ):
        if name in df.columns:
            return name
    return None


def compute_time_based_rpe(
    df: pd.DataFrame,
    target_interval_s: float,
    tolerance_s: float,
) -> np.ndarray:
    """Compute translational RPE using elapsed-time, nearest-pair matching.

    For every finite sample ``i``, a later sample ``j`` is selected when
    ``t[j] - t[i]`` is within ``tolerance_s`` of ``target_interval_s``.  The
    nearest candidate is selected; equal-distance candidates deterministically
    prefer the earlier (lower) source index.  This avoids assuming a fixed
    sample rate or row spacing.
    """
    if target_interval_s <= 0 or tolerance_s < 0:
        raise ValueError("target_interval_s must be positive and tolerance_s non-negative")
    required = {"odometry_x", "odometry_y", "vrmt_aligned_x", "vrmt_aligned_y"}
    if not required.issubset(df.columns):
        return np.array([], dtype=float)
    time_col = _time_column(df)
    if time_col is None:
        return np.array([], dtype=float)

    raw_times = df[time_col]
    if pd.api.types.is_datetime64_any_dtype(raw_times):
        times = (raw_times - raw_times.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    else:
        times = pd.to_numeric(raw_times, errors="coerce").to_numpy(dtype=float)
        if time_col.endswith("_ns"):
            times = times * 1e-9
    est = df[["odometry_x", "odometry_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    ref = df[["vrmt_aligned_x", "vrmt_aligned_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(times) & np.isfinite(est).all(axis=1) & np.isfinite(ref).all(axis=1)
    if finite.sum() < 2:
        return np.array([], dtype=float)

    # Stable sorting makes tie handling reproducible even when timestamps are
    # duplicated.  Non-finite rows are excluded before matching.
    source = np.flatnonzero(finite)
    order = np.argsort(times[finite], kind="mergesort")
    source = source[order]
    t = times[source]
    p_est = est[source]
    p_ref = ref[source]
    if "source_row_index" in df.columns:
        raw_identity = pd.to_numeric(df["source_row_index"], errors="coerce").to_numpy(dtype=float)
        identity = raw_identity[source]
        identity[~np.isfinite(identity)] = source[~np.isfinite(identity)]
    else:
        identity = source.astype(float)

    errors = []
    right = 1
    for i, start in enumerate(t):
        target = start + target_interval_s
        if right <= i:
            right = i + 1
        # ``right`` is monotonic because both ``i`` and the target elapsed
        # time are monotonic.  This is the linear-time two-pointer matcher.
        while right < len(t) and t[right] < target:
            right += 1
        candidates = []
        for candidate in (right - 1, right):
            if candidate <= i or candidate < 0 or candidate >= len(t):
                continue
            delta = float(t[candidate] - start)
            if abs(delta - target_interval_s) <= tolerance_s:
                candidates.append((abs(delta - target_interval_s), identity[candidate], candidate))
        if not candidates:
            continue
        # Tuple ordering gives nearest elapsed time first and lower stable
        # source identity on an exact tie (independent of timestamp sorting).
        _, _, j = min(candidates, key=lambda item: (item[0], item[1]))
        est_delta = p_est[j] - p_est[i]
        ref_delta = p_ref[j] - p_ref[i]
        errors.append(float(np.linalg.norm(est_delta - ref_delta)))
    return np.asarray(errors, dtype=float)

def compute_rpe(p_est: np.ndarray, p_gt: np.ndarray, delta: int = 1) -> np.ndarray:
    """
    Computes Relative Pose Error (translational) over a fixed index delta.
    """
    if len(p_est) <= delta:
        return np.array([])

    est_diff = p_est[delta:] - p_est[:-delta]
    gt_diff = p_gt[delta:] - p_gt[:-delta]

    # Distance between the relative translations
    err = np.linalg.norm(est_diff - gt_diff, axis=1)
    return err


def primary_metric_support(df_aligned: pd.DataFrame) -> list[Any]:
    """Return stable source keys contributing to primary held-out metrics.

    Membership is selected before finite residual filtering so the returned
    support is an exact audit of the rows actually scored.  A missing
    persisted partition yields an empty support (the primary metric path then
    fails closed); calibration rows can therefore never appear here.
    """
    if "partition" not in df_aligned.columns:
        return []
    partition = df_aligned["partition"].astype(str).str.strip().str.lower()
    eval_mask = partition.isin({"evaluation", "eval", "heldout", "held_out", "test"})
    required = {"odometry_x", "odometry_y", "vrmt_aligned_x", "vrmt_aligned_y"}
    if not required.issubset(df_aligned.columns):
        finite_mask = np.ones(len(df_aligned), dtype=bool)
    else:
        values = df_aligned[list(required)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(values).all(axis=1)
    # ``support_key`` is the canonical key exposed by alignment (and may be a
    # textual identity in synthetic/legacy callers); fall back to the
    # persisted source row index when that derived column is unavailable.
    key_col = "support_key" if "support_key" in df_aligned.columns else (
        "source_row_index" if "source_row_index" in df_aligned.columns else None
    )
    if key_col is None:
        return []
    keys = df_aligned[key_col]
    valid = eval_mask.to_numpy() & finite_mask & keys.notna().to_numpy()
    return keys.loc[valid].tolist()

def evaluate_trajectory_metrics(
    df_aligned: pd.DataFrame,
    cfg: Any = None,
    evaluation_only: bool = True,
) -> Dict[str, Any]:
    """
    Computes ATE and RPE on an aligned DataFrame.
    Assumes columns: vrmt_aligned_x, vrmt_aligned_y, odometry_x, odometry_y.
    """
    # Primary metrics are evaluated on the persisted evaluation partition.  A
    # missing partition is an explicit invalid result rather than permission
    # to score an unknown mixture of calibration and evaluation rows.
    if evaluation_only and "partition" not in df_aligned.columns:
        return {
            "ate_rmse": np.nan,
            "ate_mean": np.nan,
            "ate_std": np.nan,
            "rpe_rmse": np.nan,
            "rpe_mean": np.nan,
            "rpe_std": np.nan,
            "rpe_median": np.nan,
            "num_samples": 0,
            "status": "failed_metric_computation",
            "failure_reason": "missing_evaluation_partition",
            "support_keys": [],
        }
    if evaluation_only:
        partition = df_aligned["partition"].astype(str).str.strip().str.lower()
        eval_mask = partition.isin({"evaluation", "eval", "heldout", "held_out", "test"})
        if not bool(eval_mask.any()):
            return {
                "ate_rmse": np.nan,
                "ate_mean": np.nan,
                "ate_std": np.nan,
                "rpe_rmse": np.nan,
                "rpe_mean": np.nan,
                "rpe_std": np.nan,
                "rpe_median": np.nan,
                "num_samples": 0,
                "status": "failed_metric_computation",
                "failure_reason": "missing_evaluation_partition",
                "support_keys": [],
            }
        df_aligned = df_aligned.loc[eval_mask].copy()
    elif not evaluation_only:
        df_aligned = df_aligned.copy()

    # If EV3 doesn't have odometry_x and odometry_y, we can't compute ATE.
    # In some datasets, motor_speed is integrated to get 1D distance, but ATE needs 2D.
    # Let's check if odometry exists.
    required_cols = ["odometry_x", "odometry_y", "vrmt_aligned_x", "vrmt_aligned_y"]
    has_odom = all(c in df_aligned.columns for c in required_cols)

    if df_aligned.empty:
        return {
            "ate_rmse": np.nan, "ate_mean": np.nan, "ate_std": np.nan,
            "rpe_rmse": np.nan, "rpe_mean": np.nan, "rpe_std": np.nan,
            "rpe_median": np.nan, "num_samples": 0,
            "status": "no_common_samples",
            "support_keys": [],
        }

    if not has_odom:
        return {
            "ate_rmse": np.nan, "ate_mean": np.nan, "ate_std": np.nan,
            "rpe_rmse": np.nan, "rpe_mean": np.nan, "rpe_std": np.nan,
            "rpe_median": np.nan, "num_samples": 0,
            "status": "failed_metric_computation",
            "failure_reason": "missing_required_aligned_columns",
            "support_keys": [],
        }

    p_est = df_aligned[["odometry_x", "odometry_y"]].values
    p_gt = df_aligned[["vrmt_aligned_x", "vrmt_aligned_y"]].values

    finite_mask = np.isfinite(p_est).all(axis=1) & np.isfinite(p_gt).all(axis=1)
    # Keep support evidence synchronized with the rows used for residuals.
    # Missing source identities are excluded from the evidence rather than
    # replaced by post-trim positional indexes.
    support_col = "support_key" if "support_key" in df_aligned.columns else (
        "source_row_index" if "source_row_index" in df_aligned.columns else None
    )
    if support_col is not None:
        finite_mask &= df_aligned[support_col].notna().to_numpy()
    p_est = p_est[finite_mask]
    p_gt = p_gt[finite_mask]

    if len(p_est) == 0:
        return {
            "ate_rmse": np.nan, "ate_mean": np.nan, "ate_std": np.nan,
            "rpe_rmse": np.nan, "rpe_mean": np.nan, "rpe_std": np.nan,
            "rpe_median": np.nan, "num_samples": 0,
            "status": "failed_metric_computation",
            "failure_reason": "no_finite_aligned_samples",
            "support_keys": [],
        }

    ate = compute_ate(p_est, p_gt)

    target_interval_s = _metric_value(
        cfg, "rpe_target_interval_s", "target_interval_s", default=1.0
    )
    tolerance_s = _metric_value(
        cfg, "rpe_tolerance_s", "tolerance_s", default=0.02
    )
    rpe_frame = df_aligned.loc[finite_mask].copy()
    # Legacy aligned frames may not retain timestamps.  Their source sampling
    # interval is configured rather than encoded as a row-count delta.
    if _time_column(rpe_frame) is None:
        sample_interval_s = _metric_value(
            cfg, "rpe_default_sample_interval_s", default=0.1
        )
        rpe_frame.insert(0, "_metric_time_s", np.arange(len(rpe_frame)) * sample_interval_s)
    # The helper discovers any existing elapsed-time column directly; no
    # interpolation or endpoint extrapolation is performed.
    rpe = compute_time_based_rpe(rpe_frame, target_interval_s, tolerance_s)

    metrics = {
        "ate_rmse": float(np.sqrt(np.mean(ate**2))),
        "ate_mean": float(np.mean(ate)),
        "ate_std": float(np.std(ate)),
        "num_samples": int(len(ate)),
        "status": "success",
        "support_keys": primary_metric_support(df_aligned.loc[finite_mask].copy()),
    }

    # Robust metrics
    percentile = _metric_value(
        cfg, "trimmed_percentile", "trim_percentile", default=90.0
    )
    if not 0.0 < percentile <= 100.0:
        raise ValueError("metrics.trimmed_percentile must be in (0, 100]")
    # ``trimmed_percentile`` denotes the central percentage retained.  The
    # robust helper accepts a per-tail trim ratio.
    trim_ratio = (100.0 - percentile) / 200.0
    huber_delta = _metric_value(cfg, "huber_delta", default=1.0)
    outlier_threshold = _metric_value(
        cfg, "outlier_threshold_m", "outlier_threshold", default=1.0
    )
    robust_ate = compute_robust_metrics(
        ate,
        huber_delta=huber_delta,
        trim_ratio=trim_ratio,
        outlier_threshold=outlier_threshold,
    )
    metrics["ate_mae"] = robust_ate["mae"]
    metrics["ate_trimmed_rmse"] = robust_ate["trimmed_rmse"]
    metrics["ate_huber_loss"] = robust_ate["huber_loss"]
    metrics["ate_outlier_count"] = robust_ate["outlier_count"]
    metrics["ate_p75"] = robust_ate["p75"]
    metrics["ate_p95"] = robust_ate["p95"]
    metrics["ate_max"] = robust_ate["max"]

    # Phase specific
    # We evaluate phases only if they are annotated. No synthetic fallback.
    phases = []
    if "movement_phase" in df_aligned.columns:
        phases = df_aligned.loc[finite_mask, "movement_phase"].unique()

    for phase in phases:
        mask = (df_aligned.loc[finite_mask, "movement_phase"] == phase).values
        if np.sum(mask) > 0:
            phase_ate = ate[mask]
            metrics[f"ate_rmse_{phase}"] = float(np.sqrt(np.mean(phase_ate**2)))
            metrics[f"num_samples_{phase}"] = int(np.sum(mask))

    if len(rpe) > 0:
        metrics.update({
            "rpe_rmse": float(np.sqrt(np.mean(rpe**2))),
            "rpe_mean": float(np.mean(rpe)),
            "rpe_std": float(np.std(rpe)),
            "rpe_median": float(np.median(rpe))
        })
    else:
        metrics.update({
            "rpe_rmse": np.nan, "rpe_mean": np.nan, "rpe_std": np.nan, "rpe_median": np.nan
        })

    return metrics
