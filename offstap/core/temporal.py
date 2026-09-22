"""Temporal onset detection and bounded, finite-overlap correlation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from offstap.core.config import Config


@dataclass
class CorrelationResult:
    """Diagnostics and acceptance state for a lag-wise correlation search."""

    lags_s: np.ndarray
    scores: np.ndarray
    overlap_s: np.ndarray
    best_lag_s: float
    best_peak: float
    second_peak: float
    peak_ratio: float
    status: str
    rejection_reason: Optional[str] = None

    # Names used by earlier diagnostics and by downstream readers.
    @property
    def lag_scores(self) -> np.ndarray:
        return self.scores

    @property
    def valid_lags(self) -> np.ndarray:
        return self.lags_s[np.isfinite(self.scores)]

    @property
    def valid_corr(self) -> np.ndarray:
        return self.scores[np.isfinite(self.scores)]

    @property
    def best_peak_score(self) -> float:
        return self.best_peak

    @property
    def second_peak_score(self) -> float:
        return self.second_peak

    @property
    def best_corr(self) -> float:
        return self.best_peak

    @property
    def second_best_corr(self) -> float:
        return self.second_peak

    @property
    def peak_confidence_ratio(self) -> float:
        return self.peak_ratio

    @property
    def overlap_duration_s(self) -> float:
        return float(np.nanmax(self.overlap_s)) if np.any(np.isfinite(self.overlap_s)) else 0.0

    @property
    def overlap_by_lag_s(self) -> np.ndarray:
        return self.overlap_s

    @property
    def overlap(self) -> float:
        """Maximum finite overlap, retained as a scalar compatibility alias."""
        return self.overlap_duration_s

    @property
    def reason(self) -> Optional[str]:
        return self.rejection_reason


def _finite_signal(t: np.ndarray, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return a sorted, unique, finite time/signal pair."""
    t = np.asarray(t, dtype=float)
    signal = np.asarray(signal, dtype=float)
    keep = np.isfinite(t) & np.isfinite(signal)
    t, signal = t[keep], signal[keep]
    if len(t) == 0:
        return t, signal
    order = np.argsort(t, kind="stable")
    t, signal = t[order], signal[order]
    _, unique = np.unique(t, return_index=True)
    return t[unique], signal[unique]


def detect_onset(t: np.ndarray, speed: np.ndarray, threshold: float = 0.5) -> float:
    """Find the first finite sample whose speed exceeds ``threshold``."""
    t, speed = _finite_signal(t, speed)
    if len(t) == 0:
        return np.nan
    idx = np.where(speed > threshold)[0]
    return float(t[idx[0]]) if len(idx) else float(t[0])


def _temporal_value(cfg: Any, key: str, default: Any) -> Any:
    if hasattr(cfg, "temporal"):
        temporal = cfg.temporal
    elif isinstance(cfg, dict) and "temporal" in cfg:
        temporal = cfg["temporal"]
    elif isinstance(cfg, dict):
        temporal = cfg
    else:
        temporal = {}
    return temporal.get(key, default)


def _lag_grid(window: float, step: float) -> np.ndarray:
    if not np.isfinite(window) or window < 0 or not np.isfinite(step) or step <= 0:
        return np.array([], dtype=float)
    # Include both configured search-window endpoints when they lie on the grid.
    n = int(np.floor(window / step + 1e-10))
    grid = np.arange(-n, n + 1, dtype=float) * step
    if grid.size == 0:
        return np.array([-window, window], dtype=float)
    if not np.isclose(grid[-1], window, rtol=0.0, atol=max(step * 1e-9, 1e-12)):
        grid = np.concatenate(([-window], grid, [window]))
    return np.unique(np.sort(grid))


def _local_peaks(lags: np.ndarray, scores: np.ndarray) -> list[tuple[int, float]]:
    """Find strict local maxima, collapsing equal plateaus to their low index."""
    peaks: list[tuple[int, float]] = []
    n = len(scores)
    i = 1
    while i < n - 1:
        if not np.isfinite(scores[i]):
            i += 1
            continue
        start = i
        while i + 1 < n and np.isfinite(scores[i + 1]) and np.isclose(
            scores[i + 1], scores[start], rtol=0.0, atol=1e-12
        ):
            i += 1
        end = i
        if start > 0 and end < n - 1 and np.isfinite(scores[start - 1]) and np.isfinite(scores[end + 1]):
            if scores[start] > scores[start - 1] and scores[start] > scores[end + 1]:
                peaks.append((start, float(scores[start])))
        i += 1
    return peaks


def normalized_lag_correlation(
    t_x: np.ndarray,
    x: np.ndarray,
    t_y: np.ndarray,
    y: np.ndarray,
    cfg: Config,
) -> CorrelationResult:
    """Compute bounded Pearson correlation for each configured lag.

    At lag ``ell`` the score compares ``x(t)`` with ``y(t + ell)`` over the
    finite intersection of their domains.  This convention recovers a
    positive lag when ``y`` is a delayed copy of ``x``.
    """
    grid_step = float(_temporal_value(cfg, "grid_step_s", 0.05))
    window = float(_temporal_value(cfg, "cross_corr_window_s", 5.0))
    min_overlap = float(_temporal_value(cfg, "min_overlap_s", 0.0))
    min_correlation = float(_temporal_value(cfg, "min_correlation", 0.0))
    min_separation = float(_temporal_value(cfg, "min_peak_separation_s", grid_step))
    min_ratio = float(_temporal_value(cfg, "min_peak_confidence_ratio", 1.0))

    tx, sx = _finite_signal(t_x, x)
    ty, sy = _finite_signal(t_y, y)
    lags = _lag_grid(window, grid_step)
    scores = np.full(lags.shape, np.nan, dtype=float)
    overlap = np.zeros(lags.shape, dtype=float)

    if len(tx) < 3 or len(ty) < 3 or lags.size == 0:
        return CorrelationResult(lags, scores, overlap, np.nan, np.nan, np.nan, np.inf,
                                 "insufficient_overlap", "insufficient_overlap")

    for idx, lag in enumerate(lags):
        # y(t + lag) is defined only where t + lag lies within y's domain.
        start = max(float(tx[0]), float(ty[0] - lag))
        end = min(float(tx[-1]), float(ty[-1] - lag))
        duration = max(0.0, end - start)
        overlap[idx] = duration
        if duration < min_overlap:
            continue
        query = np.arange(start, end + grid_step * 0.5, grid_step, dtype=float)
        query = query[(query >= tx[0]) & (query <= tx[-1]) & (query + lag >= ty[0]) & (query + lag <= ty[-1])]
        if len(query) < 3:
            continue
        xv = np.interp(query, tx, sx)
        yv = np.interp(query + lag, ty, sy)
        finite = np.isfinite(xv) & np.isfinite(yv)
        if np.count_nonzero(finite) < 3:
            continue
        xv, yv = xv[finite], yv[finite]
        xc, yc = xv - np.mean(xv), yv - np.mean(yv)
        denominator = float(np.sqrt(np.dot(xc, xc) * np.dot(yc, yc)))
        scores[idx] = float(np.dot(xc, yc) / denominator) if denominator > 0.0 else 0.0
        # Guard against round-off violating the coefficient contract.
        scores[idx] = float(np.clip(scores[idx], -1.0, 1.0))

    valid = np.isfinite(scores)
    if not np.any(valid) or not np.any(overlap[valid] >= min_overlap):
        return CorrelationResult(lags, scores, overlap, np.nan, np.nan, np.nan, np.inf,
                                 "insufficient_overlap", "insufficient_overlap")

    peaks = _local_peaks(lags, scores)
    max_score = float(np.nanmax(scores))
    max_candidates = [(i, v) for i, v in peaks if np.isclose(v, max_score, rtol=0.0, atol=1e-12)]
    if not max_candidates:
        max_candidates = [(int(i), float(scores[i])) for i in np.flatnonzero(valid)
                          if np.isclose(scores[i], max_score, rtol=0.0, atol=1e-12)]
    best_idx, best_peak = min(max_candidates, key=lambda p: (abs(float(lags[p[0]])), float(lags[p[0]]), p[0]))
    best_lag = float(lags[best_idx])

    eligible_second = [
        (i, value) for i, value in peaks
        if i != best_idx and abs(float(lags[i]) - best_lag) >= min_separation - 1e-12
    ]
    if eligible_second:
        second_idx, second_peak = min(
            eligible_second,
            key=lambda p: (-p[1], abs(float(lags[p[0]]) - best_lag), float(lags[p[0]]), p[0]),
        )
        second_peak = float(second_peak)
    else:
        second_peak = np.nan
    peak_ratio = np.inf if not np.isfinite(second_peak) or second_peak <= 0 else float(best_peak / second_peak)

    at_boundary = np.isclose(abs(best_lag), window, rtol=0.0, atol=max(grid_step * 1e-9, 1e-12))
    if at_boundary:
        status, reason = "boundary_limit", "boundary_limit"
    elif best_peak < min_correlation:
        status, reason = "low_correlation", "low_correlation"
    elif np.isfinite(second_peak) and second_peak > 0 and peak_ratio < min_ratio:
        status, reason = "ambiguous", "ambiguous"
    else:
        status, reason = "ok", None

    return CorrelationResult(lags, scores, overlap, best_lag, float(best_peak), second_peak,
                             float(peak_ratio), status, reason)


def estimate_temporal_alignment(ev3_calib: pd.DataFrame, vrmt_calib: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    """Estimate onset offset and refine it with normalized lag correlation."""
    ev3_proxy = _temporal_value(cfg, "ev3_onset_proxy", "speed_magnitude")
    vrmt_proxy = _temporal_value(cfg, "vrmt_onset_proxy", "speed_magnitude")
    if ev3_proxy not in ev3_calib.columns or vrmt_proxy not in vrmt_calib.columns:
        return {"b_onset": 0.0, "b_refined": 0.0, "b_final": 0.0, "method_used": "none",
                "peak_confidence_ratio": np.nan, "status": "missing_input"}

    t_e, s_e = _finite_signal(ev3_calib["local_time_s"].values, ev3_calib[ev3_proxy].values)
    t_v, s_v = _finite_signal(vrmt_calib["relative_time_s"].values, vrmt_calib[vrmt_proxy].values)
    if len(t_e) < 4 or len(t_v) < 4:
        return {"b_onset": np.nan, "b_refined": np.nan, "b_final": np.nan,
                "method_used": "none", "lag": np.nan, "best_corr": np.nan,
                "second_best_corr": np.nan, "peak_confidence_ratio": np.nan,
                "overlap_duration_s": 0.0, "status": "invalid_data",
                "valid_lags": np.array([]), "valid_corr": np.array([])}

    t_ev3_onset = detect_onset(t_e, s_e)
    t_vive_onset = detect_onset(t_v, s_v)
    b_onset = float(t_vive_onset - t_ev3_onset)
    if _temporal_value(cfg, "alignment_method", "crosscorr") == "onset_only":
        return {"b_onset": b_onset, "b_refined": b_onset, "b_final": b_onset, "method_used": "onset_only",
                "lag": 0.0, "best_corr": 0.0, "second_best_corr": np.nan,
                "peak_confidence_ratio": np.inf, "status": "ok", "valid_lags": np.array([]),
                "valid_corr": np.array([])}

    correlation = normalized_lag_correlation(
        t_e - t_ev3_onset, s_e, t_v - t_vive_onset, s_v, cfg
    )
    b_refined = b_onset + correlation.best_lag_s if np.isfinite(correlation.best_lag_s) else b_onset
    b_final = b_refined if correlation.status == "ok" else b_onset
    return {
        "b_onset": b_onset,
        "b_refined": float(b_refined),
        "b_final": float(b_final),
        "method_used": "crosscorr",
        "lag": correlation.best_lag_s,
        "best_corr": correlation.best_peak,
        "second_best_corr": correlation.second_peak,
        "peak_confidence_ratio": correlation.peak_ratio,
        "overlap_duration_s": correlation.overlap_duration_s,
        "status": correlation.status,
        "rejection_reason": correlation.rejection_reason,
        "valid_lags": correlation.lags_s[np.isfinite(correlation.scores)],
        "valid_corr": correlation.scores[np.isfinite(correlation.scores)],
        "correlation_result": correlation,
    }
