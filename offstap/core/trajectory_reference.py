"""
src/trajectory_reference.py

Builds a continuous-time representation of the Vive trajectory (Level B: Spline).
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Any, Optional
from scipy.interpolate import CubicSpline, PchipInterpolator
from offstap.core.config import Config
from offstap.core.spatial import get_planar_trajectory, apply_spatial_transform


@dataclass
class ReferenceQuery:
    """Finite-support result of querying a Vive reference representation.

    ``valid_mask`` identifies query points inside the fitted Vive support and
    with finite interpolated coordinates.  Values outside that support are
    returned as NaN; they are never extrapolated.  The object deliberately
    keeps the original query times so every H2 arm can be audited against the
    same EV3 evaluation observations.
    """

    method: str
    query_times: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    extrapolated_count: int = 0
    source_row_index: Optional[np.ndarray] = None

    @property
    def vive_operator(self) -> str:
        return self.method

    @property
    def values(self) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        return self.x, self.y, self.heading


_REFERENCE_METHODS = ("zoh", "linear", "pchip", "spline")


def _coerce_vive_samples(vive_samples: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract time, planar coordinates, heading and source identities.

    Lightweight contract tests use either a DataFrame or an ``(t, x, y)``
    tuple; production callers may pass the cleaned Vive frame directly.
    """
    source = None
    heading = None
    if isinstance(vive_samples, pd.DataFrame):
        frame = vive_samples
        t_col = next((c for c in ("relative_time_s", "elapsed_time", "time_s", "timestamp_s") if c in frame), None)
        x_col = next((c for c in ("planar_x", "x", "v1_planar_x", "vrmt_aligned_x") if c in frame), None)
        y_col = next((c for c in ("planar_y", "y", "v1_planar_y", "vrmt_aligned_y") if c in frame), None)
        if t_col is None or x_col is None or y_col is None:
            raise ValueError("Vive samples require time, planar x and planar y columns")
        t = pd.to_numeric(frame[t_col], errors="coerce").to_numpy(float)
        x = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(float)
        h_col = next((c for c in ("heading", "planar_heading", "theta", "rtheta") if c in frame), None)
        if h_col is not None:
            heading = pd.to_numeric(frame[h_col], errors="coerce").to_numpy(float)
        if "source_row_index" in frame.columns:
            source = frame["source_row_index"].to_numpy(copy=True)
    elif isinstance(vive_samples, dict):
        def arr(*names):
            for name in names:
                if name in vive_samples:
                    return np.asarray(vive_samples[name], dtype=float)
            return None
        t, x, y = arr("relative_time_s", "elapsed_time", "time_s", "t"), arr("planar_x", "x"), arr("planar_y", "y")
        heading = arr("heading", "planar_heading", "theta")
        source = np.asarray(vive_samples["source_row_index"]) if "source_row_index" in vive_samples else None
        if t is None or x is None or y is None:
            raise ValueError("Vive samples mapping requires time, x and y arrays")
    else:
        values = tuple(vive_samples)
        if len(values) < 3:
            raise ValueError("Vive samples require at least (time, x, y)")
        t, x, y = (np.asarray(values[i], dtype=float) for i in range(3))
        heading = np.asarray(values[3], dtype=float) if len(values) > 3 and values[3] is not None else None

    n = min(len(t), len(x), len(y))
    t, x, y = t[:n], x[:n], y[:n]
    if heading is not None:
        heading = heading[:n]
    if source is not None:
        source = source[:n]
    finite = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    if heading is not None:
        finite &= np.isfinite(heading)
    t, x, y = t[finite], x[finite], y[finite]
    if heading is not None:
        heading = heading[finite]
    if source is not None:
        source = source[finite]
    order = np.argsort(t, kind="mergesort")
    t, x, y = t[order], x[order], y[order]
    if heading is not None:
        heading = heading[order]
    if source is not None:
        source = source[order]
    # Cleaned streams should be unique in time.  Keep the first duplicate
    # deterministically rather than creating an ill-defined interpolator.
    _, unique = np.unique(t, return_index=True)
    unique = np.sort(unique)
    return t[unique], x[unique], y[unique], heading[unique] if heading is not None else None, source[unique] if source is not None else None


def query_reference(method: str, vive_samples: Any, query_times: np.ndarray) -> ReferenceQuery:
    """Query one of the four finite-support Vive reference operators.

    The method factor is deliberately applied to Vive samples only.  Query
    times are not resampled or otherwise altered, and endpoint extrapolation
    is rejected (represented by NaN output and ``valid_mask=False``).
    """
    method = str(method).lower()
    if method not in _REFERENCE_METHODS:
        raise ValueError(f"unsupported Vive reference method: {method!r}")
    t, x, y, heading, source = _coerce_vive_samples(vive_samples)
    q = np.asarray(query_times, dtype=float).reshape(-1)
    valid = np.isfinite(q)
    if len(t) == 0:
        valid[:] = False
    else:
        valid &= (q >= t[0]) & (q <= t[-1])
    extrapolated_count = int((np.isfinite(q) & ~((q >= t[0]) & (q <= t[-1]))).sum()) if len(t) else int(np.isfinite(q).sum())
    out_x = np.full(q.shape, np.nan, dtype=float)
    out_y = np.full(q.shape, np.nan, dtype=float)
    out_h = np.full(q.shape, np.nan, dtype=float) if heading is not None else None
    if valid.any() and len(t):
        tq = q[valid]
        if method == "zoh":
            idx = np.searchsorted(t, tq, side="right") - 1
            idx = np.clip(idx, 0, len(t) - 1)
            out_x[valid], out_y[valid] = x[idx], y[idx]
            if out_h is not None:
                out_h[valid] = heading[idx]
        else:
            if method == "linear" or len(t) < 3:
                fn_x = lambda z: np.interp(z, t, x)
                fn_y = lambda z: np.interp(z, t, y)
                fn_h = (lambda z: np.interp(z, t, np.unwrap(heading))) if heading is not None else None
            elif method == "pchip":
                fn_x, fn_y = PchipInterpolator(t, x, extrapolate=False), PchipInterpolator(t, y, extrapolate=False)
                fn_h = PchipInterpolator(t, np.unwrap(heading), extrapolate=False) if heading is not None else None
            else:
                # CubicSpline needs four points; short cleaned streams fall
                # back to linear while retaining finite-support semantics.
                if len(t) >= 4:
                    fn_x, fn_y = CubicSpline(t, x, bc_type="natural", extrapolate=False), CubicSpline(t, y, bc_type="natural", extrapolate=False)
                    fn_h = CubicSpline(t, np.unwrap(heading), bc_type="natural", extrapolate=False) if heading is not None else None
                else:
                    fn_x = lambda z: np.interp(z, t, x)
                    fn_y = lambda z: np.interp(z, t, y)
                    fn_h = (lambda z: np.interp(z, t, np.unwrap(heading))) if heading is not None else None
            out_x[valid], out_y[valid] = np.asarray(fn_x(tq), float), np.asarray(fn_y(tq), float)
            if out_h is not None and fn_h is not None:
                out_h[valid] = np.asarray(fn_h(tq), float)
    valid &= np.isfinite(out_x) & np.isfinite(out_y)
    if out_h is not None:
        valid &= np.isfinite(out_h)
    return ReferenceQuery(method, q, out_x, out_y, out_h, valid, extrapolated_count, source)

class ContinuousTrajectory:
    def __init__(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, heading: np.ndarray = None):
        """
        Fits a cubic spline to the trajectory (x(t), y(t)) and optionally heading(t).
        """
        # Ensure time is strictly increasing and there are no NaNs
        valid_mask = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
        if heading is not None:
            valid_mask &= np.isfinite(heading)

        t = t[valid_mask]
        x = x[valid_mask]
        y = y[valid_mask]
        if heading is not None:
            heading = heading[valid_mask]

        idx = np.argsort(t)
        t_sorted = t[idx]
        x_sorted = x[idx]
        y_sorted = y[idx]

        if heading is not None:
            # Unwrap heading to avoid spline artifacts at +/- pi
            h_unwrapped = np.unwrap(heading[idx])
        else:
            h_unwrapped = None

        # Remove duplicates
        _, unique_idx = np.unique(t_sorted, return_index=True)
        self.t = t_sorted[unique_idx]
        self.x = x_sorted[unique_idx]
        self.y = y_sorted[unique_idx]
        if h_unwrapped is not None:
            self.heading = h_unwrapped[unique_idx]
        else:
            self.heading = None

        if len(self.t) > 3:
            self.spline_x = CubicSpline(self.t, self.x, bc_type='natural')
            self.spline_y = CubicSpline(self.t, self.y, bc_type='natural')
            if self.heading is not None:
                self.spline_h = CubicSpline(self.t, self.heading, bc_type='natural')
            self.valid = True
        else:
            self.valid = False

    def query(self, t_query: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Queries the continuous representation at arbitrary times."""
        if not self.valid:
            if self.heading is not None:
                return np.zeros_like(t_query), np.zeros_like(t_query), np.zeros_like(t_query)
            return np.zeros_like(t_query), np.zeros_like(t_query)

        qx = self.spline_x(t_query)
        qy = self.spline_y(t_query)

        if self.heading is not None:
            qh = self.spline_h(t_query)
            # Wrap back to [-pi, pi]
            qh = (qh + np.pi) % (2 * np.pi) - np.pi
            return qx, qy, qh

        return qx, qy

def build_reference_trajectory(df_vrmt_full: pd.DataFrame, spatial_params: dict, cfg: Config) -> ContinuousTrajectory:
    """
    Builds the continuous trajectory from the full VRMT stream (lever-arm corrected).
    Spatial alignment to EV3 frame happens during the query phase in alignment.py.
    """
    if df_vrmt_full.empty:
        return ContinuousTrajectory(np.array([]), np.array([]), np.array([]))

    t = df_vrmt_full["relative_time_s"].values

    # 1. Project to 2D and apply lever arm
    rx, ry, rtheta, _ = get_planar_trajectory(cfg, df_vrmt_full)

    # 2. Fit spline (no spatial transform to EV3 frame yet)
    return ContinuousTrajectory(t, rx, ry, heading=rtheta)
