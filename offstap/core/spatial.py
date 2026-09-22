"""
src/spatial.py

Estimates rigid spatial transformation (R, t) from Vive planar coordinates
to the EV3/robot frame ONLY on the calibration interval.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Any, Mapping
import hashlib
import json
from scipy.optimize import least_squares
from offstap.core.config import Config
from offstap.core.lever_arm import apply_lever_arm_2d


class SpatialFit(dict):
    """Dictionary-compatible spatial parameters with fit provenance."""

    @property
    def provenance(self) -> dict:
        return self.get("provenance", {})


def _spatial_provenance(
    ev3_calib: pd.DataFrame,
    vrmt_calib: pd.DataFrame,
    temporal_params: Optional[Mapping[str, Any]],
    clock_params: Optional[Mapping[str, Any]],
) -> dict[str, str]:
    """Create deterministic provenance for one calibration-only fit.

    The temporal mapping is included deliberately: two fits using the same
    calibration rows under different mappings are distinct estimators.
    """
    keys = []
    for frame in (ev3_calib, vrmt_calib):
        if isinstance(frame, pd.DataFrame) and "source_row_index" in frame.columns:
            keys.append(frame["source_row_index"].tolist())
        else:
            keys.append([])
    mapping = {
        "a": float((clock_params or {}).get("scale_a", 1.0)),
        "b": float((clock_params or {}).get("offset_b", 0.0)),
        "temporal": str((temporal_params or {}).get("method_used", "")),
    }
    source_payload = json.dumps({"keys": keys, "mapping": mapping}, sort_keys=True, default=str)
    key_hash = hashlib.sha256(source_payload.encode("utf-8")).hexdigest()
    fit_id = hashlib.sha256(("spatial-fit:" + source_payload).encode("utf-8")).hexdigest()
    return {
        "fit_source": "calibration_source",
        "fit_source_key_hash": key_hash,
        "spatial_fit_id": fit_id,
    }

def fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fits a plane to 3D points, returns (normal, d)"""
    # Centering
    centroid = np.array([np.mean(x), np.mean(y), np.mean(z)])
    points = np.c_[x, y, z] - centroid
    # SVD
    _, _, vh = np.linalg.svd(points)
    normal = vh[2, :]
    if normal[1] < 0: # Ensure normal points "up" (assuming Y is generally up in VRMT)
        normal = -normal
    d = -np.dot(normal, centroid)
    return normal, d

def get_planar_trajectory(cfg: Config, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projects Vive 3D poses to a 2D plane and applies lever-arm offset.
    Returns (x_robot, y_robot, theta).
    """
    x = df["x"].values
    z = df["z"].values

    # Calculate yaw from quaternion (assuming Z-up or Y-up)
    # Simple approx: heading in the projection plane.
    # For VRMT usually rotation around Y axis.
    qy_value = df.get("qy", np.zeros(len(df)))
    qw_value = df.get("qw", np.ones(len(df)))
    qy = qy_value.to_numpy() if hasattr(qy_value, "to_numpy") else np.asarray(qy_value)
    qw = qw_value.to_numpy() if hasattr(qw_value, "to_numpy") else np.asarray(qw_value)
    theta = 2 * np.arctan2(qy, qw)

    mode = cfg.geometry.get("planar_projection_mode", "direct")

    if mode == "fitted":
        y = df["y"].values
        normal, d = fit_plane(x, y, z)
        # Distance to plane:
        dists = np.dot(np.c_[x, y, z], normal) + d
        planarity_residual = float(np.sqrt(np.mean(dists**2)))
        proj_x = x
        proj_y = z
    else:
        # direct mode
        planarity_residual = 0.0
        x_col = cfg.geometry.get("direct_map", {}).get("planar_x", "x")
        y_col = cfg.geometry.get("direct_map", {}).get("planar_y", "z")
        proj_x = df[x_col].values
        proj_y = df[y_col].values

    if cfg.geometry.get("apply_lever_arm_correction", True):
        lx, ly, _ = cfg.geometry.get("lever_arm_offset_xyz", [0.0, 0.0, 0.0])
        rx, ry = apply_lever_arm_2d(proj_x, proj_y, theta, lx, ly)
    else:
        rx, ry = proj_x, proj_y

    return rx, ry, theta, planarity_residual

def _rigid_se2(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a proper unit-scale rigid transform from source to target."""
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    h = (source - source_centroid).T @ (target - target_centroid)
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    return rotation, translation


def _estimate_rigid_calibration_transform(
    ev3_calib: pd.DataFrame,
    vrmt_calib: pd.DataFrame,
    cfg: Config,
    temporal_params: Optional[Dict[str, float]],
    clock_params: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Fit SE(2) on all finite time-matched calibration samples."""
    invalid = {
        "theta_rad": 0.0, "tx": 0.0, "ty": 0.0, "ex0": 0.0, "ey0": 0.0,
        "vx0": 0.0, "vy0": 0.0, "start_residual": np.nan,
        "trajectory_residual": np.nan, "heading_residual_deg": np.nan,
        "planarity_residual": np.nan, "vrmt_flip_y": 0.0,
        "method_used": "calibration_partition_rigid_se2",
    }
    required_ev3 = {"local_time_s", "odometry_x", "odometry_y"}
    if ev3_calib.empty or vrmt_calib.empty or not required_ev3.issubset(ev3_calib.columns):
        return invalid
    rx, ry, _, planarity_residual = get_planar_trajectory(cfg, vrmt_calib)
    if cfg.geometry.get("spatial_vrmt_flip_y", True):
        ry = -ry
    t_v = pd.to_numeric(vrmt_calib["relative_time_s"], errors="coerce").to_numpy(float)
    vfinite = np.isfinite(t_v) & np.isfinite(rx) & np.isfinite(ry)
    t_v, rx, ry = t_v[vfinite], rx[vfinite], ry[vfinite]
    if len(t_v) < 3:
        return invalid | {"planarity_residual": planarity_residual}
    order = np.argsort(t_v, kind="stable")
    t_v, rx, ry = t_v[order], rx[order], ry[order]
    _, unique = np.unique(t_v, return_index=True)
    t_v, rx, ry = t_v[unique], rx[unique], ry[unique]

    t_e = pd.to_numeric(ev3_calib["local_time_s"], errors="coerce").to_numpy(float)
    ex = pd.to_numeric(ev3_calib["odometry_x"], errors="coerce").to_numpy(float)
    ey = pd.to_numeric(ev3_calib["odometry_y"], errors="coerce").to_numpy(float)
    temporal_params = temporal_params or {}
    clock_params = clock_params or {}
    a = float(clock_params.get("scale_a", 1.0))
    b = float(clock_params.get("offset_b", temporal_params.get("b_final", temporal_params.get("b_onset", 0.0))))
    query = a * t_e + b
    keep = (
        np.isfinite(query) & np.isfinite(ex) & np.isfinite(ey)
        & (query >= t_v[0]) & (query <= t_v[-1])
    )
    if keep.sum() < 3:
        return invalid | {"planarity_residual": planarity_residual}
    source = np.column_stack((np.interp(query[keep], t_v, rx), np.interp(query[keep], t_v, ry)))
    target = np.column_stack((ex[keep], ey[keep]))
    rotation, translation = _rigid_se2(source, target)
    prediction = source @ rotation.T + translation
    residual = np.linalg.norm(target - prediction, axis=1)
    theta = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return {
        "theta_rad": theta,
        "tx": float(translation[0]), "ty": float(translation[1]),
        "ex0": float(translation[0]), "ey0": float(translation[1]),
        "vx0": 0.0, "vy0": 0.0,
        "start_residual": float(residual[0]),
        "trajectory_residual": float(np.sqrt(np.mean(residual ** 2))),
        "heading_residual_deg": np.nan,
        "planarity_residual": float(planarity_residual),
        "vrmt_flip_y": 1.0 if cfg.geometry.get("spatial_vrmt_flip_y", True) else 0.0,
        "method_used": "calibration_partition_rigid_se2",
        "num_fit_samples": int(keep.sum()),
    }


def _estimate_spatial_transform_raw(
    ev3_calib: pd.DataFrame,
    vrmt_calib: pd.DataFrame,
    cfg: Config,
    temporal_params: Optional[Dict[str, float]] = None,
    clock_params: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Estimates rigid transform mapping VRMT planar coords to EV3 odometry frame.
    Aligns the initial `align_dist` (0.15m) movement vector of VRMT to the EV3 vector.
    """
    if cfg.geometry.get("spatial_fit_method", "initial_position_heading") == "calibration_partition_rigid_se2":
        return _estimate_rigid_calibration_transform(
            ev3_calib, vrmt_calib, cfg, temporal_params, clock_params
        )

    if ev3_calib.empty or vrmt_calib.empty:
        return {"theta_rad": 0.0, "tx": 0.0, "ty": 0.0, "ex0": np.nan, "ey0": np.nan, "vx0": np.nan, "vy0": np.nan, "start_residual": np.nan, "planarity_residual": np.nan, "vrmt_flip_y": 0.0}

    # Get EV3 start position (assume 0,0) and trajectory
    if "odometry_x" in ev3_calib.columns and "odometry_y" in ev3_calib.columns:
        ev3_x = ev3_calib["odometry_x"].values
        ev3_y = ev3_calib["odometry_y"].values
    else:
        # Fallback to NaN if odometry isn't available
        ev3_x = np.full(len(ev3_calib), np.nan)
        ev3_y = np.full(len(ev3_calib), np.nan)

    # Get VRMT planar trajectory and map it onto EV3 calibration timestamps
    # before deriving the initial heading.  Previously this branch paired raw
    # row order, silently ignoring affine/constant clock parameters.
    rx, ry, _, planarity_residual = get_planar_trajectory(cfg, vrmt_calib)

    align_dist = cfg.geometry.get("spatial_align_dist_m", 0.075)

    # Check for vertical flip
    vrmt_flip_y = cfg.geometry.get("spatial_vrmt_flip_y", True)
    if vrmt_flip_y:
        ry = -ry

    temporal_params = temporal_params or {}
    clock_params = clock_params or {}
    if "relative_time_s" in vrmt_calib.columns and "local_time_s" in ev3_calib.columns:
        tv = pd.to_numeric(vrmt_calib["relative_time_s"], errors="coerce").to_numpy(float)
        te = pd.to_numeric(ev3_calib["local_time_s"], errors="coerce").to_numpy(float)
        a = float(clock_params.get("scale_a", 1.0))
        b = float(clock_params.get("offset_b", temporal_params.get("b_final", temporal_params.get("b_onset", 0.0))))
        finite = np.isfinite(tv) & np.isfinite(rx) & np.isfinite(ry)
        if finite.sum() >= 2:
            order = np.argsort(tv[finite], kind="stable")
            tvf, rxf, ryf = tv[finite][order], rx[finite][order], ry[finite][order]
            tvf, unique = np.unique(tvf, return_index=True)
            rxf, ryf = rxf[unique], ryf[unique]
            query = a * te + b
            rx = np.interp(query, tvf, rxf, left=np.nan, right=np.nan)
            ry = np.interp(query, tvf, ryf, left=np.nan, right=np.nan)

    valid_pair = np.isfinite(ev3_x) & np.isfinite(ev3_y) & np.isfinite(rx) & np.isfinite(ry)
    ev3_x, ev3_y, rx, ry = ev3_x[valid_pair], ev3_y[valid_pair], rx[valid_pair], ry[valid_pair]
    if len(ev3_x) < 2:
        return {"theta_rad": 0.0, "tx": 0.0, "ty": 0.0, "ex0": float(ev3_x[0]) if len(ev3_x) > 0 else np.nan, "ey0": float(ev3_y[0]) if len(ev3_y) > 0 else np.nan, "vx0": float(rx[0]) if len(rx) > 0 else np.nan, "vy0": float(ry[0]) if len(ry) > 0 else np.nan, "start_residual": np.nan, "planarity_residual": planarity_residual, "vrmt_flip_y": 1.0 if vrmt_flip_y else 0.0}

    # 1. Compute cumulative distance for EV3
    ev3_dx = np.diff(ev3_x)
    ev3_dy = np.diff(ev3_y)
    ev3_dist = np.concatenate(([0.0], np.cumsum(np.sqrt(ev3_dx**2 + ev3_dy**2))))

    # 2. Compute cumulative distance for VRMT
    vrmt_dx = np.diff(rx)
    vrmt_dy = np.diff(ry)
    vrmt_dist = np.concatenate(([0.0], np.cumsum(np.sqrt(vrmt_dx**2 + vrmt_dy**2))))

    # 3. Find indices for align_dist
    ev3_idx = np.searchsorted(ev3_dist, align_dist)
    vrmt_idx = np.searchsorted(vrmt_dist, align_dist)

    # Fallback if trajectory is shorter than align_dist
    if ev3_idx >= len(ev3_x):
        ev3_idx = len(ev3_x) - 1
    if vrmt_idx >= len(rx):
        vrmt_idx = len(rx) - 1

    if ev3_idx == 0 or vrmt_idx == 0:
        return {"theta_rad": 0.0, "tx": 0.0, "ty": 0.0, "ex0": float(ev3_x[0]), "ey0": float(ev3_y[0]), "vx0": float(rx[0]), "vy0": float(ry[0]), "start_residual": np.nan, "planarity_residual": planarity_residual, "vrmt_flip_y": 1.0 if vrmt_flip_y else 0.0}

    # 4. Extract vectors from origin to the align_dist point
    ev3_vec = np.array([ev3_x[ev3_idx] - ev3_x[0], ev3_y[ev3_idx] - ev3_y[0]])
    vrmt_vec = np.array([rx[vrmt_idx] - rx[0], ry[vrmt_idx] - ry[0]])

    # Compute angles using arctan2
    ev3_angle = np.arctan2(ev3_vec[1], ev3_vec[0])
    vrmt_angle = np.arctan2(vrmt_vec[1], vrmt_vec[0])

    # Rotation required to map VRMT vector to EV3 vector
    theta = ev3_angle - vrmt_angle

    # 5. Compute true start residual (error at the tip of the alignment vector)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    vrmt_vec_rot = np.array([
        vrmt_vec[0] * cos_t - vrmt_vec[1] * sin_t,
        vrmt_vec[0] * sin_t + vrmt_vec[1] * cos_t
    ])
    start_residual = float(np.sqrt((vrmt_vec_rot[0] - ev3_vec[0])**2 + (vrmt_vec_rot[1] - ev3_vec[1])**2))

    return {
        "theta_rad": float(theta),
        "tx": 0.0, # Handled explicitly in apply
        "ty": 0.0,
        "ex0": float(ev3_x[0]),
        "ey0": float(ev3_y[0]),
        "vx0": float(rx[0]),
        "vy0": float(ry[0]),
        "start_residual": start_residual,
        "planarity_residual": planarity_residual,
        "vrmt_flip_y": 1.0 if vrmt_flip_y else 0.0,
        "method_used": "initial_position_heading",
        "trajectory_residual": np.nan,
        "heading_residual_deg": 0.0,
    }


def estimate_spatial_transform(
    ev3_calib: pd.DataFrame,
    vrmt_calib: pd.DataFrame,
    cfg: Config,
    temporal_params: Optional[Dict[str, float]] = None,
    clock_params: Optional[Dict[str, float]] = None,
) -> SpatialFit:
    """Fit a spatial transform and attach calibration provenance.

    The returned object remains a ``dict`` for backwards compatibility while
    exposing ``.provenance`` for audit and H1 arm-isolation checks.
    """
    params = _estimate_spatial_transform_raw(
        ev3_calib, vrmt_calib, cfg, temporal_params, clock_params
    )
    fit = SpatialFit(params)
    fit["provenance"] = _spatial_provenance(
        ev3_calib, vrmt_calib, temporal_params, clock_params
    )
    return fit

def apply_spatial_transform(rx: np.ndarray, ry: np.ndarray, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Applies the spatial parameters to a trajectory."""
    vx0 = params.get("vx0", 0.0)
    vy0 = params.get("vy0", 0.0)
    ex0 = params.get("ex0", 0.0)
    ey0 = params.get("ey0", 0.0)
    vrmt_flip_y = params.get("vrmt_flip_y", 0.0) > 0.5

    if vrmt_flip_y:
        ry = -ry

    # 1. Translate VRMT so its start is at (0, 0)
    x_shifted = rx - vx0
    y_shifted = ry - vy0

    # 2. Rotate by theta
    theta = params["theta_rad"]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x_rot = x_shifted * cos_t - y_shifted * sin_t
    y_rot = x_shifted * sin_t + y_shifted * cos_t

    # 3. Translate to EV3 origin
    x_final = x_rot + ex0
    y_final = y_rot + ey0

    return x_final, y_final
