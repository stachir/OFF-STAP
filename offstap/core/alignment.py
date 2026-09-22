"""
src/alignment.py

Queries the continuous Vive trajectory at the transformed EV3 timestamps.
Generates the final AlignedRecord structures.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Mapping, Sequence, Optional
from offstap.core.config import Config
from offstap.core.trajectory_reference import ContinuousTrajectory, query_reference
from offstap.core.spatial import apply_spatial_transform, get_planar_trajectory
from offstap.core.support import CommonSupport, intersect_support


def filter_common_support(
    arms: Mapping[str, pd.DataFrame],
    key_columns: Sequence[str] = ("source_row_index",),
) -> tuple[dict[str, pd.DataFrame], CommonSupport]:
    """Filter every arm to one exact finite support.

    Validation runs before any intersection.  Therefore a duplicate or
    non-finite source key invalidates the comparison and returns empty frames
    while preserving the :class:`CommonSupport` evidence for the caller.
    Valid arms are returned in the canonical numeric key order.
    """
    support = intersect_support(arms, key_columns)
    columns = tuple(key_columns)
    filtered: dict[str, pd.DataFrame] = {}
    if support.status != "valid":
        for name, frame in arms.items():
            filtered[str(name)] = frame.iloc[0:0].copy()
        return filtered, support

    common_tuples = list(support.common_keys.itertuples(index=False, name=None))
    order = {key: position for position, key in enumerate(common_tuples)}
    for name, frame in arms.items():
        validation = support.validations[str(name)]
        positions = [order.get(key, -1) for key in validation.canonical_keys.itertuples(index=False, name=None)]
        selected = [index for index, rank in enumerate(positions) if rank >= 0]
        selected.sort(key=lambda index: positions[index])
        filtered[str(name)] = frame.iloc[selected].copy()
    return filtered, support


# A descriptive alias used by downstream paired-comparison branches.
apply_common_support = filter_common_support

def get_common_time_overlap(t_ev3_mapped: np.ndarray, t_vrmt: np.ndarray) -> Tuple[float, float]:
    """Finds the overlapping time window."""
    start = max(t_ev3_mapped[0], t_vrmt[0])
    end = min(t_ev3_mapped[-1], t_vrmt[-1])
    return start, end



def _preserve_source_rows(
    aligned: pd.DataFrame,
    source_rows: Optional[Any],
) -> pd.DataFrame:
    """Carry persisted source identity/partition metadata into aligned rows.

    ``source_rows`` is normally the EV3 frame persisted by the partition stage
    (or a three-column subset of it).  The join is performed on the stable
    ``source_row_index`` rather than the post-trim positional index.  A mapping
    keyed by source row index is also accepted for lightweight callers.
    Existing values are retained unless they are missing in the aligned frame.
    """
    if source_rows is None or aligned.empty:
        return aligned

    metadata_cols = ("source_row_index", "source_timestamp_ns", "partition")
    if isinstance(source_rows, pd.DataFrame):
        src = source_rows.copy()
        if "source_row_index" in src.columns and "source_row_index" in aligned.columns:
            src = src.drop_duplicates("source_row_index", keep="first")
            # ``source_row_index`` is the join key and therefore no longer a
            # data column after ``set_index``; only carry the remaining
            # persisted metadata fields.
            cols = [c for c in metadata_cols[1:] if c in src.columns]
            if cols:
                lookup = src.set_index("source_row_index")[cols]
                keys = aligned["source_row_index"]
                for col in cols:
                    values = lookup[col].to_dict()
                    mapped = keys.map(
                        lambda key: values.get(key, values.get(str(key), np.nan))
                    )
                    if col not in aligned.columns:
                        aligned[col] = mapped
                    else:
                        aligned[col] = aligned[col].where(aligned[col].notna(), mapped)
        elif len(src) == len(aligned):
            # Positional fallback is safe only when the caller explicitly
            # supplies one metadata row per already-trimmed aligned row.
            for col in metadata_cols:
                if col in src.columns:
                    aligned[col] = src[col].to_numpy()
        return aligned

    if isinstance(source_rows, Mapping) and "source_row_index" in aligned.columns:
        for col in metadata_cols[1:]:
            values = []
            found = False
            for key in aligned["source_row_index"]:
                item = source_rows.get(key, source_rows.get(str(key)))
                if isinstance(item, Mapping):
                    value = item.get(col, np.nan)
                elif col == "partition":
                    value = item
                else:
                    value = np.nan
                found = found or pd.notna(value)
                values.append(value)
            if found:
                mapped = pd.Series(values, index=aligned.index)
                if col not in aligned.columns:
                    aligned[col] = mapped
                else:
                    aligned[col] = aligned[col].where(aligned[col].notna(), mapped)
        return aligned

    # A sequence of row dictionaries is convenient for callers that read the
    # persisted ledger as records rather than a DataFrame.  Positional use is
    # accepted only when cardinality already matches the aligned frame.
    if not isinstance(source_rows, (str, bytes)):
        try:
            records = list(source_rows)
        except TypeError:
            records = []
        if len(records) == len(aligned) and records and all(isinstance(item, Mapping) for item in records):
            for col in metadata_cols:
                values = [item.get(col, np.nan) for item in records]
                if any(pd.notna(value) for value in values):
                    if col not in aligned.columns:
                        aligned[col] = values
                    else:
                        aligned[col] = aligned[col].where(aligned[col].notna(), values)
    return aligned


def generate_aligned_trajectory(df_ev3: pd.DataFrame, df_vrmt: pd.DataFrame,
                                spline_traj: ContinuousTrajectory,
                                temporal_params: Dict[str, float],
                                clock_params: Dict[str, float],
                                spatial_params: Dict[str, float],
                                cfg: Config,
                                source_rows: Optional[Any] = None) -> pd.DataFrame:
    """
    1. Transforms EV3 time to Vive time using clock model.
    2. Queries Vive continuous trajectory.
    3. Trims to overlapping time window.
    4. Applies spatial transformation to match initial 0.15m vectors.
    """
    if df_ev3.empty or not spline_traj.valid:
        return pd.DataFrame()

    # Transform EV3 time: tau = a * t_ev3 + b
    t_ev3 = df_ev3["local_time_s"].values
    a = clock_params.get("scale_a", 1.0)
    b = clock_params.get("offset_b", 0.0)

    t_ev3_mapped = a * t_ev3 + b

    # Overlap interval
    t_vrmt = df_vrmt["relative_time_s"].values
    start, end = get_common_time_overlap(t_ev3_mapped, t_vrmt)

    if end <= start:
        return pd.DataFrame()

    mask = (t_ev3_mapped >= start) & (t_ev3_mapped <= end)
    df_aligned = df_ev3[mask].copy().reset_index(drop=True)
    # Source identities and the persisted calibration/evaluation membership
    # must survive trimming and interpolation.  This metadata is consumed by
    # primary held-out metrics and exact-support comparisons downstream.
    df_aligned = _preserve_source_rows(df_aligned, source_rows)
    t_query = t_ev3_mapped[mask]

    # Keep the stable source identity visible to every downstream metric.  A
    # generated positional index is only a direct-call fallback; loader
    # supplied identities are never replaced.
    if "source_row_index" in df_aligned.columns:
        numeric_keys = pd.to_numeric(df_aligned["source_row_index"], errors="coerce")
        # Preserve textual/non-numeric identities verbatim rather than
        # replacing them with all-NA nullable integers.
        df_aligned["support_key"] = (
            numeric_keys.astype("Int64")
            if numeric_keys.notna().all()
            else df_aligned["source_row_index"].copy()
        )
    else:
        df_aligned["support_key"] = np.arange(len(df_aligned), dtype=np.int64)

    # The frozen production default is the natural-cubic continuous query.
    # A comparator can opt into an explicit ZOH reference policy without
    # changing any primary caller that does not set ``cfg.preprocessing``.
    preprocessing = getattr(cfg, "preprocessing", None) or {}
    reference_policy = str(preprocessing.get("resampling_method", "spline")).lower()
    if reference_policy == "zoh":
        # ``df_vrmt`` is the raw cleaned tracker frame, whose ``y`` column is
        # vertical under the frozen direct projection.  Build the same planar
        # X/Z representation used to construct the frozen spline before ZOH
        # querying it.
        planar_x, planar_y, _, _ = get_planar_trajectory(cfg, df_vrmt)
        vive_planar = pd.DataFrame({
            "relative_time_s": df_vrmt["relative_time_s"].to_numpy(),
            "planar_x": planar_x,
            "planar_y": planar_y,
        })
        reference = query_reference("zoh", vive_planar, t_query)
        x_query, y_query, h_query = reference.values
    else:
        reference_policy = "spline"
        res = spline_traj.query(t_query)
        x_query = res[0]
        y_query = res[1]
        h_query = res[2] if len(res) == 3 else None

    # Apply spatial alignment
    if "odometry_x" in df_aligned.columns and "odometry_y" in df_aligned.columns:
        x_query, y_query = apply_spatial_transform(x_query, y_query, spatial_params)
        if h_query is not None:
            vrmt_flip_y = spatial_params.get("vrmt_flip_y", 0.0) > 0.5
            if vrmt_flip_y:
                h_query = -h_query
            theta_align = spatial_params.get("theta_rad", 0.0)
            h_query = h_query + theta_align
            h_query = (h_query + np.pi) % (2 * np.pi) - np.pi

    trial_id = str(df_aligned["trial_id"].iloc[0]) if "trial_id" in df_aligned.columns and len(df_aligned) else ""
    parts = trial_id.split("_")
    df_aligned["drive_id"] = parts[0] if len(parts) > 0 and parts[0] else "unknown"
    df_aligned["route_id"] = parts[1] if len(parts) > 1 and parts[1] else "unknown"
    df_aligned["scenario_id"] = "_".join(parts[:2]) if len(parts) >= 2 else "unknown"
    df_aligned["repetition_id"] = cfg._config.get("repetition_id", "") if hasattr(cfg, "_config") else ""
    df_aligned["t_common_s"] = t_query
    df_aligned["ev3_time_original_s"] = df_aligned["local_time_s"]
    df_aligned["vive_time_original_s"] = t_query
    df_aligned["vrmt_aligned_x"] = x_query
    df_aligned["vrmt_aligned_y"] = y_query
    if h_query is not None:
        df_aligned["vrmt_aligned_heading"] = h_query

    if "odometry_x" in df_aligned.columns and "odometry_y" in df_aligned.columns:
        df_aligned["residual_x"] = df_aligned["odometry_x"] - df_aligned["vrmt_aligned_x"]
        df_aligned["residual_y"] = df_aligned["odometry_y"] - df_aligned["vrmt_aligned_y"]
        df_aligned["residual_norm"] = np.sqrt(df_aligned["residual_x"]**2 + df_aligned["residual_y"]**2)
        df_aligned["sample_valid"] = np.isfinite(df_aligned["residual_norm"])
    else:
        df_aligned["residual_x"] = np.nan
        df_aligned["residual_y"] = np.nan
        df_aligned["residual_norm"] = np.nan
        df_aligned["sample_valid"] = False

    # Compute real Vive velocity
    dt = np.diff(t_query)
    dx = np.diff(x_query)
    dy = np.diff(y_query)
    v_vive = np.zeros(len(x_query))
    valid_dt = dt > 0
    if np.any(valid_dt):
        v_vive[1:][valid_dt] = np.sqrt(dx[valid_dt]**2 + dy[valid_dt]**2) / dt[valid_dt]
    df_aligned["vive_velocity_ct"] = v_vive

    # Extract EV3 signals if present
    # Assume we have left_motor_speed, right_motor_speed or similar.
    # Let's derive turning / straight from ev3_signals or fallback to basic motor speeds if available.
    phase_arr = np.full(len(df_aligned), "uncertain", dtype=object)

    # Try to find speed magnitude
    ev3_speed = None
    if "speed_magnitude" in df_aligned.columns:
        ev3_speed = df_aligned["speed_magnitude"].values
    elif "motor_speed" in df_aligned.columns:
        ev3_speed = df_aligned["motor_speed"].abs().values

    # Try to find turn rate
    ev3_turn = None
    if "turn_rate_proxy" in df_aligned.columns:
        ev3_turn = df_aligned["turn_rate_proxy"].values

    if ev3_speed is not None:
        for i in range(len(df_aligned)):
            v_e = ev3_speed[i]
            v_v = v_vive[i]
            turn = np.abs(ev3_turn[i]) if ev3_turn is not None else 0.0

            # thresholds
            if v_e < 1.0 and v_v < 0.05: # low speed on both -> stationary
                phase_arr[i] = "stationary"
            elif v_e >= 1.0 and v_v >= 0.05: # both moving
                if turn > 5.0:
                    phase_arr[i] = "turning"
                else:
                    phase_arr[i] = "straight"
            else:
                phase_arr[i] = "uncertain"
    else:
        phase_arr[:] = "uncertain"

    df_aligned["movement_phase"] = phase_arr
    df_aligned["motion_phase"] = phase_arr
    df_aligned["alignment_method"] = temporal_params.get("method_used", "unspecified")
    df_aligned["clock_model"] = clock_params.get("selected_model", "constant_offset")
    df_aligned["spatial_config_id"] = cfg.config_id
    df_aligned["temporal_config_id"] = cfg.config_id
    df_aligned["representation_method"] = reference_policy
    df_aligned["processing_config_id"] = cfg.config_id
    df_aligned["provenance_id"] = f"{cfg.config_id}:{trial_id}" if trial_id else cfg.config_id

    return df_aligned
