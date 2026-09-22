"""
src/loaders.py

Loads raw EV3 and Vive data, standardizes them, and saves them to Parquet format.
"""
import os
import json
import math
import ast
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import pandas as pd
import numpy as np
from typing import Dict, Any
from offstap.core.config import Config
from offstap.core.schema import standardize_columns, generate_schema_report


def _configured_time_factor(cfg: Config, key: str) -> float:
    """Return an explicitly configured raw-time-to-seconds factor.

    Timestamp units are part of the input contract.  In particular, loaders
    must not guess whether a value is seconds, milliseconds, or another unit.
    A missing/non-finite factor is therefore an input-contract error rather
    than silently defaulting to seconds.
    """
    if key not in cfg.units:
        raise ValueError(f"Missing explicit timestamp conversion '{key}'")
    factor = float(cfg.units[key])
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"Timestamp conversion '{key}' must be a positive finite number")
    return factor


def _timestamp_ns(seconds: pd.Series) -> pd.Series:
    """Convert seconds to deterministic integer nanosecond keys.

    ``np.rint`` specifies deterministic nearest-integer rounding (ties to
    even) before the nullable integer cast.  Non-numeric/invalid values remain
    missing and are excluded from duplicate-key accounting.
    """
    numeric = pd.to_numeric(seconds, errors="coerce")
    rounded = np.rint(numeric.to_numpy(dtype=float, na_value=np.nan) * 1_000_000_000)
    return pd.Series(rounded, index=seconds.index, name="source_timestamp_ns").astype("Int64")


def _exact_timestamp_ns(values: pd.Series, factor: float) -> pd.Series:
    """Convert raw numeric/datetime values to nanoseconds without float loss."""
    result: list[object] = []
    factor_decimal = Decimal(str(factor))
    for value in values.tolist():
        if pd.isna(value):
            result.append(pd.NA)
            continue
        try:
            if isinstance(value, (str, bytes)):
                # Numeric text is a configured-unit value, not a calendar
                # year (e.g. ``"1000"`` must remain 1000 seconds).
                try:
                    raw_decimal = Decimal(value.decode() if isinstance(value, bytes) else value)
                    if raw_decimal.is_finite():
                        result.append(int((raw_decimal * factor_decimal * Decimal(1_000_000_000)).to_integral_value(rounding=ROUND_HALF_EVEN)))
                        continue
                except InvalidOperation:
                    pass
                # Parse fractional seconds explicitly so scalar Timestamp
                # conversion cannot truncate beyond pandas' native precision.
                text = value.decode() if isinstance(value, bytes) else value
                match = re.match(r"^(.*?)(?:[,.](\d+))?$", text.strip())
                base_text = match.group(1) if match else text
                fraction = (match.group(2) if match else "")
                parsed = pd.to_datetime(base_text, errors="coerce", utc=True)
                if not pd.isna(parsed):
                    base_ns = int(parsed.value) + int((fraction[:9]).ljust(9, "0") or 0)
                    result.append(int((Decimal(base_ns) * factor_decimal).to_integral_value(rounding=ROUND_HALF_EVEN)))
                    continue
            raw_decimal = Decimal(str(value))
            if not raw_decimal.is_finite():
                result.append(pd.NA)
                continue
            result.append(int((raw_decimal * factor_decimal * Decimal(1_000_000_000)).to_integral_value(rounding=ROUND_HALF_EVEN)))
        except (InvalidOperation, TypeError, ValueError, OverflowError):
            result.append(pd.NA)
    return pd.Series(result, index=values.index, name="source_timestamp_ns").astype("Int64")


def _duplicate_key_counts(keys: pd.Series) -> tuple[int, int]:
    """Return (number of duplicated key values, number of affected rows)."""
    valid = keys.dropna()
    duplicated = valid[valid.duplicated(keep=False)]
    return int(duplicated.nunique()), int(len(duplicated))


def _attach_source_identity(
    df: pd.DataFrame,
    *,
    seconds_col: str,
    exact_ns_col: str | None = None,
    exact_scale: float = 1.0,
    raw_time_col: str | None = None,
) -> pd.DataFrame:
    """Attach deterministic source identity columns and duplicate evidence."""
    if raw_time_col and raw_time_col in df.columns:
        df["source_timestamp_ns"] = _exact_timestamp_ns(df[raw_time_col], exact_scale)
    elif exact_ns_col and exact_ns_col in df.columns:
        exact_values = pd.to_numeric(df[exact_ns_col], errors="coerce") * exact_scale
        df["source_timestamp_ns"] = pd.Series(
            np.rint(exact_values.to_numpy(dtype=float, na_value=np.nan)),
            index=df.index,
            name="source_timestamp_ns",
        ).astype("Int64")
    else:
        df["source_timestamp_ns"] = _timestamp_ns(df[seconds_col])
    duplicate_count, duplicate_rows = _duplicate_key_counts(df["source_timestamp_ns"])
    # attrs intentionally carry diagnostics without changing the row-level
    # table.  Duplicate rows are retained for support validation to reject.
    df.attrs["duplicate_source_key_count"] = duplicate_count
    df.attrs["duplicate_source_key_rows"] = duplicate_rows
    return df


def _coerce_wall_time_seconds(values: pd.Series, factor: float) -> pd.Series:
    """Convert numeric or timestamp-like values using the configured factor."""
    numeric = pd.to_numeric(values, errors="coerce")
    missing = numeric.isna()
    if missing.any():
        parsed = pd.to_datetime(values.where(missing), errors="coerce", utc=True)
        # ``view``/astype is deterministic for UTC timestamps and avoids
        # machine-local timezone assumptions.  NaT is converted back to NaN.
        parsed_ns = parsed.dt.as_unit("ns").astype("int64", copy=False).astype("float64")
        parsed_ns[parsed.isna()] = np.nan
        numeric = numeric.where(~missing, parsed_ns / 1_000_000_000)
    return numeric * factor

def load_and_standardize_ev3(cfg: Config, filepath: str, trial_id: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath)
    except Exception:
        return pd.DataFrame()

    # Preserve the raw row identity before any sorting, filtering, or
    # standardization.  This is the stable key used by downstream support
    # validation.
    df["source_row_index"] = np.arange(len(df), dtype=np.int64)

    # Standardize names (values are not changed by this operation).
    aliases = cfg.schema.get("ev3", {}).get("aliases", {})
    df = standardize_columns(df, aliases)

    time_col = cfg.schema.get("ev3", {}).get("time_column", "time")

    # Ensure time exists
    if time_col not in df.columns:
        raise ValueError(f"Missing time column '{time_col}' in {filepath}")

    # Preserve the untouched source timestamp before conversion/sorting.
    df["source_timestamp_raw"] = df[time_col].copy()

    # Convert time units using the explicit configuration; units are never
    # inferred from the magnitude or spacing of the values.
    time_factor = _configured_time_factor(cfg, "ev3_time_to_s")
    raw_numeric = pd.to_numeric(df[time_col], errors="coerce")
    invalid_timestamp_count = int(raw_numeric.isna().sum())
    df["local_time_s"] = raw_numeric * time_factor

    # Create integer nanosecond keys before sorting.  Duplicate timestamps are
    # retained so that support validation can record/reject them explicitly.
    df = _attach_source_identity(df, seconds_col="local_time_s", raw_time_col="source_timestamp_raw", exact_scale=time_factor)
    df.attrs["invalid_timestamp_count"] = invalid_timestamp_count
    df.attrs["timestamp_status"] = "invalid_timestamp_values" if invalid_timestamp_count else "ok"
    df = df.sort_values(by="local_time_s", kind="mergesort").reset_index(drop=True)

    # Add standard odometry computation if speeds exist
    if "left_speed" in df.columns and "right_speed" in df.columns:
        WHEEL_RADIUS = 0.028  # meters
        TRACK_WIDTH = 0.120   # meters

        # Convert speed to numeric in case of string parsing issues in extended logs
        df["left_speed"] = pd.to_numeric(df["left_speed"], errors="coerce")
        df["right_speed"] = pd.to_numeric(df["right_speed"], errors="coerce")

        # Convert speed from deg/s to rad/s
        omega_l = np.radians(df["left_speed"].fillna(0.0))
        omega_r = np.radians(df["right_speed"].fillna(0.0))

        # Linear velocities of each wheel (m/s)
        v_l = omega_l * WHEEL_RADIUS
        v_r = omega_r * WHEEL_RADIUS

        # Robot linear and angular velocity
        v = (v_r + v_l) / 2.0
        w = (v_r - v_l) / TRACK_WIDTH

        # Time deltas
        dt = np.diff(df["local_time_s"].values, prepend=df["local_time_s"].values[0])

        # Dead Reckoning Integration
        theta = np.cumsum(w * dt)
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt

        df["odometry_x"] = np.cumsum(dx)
        df["odometry_y"] = np.cumsum(dy)
        df["odometry_theta"] = theta

    # Add trial ID
    df["trial_id"] = trial_id

    return df

def load_and_standardize_vrmt(cfg: Config, filepath: str, trial_id: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        return pd.DataFrame()

    # Read the first line to guess format
    with open(filepath, 'r') as f:
        first_line = f.readline()

    if not first_line.strip():
        return pd.DataFrame()

    if ';' in first_line and ('INFO' in first_line or 'tracker' in first_line):
        # Custom tracker log format
        cols = [
            "timestamp", "logger", "level",
            "x", "y", "z",
            "roll", "pitch", "yaw",
            "qx", "qy", "qz", "qw", "extra1", "extra2" # approximate mapping for trailing columns
        ]
        # Pad columns if necessary
        num_cols = len(first_line.split(';'))
        if num_cols > len(cols):
            cols.extend([f"col_{i}" for i in range(len(cols), num_cols)])
        cols = cols[:num_cols]

        try:
            df = pd.read_csv(filepath, sep=';', names=cols)
            df["source_row_index"] = np.arange(len(df), dtype=np.int64)
            # Keep the source string untouched; parse a separate working
            # column using UTC for deterministic epoch conversion.
            df["source_timestamp_raw"] = df["timestamp"].copy()
            parsed = pd.to_datetime(
                df["timestamp"].astype(str).str.strip(),
                format="%Y-%m-%d %H:%M:%S,%f",
                errors="coerce",
                utc=True,
            )
            exact_ns = parsed.dt.as_unit("ns").astype("int64", copy=False)
            df["_source_timestamp_ns_exact"] = pd.Series(exact_ns, index=df.index).where(
                ~parsed.isna(), pd.NA
            ).astype("Int64")
            df["timestamp"] = parsed.dt.as_unit("ns").astype("int64", copy=False).astype("float64") / 1_000_000_000
            df.loc[parsed.isna(), "timestamp"] = np.nan
        except Exception:
            return pd.DataFrame()

    else:
        try:
            df = pd.read_csv(filepath)
            df["source_row_index"] = np.arange(len(df), dtype=np.int64)
            aliases = cfg.schema.get("vrmt", {}).get("aliases", {})
            df = standardize_columns(df, aliases)
        except Exception:
            return pd.DataFrame()

    time_col = cfg.schema.get("vrmt", {}).get("wall_time_column", "timestamp")

    # Ensure time exists
    if time_col not in df.columns:
        raise ValueError(f"Missing time column '{time_col}' in {filepath}")

    # Preserve the untouched source timestamp before conversion/sorting.
    if "source_timestamp_raw" not in df.columns:
        df["source_timestamp_raw"] = df[time_col].copy()

    # Convert time units explicitly.  Datetime strings are parsed only as a
    # representation of the configured wall-clock unit; no magnitude-based
    # unit inference occurs.
    time_factor = _configured_time_factor(cfg, "vrmt_time_to_s")
    df["wall_time_s"] = _coerce_wall_time_seconds(df[time_col], time_factor)

    # Relative time
    valid_time = df["wall_time_s"].dropna()
    if valid_time.empty:
        df["relative_time_s"] = np.nan
        df.attrs["timestamp_anchor_source_row_index"] = None
        df.attrs["invalid_timestamp_count"] = int(df["wall_time_s"].isna().sum())
        df.attrs["timestamp_status"] = "no_valid_timestamps"
    else:
        anchor_index = valid_time.index[0]
        df["relative_time_s"] = df["wall_time_s"] - df.loc[anchor_index, "wall_time_s"]
        df.attrs["timestamp_anchor_source_row_index"] = int(df.loc[anchor_index, "source_row_index"])
        invalid_timestamp_count = int(df["wall_time_s"].isna().sum())
        df.attrs["invalid_timestamp_count"] = invalid_timestamp_count
        df.attrs["timestamp_status"] = "invalid_timestamp_values" if invalid_timestamp_count else "ok"

    # Apply position scaling if needed
    pos_factor = cfg.units.get("vrmt_pos_to_m", 1.0)
    x_col = cfg.schema.get("vrmt", {}).get("x_column", "x")
    y_col = cfg.schema.get("vrmt", {}).get("y_column", "y")
    z_col = cfg.schema.get("vrmt", {}).get("z_column", "z")

    for c in [x_col, y_col, z_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c] * pos_factor

    # Preserve duplicate source rows for support validation.
    df = _attach_source_identity(
        df,
        seconds_col="wall_time_s",
        exact_scale=time_factor,
        raw_time_col="source_timestamp_raw",
    )
    if "_source_timestamp_ns_exact" in df.columns:
        df = df.drop(columns=["_source_timestamp_ns_exact"])
    df = df.sort_values(by="relative_time_s", kind="mergesort").reset_index(drop=True)

    df["trial_id"] = trial_id

    return df

def process_file_pair(cfg: Config, pair: Dict[str, Any]):
    trial_id = pair["trial_id"]
    ev3_file = pair["ev3_file"]
    vrmt_file = pair["vrmt_file"]

    out_dir_ev3 = os.path.join(cfg.paths.get("output_cleaned", "outputs/cleaned"), "ev3")
    out_dir_vrmt = os.path.join(cfg.paths.get("output_cleaned", "outputs/cleaned"), "vrmt")
    schema_dir = cfg.paths.get("output_schema", "outputs/schema_reports")

    os.makedirs(out_dir_ev3, exist_ok=True)
    os.makedirs(out_dir_vrmt, exist_ok=True)
    os.makedirs(schema_dir, exist_ok=True)

    if ev3_file:
        df_ev3 = load_and_standardize_ev3(cfg, ev3_file, trial_id)
        df_ev3.to_parquet(os.path.join(out_dir_ev3, f"{trial_id}_ev3_cleaned.parquet"), index=False)

        rep = generate_schema_report(df_ev3, trial_id, "ev3")
        rep["duplicate_source_key_count"] = int(df_ev3.attrs.get("duplicate_source_key_count", 0))
        rep["duplicate_source_key_rows"] = int(df_ev3.attrs.get("duplicate_source_key_rows", 0))
        rep["invalid_timestamp_count"] = int(df_ev3.attrs.get("invalid_timestamp_count", 0))
        rep["timestamp_status"] = df_ev3.attrs.get("timestamp_status", "ok")
        pair["ev3_duplicate_source_key_count"] = rep["duplicate_source_key_count"]
        pair["ev3_duplicate_source_key_rows"] = rep["duplicate_source_key_rows"]
        pair["ev3_timestamp_status"] = rep["timestamp_status"]
        with open(os.path.join(schema_dir, f"{trial_id}_ev3_schema.json"), 'w') as f:
            json.dump(rep, f, indent=2)

    if vrmt_file:
        df_vrmt = load_and_standardize_vrmt(cfg, vrmt_file, trial_id)
        df_vrmt.to_parquet(os.path.join(out_dir_vrmt, f"{trial_id}_vrmt_cleaned.parquet"), index=False)

        rep = generate_schema_report(df_vrmt, trial_id, "vrmt")
        rep["duplicate_source_key_count"] = int(df_vrmt.attrs.get("duplicate_source_key_count", 0))
        rep["duplicate_source_key_rows"] = int(df_vrmt.attrs.get("duplicate_source_key_rows", 0))
        rep["invalid_timestamp_count"] = int(df_vrmt.attrs.get("invalid_timestamp_count", 0))
        rep["timestamp_status"] = df_vrmt.attrs.get("timestamp_status", "ok")
        pair["vrmt_duplicate_source_key_count"] = rep["duplicate_source_key_count"]
        pair["vrmt_duplicate_source_key_rows"] = rep["duplicate_source_key_rows"]
        pair["vrmt_timestamp_status"] = rep["timestamp_status"]
        with open(os.path.join(schema_dir, f"{trial_id}_vrmt_schema.json"), 'w') as f:
            json.dump(rep, f, indent=2)

    # Expose loader diagnostics on the in-memory pair record for callers that
    # persist a cleaning ledger after processing.  No source rows are removed.
    pair["duplicate_source_key_count"] = int(
        pair.get("ev3_duplicate_source_key_count", 0)
        + pair.get("vrmt_duplicate_source_key_count", 0)
    )
    pair["duplicate_source_key_rows"] = int(
        pair.get("ev3_duplicate_source_key_rows", 0)
        + pair.get("vrmt_duplicate_source_key_rows", 0)
    )
    timestamp_reasons = [
        value
        for value in (pair.get("ev3_timestamp_status"), pair.get("vrmt_timestamp_status"))
        if value and value != "ok"
    ]
    if timestamp_reasons:
        existing = pair.get("status_reasons", [])
        if isinstance(existing, str):
            text = existing.strip()
            try:
                parsed = ast.literal_eval(text)
                existing = parsed if isinstance(parsed, (list, tuple)) else ([parsed] if parsed else [])
            except (SyntaxError, ValueError):
                existing = [] if text in ("", "[]") else [text]
        else:
            existing = list(existing or [])
        pair["status_reasons"] = existing + timestamp_reasons

    return [
        os.path.join(out_dir_ev3, f"{trial_id}_ev3_cleaned.parquet") if ev3_file else None,
        os.path.join(out_dir_vrmt, f"{trial_id}_vrmt_cleaned.parquet") if vrmt_file else None
    ]
