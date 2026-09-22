"""Canonical EV3-derived calibration/evaluation partition.

The partition boundary is computed once from the EV3 elapsed-time support and
then applied unchanged to both streams.  This module is deliberately small so
all downstream stages can consume the same persisted membership rather than
reimplementing local split logic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _section(cfg: Any, name: str) -> Mapping[str, Any]:
    if hasattr(cfg, name):
        value = getattr(cfg, name)
    elif isinstance(cfg, Mapping):
        value = cfg.get(name, {})
    else:
        value = {}
    return value if isinstance(value, Mapping) else {}


def _configured_float(cfg: Any, section: str, key: str, default: float) -> float:
    value = _section(cfg, section).get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{section}.{key} must be numeric") from exc
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{section}.{key} must be finite and non-negative")
    return value


def _time_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _normalise_stream(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.DataFrame:
    """Copy a stream and attach elapsed seconds anchored to its first finite row."""
    out = df.copy()
    if "source_row_index" not in out.columns:
        # This fallback is only for direct API callers that did not use a
        # loader.  Existing source identities are never replaced.
        out["source_row_index"] = np.arange(len(out), dtype=np.int64)
    time_col = _time_column(out, candidates)
    if time_col is None:
        out["partition_elapsed_s"] = np.nan
        return out
    numeric = pd.to_numeric(out[time_col], errors="coerce")
    finite = numeric[np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))]
    anchor = float(finite.iloc[0]) if not finite.empty else np.nan
    out["partition_elapsed_s"] = numeric - anchor if np.isfinite(anchor) else np.nan
    return out


def _trial_id(ev3: pd.DataFrame, vive: pd.DataFrame) -> str:
    for frame in (ev3, vive):
        if "trial_id" in frame.columns:
            values = frame["trial_id"].dropna()
            if not values.empty:
                return str(values.iloc[0])
        value = frame.attrs.get("trial_id")
        if value is not None:
            return str(value)
    return "unknown"


def _identity_values(frame: pd.DataFrame, column: str) -> list[Any]:
    if column not in frame.columns:
        return []
    values = frame[column].tolist()
    # JSON has no representation for numpy scalars/NA.  Convert them to
    # ordinary values while retaining source identities exactly when present.
    result: list[Any] = []
    for value in values:
        if pd.isna(value):
            result.append(None)
        elif isinstance(value, np.generic):
            result.append(value.item())
        else:
            result.append(value)
    return result


def _identity_hash(values: list[Any]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class PartitionResult:
    """Stable consumer object for one paired trial partition."""

    t_cal_s: float
    t_total_s: float
    ev3_calibration: pd.DataFrame
    ev3_evaluation: pd.DataFrame
    vive_calibration: pd.DataFrame
    vive_evaluation: pd.DataFrame
    status: str
    trial_id: str = "unknown"
    partition_hash: str = ""
    evaluation_duration_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def eligible_evaluation_fitting_keys(self) -> dict[str, list[Any]]:
        return dict(self.metadata.get("eligible_evaluation_fitting_keys", {}))

    @property
    def eligible_evaluation_fitting_key_hashes(self) -> dict[str, str]:
        return {
            stream: _eligible_key_hash(list(keys))
            for stream, keys in self.eligible_evaluation_fitting_keys.items()
        }


def _canonical_source_keys(frame: pd.DataFrame) -> list[Any]:
    """Return deterministic, finite source identities for an evaluation frame.

    CSV round-trips commonly turn integer identities into numeric strings.  We
    canonicalise those values here so persisted eligibility and downstream
    membership checks use the same representation.
    """
    values = _identity_values(frame, "source_row_index")
    keys: list[Any] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            try:
                number = float(text)
                if np.isfinite(number) and number.is_integer() and abs(number) <= 2**53:
                    value = int(number)
                elif not text:
                    continue
                else:
                    value = text
            except (TypeError, ValueError):
                value = text
        elif isinstance(value, (float, np.floating)):
            if not np.isfinite(value) or not float(value).is_integer():
                continue
            value = int(value)
        elif isinstance(value, (np.integer, int)):
            value = int(value)
        if value not in keys:
            keys.append(value)
    return sorted(keys, key=lambda item: (0, item) if isinstance(item, int) else (1, str(item)))


def _eligible_key_hash(keys: list[Any]) -> str:
    return _identity_hash(keys)


def _partition_payload(result: PartitionResult) -> dict[str, Any]:
    streams = {
        "ev3": {
            "calibration": {
                "source_row_index": _identity_values(result.ev3_calibration, "source_row_index"),
                "source_timestamp_ns": _identity_values(result.ev3_calibration, "source_timestamp_ns"),
            },
            "evaluation": {
                "source_row_index": _identity_values(result.ev3_evaluation, "source_row_index"),
                "source_timestamp_ns": _identity_values(result.ev3_evaluation, "source_timestamp_ns"),
            },
        },
        "vive": {
            "calibration": {
                "source_row_index": _identity_values(result.vive_calibration, "source_row_index"),
                "source_timestamp_ns": _identity_values(result.vive_calibration, "source_timestamp_ns"),
            },
            "evaluation": {
                "source_row_index": _identity_values(result.vive_evaluation, "source_row_index"),
                "source_timestamp_ns": _identity_values(result.vive_evaluation, "source_timestamp_ns"),
            },
        },
    }
    return {
        "trial_id": result.trial_id,
        "t_cal_s": result.t_cal_s,
        "t_total_s": result.t_total_s,
        "status": result.status,
        "streams": streams,
    }


def _records(result: PartitionResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eval_end = result.t_total_s
    for stream, calibration, evaluation in (
        ("ev3", result.ev3_calibration, result.ev3_evaluation),
        ("vive", result.vive_calibration, result.vive_evaluation),
    ):
        # The EV3 boundary is shared, but each stream's persisted end and
        # durations are capped at the EV3-derived total support.  This avoids
        # accidentally treating Vive observations after the paired EV3 trial
        # as evaluation observations.
        stream_elapsed = pd.concat(
            [calibration.get("partition_elapsed_s", pd.Series(dtype=float)),
             evaluation.get("partition_elapsed_s", pd.Series(dtype=float))],
            ignore_index=True,
        )
        finite_stream = pd.to_numeric(stream_elapsed, errors="coerce")
        finite_stream = finite_stream[np.isfinite(finite_stream.to_numpy(dtype=float, na_value=np.nan))]
        stream_end = min(result.t_total_s, float(finite_stream.max())) if not finite_stream.empty else 0.0
        stream_cal_duration = min(result.t_cal_s, stream_end)
        stream_eval_duration = max(0.0, stream_end - result.t_cal_s)
        for partition, frame, start, end in (
            ("calibration", calibration, 0.0, stream_cal_duration),
            ("evaluation", evaluation, result.t_cal_s, stream_end),
        ):
            row_indices = _identity_values(frame, "source_row_index")
            eligible_keys = _canonical_source_keys(evaluation)
            timestamp_keys = _identity_values(frame, "source_timestamp_ns")
            rows.append(
                {
                    "trial_id": result.trial_id,
                    "stream": stream,
                    "partition": partition,
                    "calibration_start_s": 0.0,
                    # Shared EV3-derived boundaries remain identical across
                    # streams.  Stream-specific availability is reported in
                    # the explicit ``stream_*`` fields below.
                    "calibration_end_s": result.t_cal_s,
                    "evaluation_start_s": result.t_cal_s,
                    "evaluation_end_s": eval_end,
                    # Unsuffixed aliases match the persisted contract wording
                    # while the ``*_s`` names remain explicit about units.
                    "calibration_start": 0.0,
                    "calibration_end": result.t_cal_s,
                    "evaluation_start": result.t_cal_s,
                    "evaluation_end": eval_end,
                    "n_calibration_rows": len(calibration),
                    "n_evaluation_rows": len(evaluation),
                    "n_rows_excluded_outside_ev3_support": int(
                        result.metadata.get("vive_rows_excluded_outside_ev3_support", 0)
                        if stream == "vive" else 0
                    ),
                    "source_row_indices_excluded_outside_ev3_support": json.dumps(
                        result.metadata.get("vive_source_rows_excluded_outside_ev3_support", []),
                        separators=(",", ":"),
                    ) if stream == "vive" else "[]",
                    "calibration_duration_s": stream_cal_duration,
                    "evaluation_duration_s": stream_eval_duration,
                    "stream_available_end_s": stream_end,
                    "stream_calibration_duration_s": stream_cal_duration,
                    "stream_evaluation_end_s": stream_end,
                    "stream_evaluation_duration_s": stream_eval_duration,
                    "stream_status": "insufficient_stream_support" if stream_end < result.t_cal_s else "ok",
                    "source_row_indices": json.dumps(row_indices, separators=(",", ":")),
                    "source_row_index": json.dumps(row_indices, separators=(",", ":")),
                    "source_timestamp_ns": json.dumps(timestamp_keys, separators=(",", ":")),
                    "source_row_index_hash": _identity_hash(row_indices),
                    "source_timestamp_ns_hash": _identity_hash(timestamp_keys),
                    # Full-fit H3 is allowed to use evaluation observations,
                    # but only the persisted eligible identities are counted
                    # as leakage evidence.  Persist this per stream so a
                    # downstream consumer cannot substitute EV3 IDs for Vive.
                    "eligible_evaluation_fitting_keys": json.dumps(eligible_keys, separators=(",", ":")),
                    "eligible_evaluation_fitting_key_hash": _eligible_key_hash(eligible_keys),
                    "eligible_evaluation_fitting_keys_hash": _eligible_key_hash(eligible_keys),
                    "partition_hash": result.partition_hash,
                    "status": result.status,
                }
            )
    return rows


def partition_records(result: PartitionResult) -> list[dict[str, Any]]:
    """Return serialisable rows used by stage 04 and :func:`persist_partition`."""
    return _records(result)


def compute_partition(ev3: pd.DataFrame, vive: pd.DataFrame, cfg: Any) -> PartitionResult:
    """Compute one EV3-derived boundary and apply it to both streams.

    The nominal boundary is always computed first as
    ``min(configured_duration_s, T_total / 2)``.  If the resulting evaluation
    interval is shorter than the configured minimum, the trial is rejected;
    calibration is never shortened to satisfy that minimum.
    """
    ev3_n = _normalise_stream(ev3, ("local_time_s", "elapsed_time_s", "elapsed_s", "time_s", "time"))
    vive_n = _normalise_stream(vive, ("relative_time_s", "elapsed_time_s", "elapsed_s", "time_s", "time"))
    ev3_elapsed = pd.to_numeric(ev3_n["partition_elapsed_s"], errors="coerce")
    finite_ev3 = ev3_elapsed[np.isfinite(ev3_elapsed.to_numpy(dtype=float, na_value=np.nan))]
    t_total = float(finite_ev3.max()) if not finite_ev3.empty else 0.0
    configured = _configured_float(cfg, "calibration", "duration_s", 20.0)
    minimum_eval = _configured_float(
        cfg,
        "calibration",
        "minimum_evaluation_duration_s",
        0.0,
    )
    t_cal = min(configured, t_total / 2.0)
    eval_duration = max(0.0, t_total - t_cal)

    ev3_cal = ev3_n[ev3_n["partition_elapsed_s"].notna() & (ev3_n["partition_elapsed_s"] <= t_cal)].copy()
    ev3_eval = ev3_n[ev3_n["partition_elapsed_s"].notna() & (ev3_n["partition_elapsed_s"] > t_cal)].copy()
    # EV3-derived elapsed time is intentionally used for Vive membership too;
    # there is no independent Vive split boundary.
    vive_finite = vive_n["partition_elapsed_s"].notna()
    vive_outside_mask = vive_finite & (vive_n["partition_elapsed_s"] > t_total)
    vive_outside_count = int(vive_outside_mask.sum())
    vive_outside_source_rows = _identity_values(
        vive_n.loc[vive_outside_mask], "source_row_index"
    )
    vive_in_ev3_support = vive_n["partition_elapsed_s"] <= t_total
    vive_cal = vive_n[vive_finite & vive_in_ev3_support & (vive_n["partition_elapsed_s"] <= t_cal)].copy()
    vive_eval = vive_n[vive_finite & vive_in_ev3_support & (vive_n["partition_elapsed_s"] > t_cal)].copy()

    status = "success"
    if finite_ev3.empty:
        status = "missing_ev3_time"
    elif eval_duration < minimum_eval:
        status = "insufficient_evaluation_duration"

    result = PartitionResult(
        t_cal_s=float(t_cal),
        t_total_s=float(t_total),
        ev3_calibration=ev3_cal,
        ev3_evaluation=ev3_eval,
        vive_calibration=vive_cal,
        vive_evaluation=vive_eval,
        status=status,
        trial_id=_trial_id(ev3_n, vive_n),
        evaluation_duration_s=float(eval_duration),
        metadata={
            "vive_rows_excluded_outside_ev3_support": vive_outside_count,
            "vive_source_rows_excluded_outside_ev3_support": vive_outside_source_rows,
            "eligible_evaluation_fitting_keys": {
                "ev3": _canonical_source_keys(ev3_eval),
                "vive": _canonical_source_keys(vive_eval),
            },
        },
    )
    payload = json.dumps(_partition_payload(result), sort_keys=True, separators=(",", ":"), default=str)
    result.partition_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return result


def persist_partition(result: PartitionResult, output_path: str | Path) -> None:
    """Persist all stream/partition membership rows with one deterministic hash."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_records(result)).to_csv(path, index=False)
