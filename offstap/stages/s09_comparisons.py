"""
scripts/09_compare_configurations.py

Executable script to compare multiple runs if available.
"""
import sys
import os
import pandas as pd
import numpy as np
import json
import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

# Add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offstap.core.config import load_config, Config
from offstap.core.statistics import compare_configurations
from offstap.core.provenance import ProvenanceTracker
import offstap.core.temporal as _temporal_module
import offstap.core.spatial as _spatial_module
from offstap.core.trajectory_reference import query_reference
from offstap.core.ev3_signals import derive_ev3_signals
from offstap.core.vrmt_signals import derive_vrmt_signals


def estimate_temporal_alignment(*args, **kwargs):
    return _temporal_module.estimate_temporal_alignment(*args, **kwargs)


def estimate_spatial_transform(*args, **kwargs):
    return _spatial_module.estimate_spatial_transform(*args, **kwargs)


def _prepare_h3_fit_frames(ev3: pd.DataFrame, vive: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Provide H3's configured temporal proxies without changing source identity.

    Stage 04 persists derived proxies in calibration-only parquet files, while
    H3 deliberately reads the full cleaned streams to construct split and full
    fits.  Re-derive only the deterministic proxy columns at this boundary so
    the full-fit arm has the same signal definition as the split-fit arm.
    """
    ev3_out = ev3.copy()
    vive_out = vive.copy()
    if not ev3_out.empty and "speed_magnitude" not in ev3_out.columns:
        ev3_out = derive_ev3_signals(ev3_out, speed_col="motor_speed", heading_col="gyro_heading")
    if not vive_out.empty and "speed_magnitude" not in vive_out.columns:
        vive_out = derive_vrmt_signals(vive_out)
    return ev3_out, vive_out


class _ClockFit(dict):
    """Dictionary-compatible affine clock mapping with attribute access."""

    @property
    def scale_a(self) -> float:
        return float(self.get("scale_a", 1.0))

    @property
    def offset_b(self) -> float:
        return float(self.get("offset_b", 0.0))


@dataclass
class Arm:
    """One isolated H1 arm and its held-out support."""

    name: str
    temporal: dict[str, Any]
    clock: _ClockFit
    spatial: Any
    evaluation: pd.DataFrame = field(default_factory=pd.DataFrame)
    support: set[Any] = field(default_factory=set)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # H3 provenance fields.  Defaults preserve the H1 constructor contract.
    fit_source_keys: set[Any] = field(default_factory=set)
    partition_hash: str = ""
    mode: str = ""

    @property
    def fit_key_count(self) -> int:
        return len(self.fit_source_keys)

    @property
    def leakage_status(self) -> str:
        if self.mode == "full":
            return "intentionally_leaky_evaluation_fitting_keys_included"
        return "split_excludes_evaluation_fitting_keys"


@dataclass
class H3Comparison:
    """Paired H3 score on one exact held-out support."""

    split: Arm
    full: Arm
    common_support: set[Any] = field(default_factory=set)
    split_rmse: float = np.nan
    full_rmse: float = np.nan
    delta_rmse: float = np.nan
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def rmse_split(self) -> float:
        return self.split_rmse

    @property
    def rmse_full(self) -> float:
        return self.full_rmse

    @property
    def split_arm(self) -> Arm:
        return self.split

    @property
    def full_arm(self) -> Arm:
        return self.full

    @property
    def rmse_split_mm(self) -> float:
        return self.split_rmse * 1000.0

    @property
    def rmse_full_mm(self) -> float:
        return self.full_rmse * 1000.0

    @property
    def delta_rmse_mm(self) -> float:
        return self.delta_rmse * 1000.0

    @property
    def common_support_hash(self) -> str:
        return str(self.diagnostics.get("common_support_hash", ""))

    @property
    def status(self) -> str:
        return str(self.diagnostics.get("status", "invalid"))

    @property
    def reason(self) -> Any:
        return self.diagnostics.get("reason")


@dataclass
class H1Comparison:
    onset: Arm
    refined: Arm
    common_support: set[Any]
    onset_rmse: float = np.nan
    refined_rmse: float = np.nan
    delta_rmse: float = np.nan
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # Explicit names used in reports and by external callers.
    @property
    def rmse_onset(self) -> float:
        return self.onset_rmse

    @property
    def rmse_refined(self) -> float:
        return self.refined_rmse

    @property
    def rmse_onset_mm(self) -> float:
        return self.onset_rmse * 1000.0

    @property
    def rmse_refined_mm(self) -> float:
        return self.refined_rmse * 1000.0

    @property
    def delta_rmse_mm(self) -> float:
        return self.delta_rmse * 1000.0

    @property
    def support_hash(self) -> str:
        return str(self.diagnostics.get("common_support_hash", ""))

    @property
    def common_support_hash(self) -> str:
        return self.support_hash

    @property
    def reason(self) -> Any:
        return self.diagnostics.get("reason")


def _partition_frames(partition: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract EV3/Vive calibration and evaluation frames from common forms."""
    if isinstance(partition, (tuple, list)) and len(partition) == 4:
        frames = tuple(item.copy() if isinstance(item, pd.DataFrame) else pd.DataFrame() for item in partition)
        return frames  # type: ignore[return-value]
    def pick(names: tuple[str, ...]) -> pd.DataFrame:
        for name in names:
            if isinstance(partition, Mapping) and name in partition:
                value = partition[name]
            else:
                value = getattr(partition, name, None)
            if isinstance(value, pd.DataFrame):
                return value.copy()
        return pd.DataFrame()

    ev3_cal = pick(("ev3_calibration", "ev3_calib", "calibration_ev3"))
    vive_cal = pick(("vive_calibration", "vrmt_calibration", "vrmt_calib", "vive_calib"))
    ev3_eval = pick(("ev3_evaluation", "ev3_eval", "evaluation_ev3"))
    vive_eval = pick(("vive_evaluation", "vrmt_evaluation", "vrmt_eval", "vive_eval"))
    # Lightweight fixtures often provide nested stream mappings.
    if isinstance(partition, Mapping):
        # Alternate shape: {"calibration": {"ev3": frame, "vive": frame},
        # "evaluation": {...}}.
        cal_map, eval_map = partition.get("calibration"), partition.get("evaluation")
        if isinstance(cal_map, Mapping):
            ev3_value = cal_map.get("ev3", cal_map.get("ev3_calibration"))
            vive_value = cal_map.get("vive", cal_map.get("vrmt"))
            if isinstance(ev3_value, pd.DataFrame):
                ev3_cal = ev3_value.copy()
            if isinstance(vive_value, pd.DataFrame):
                vive_cal = vive_value.copy()
        if isinstance(eval_map, Mapping):
            ev3_value = eval_map.get("ev3", eval_map.get("ev3_evaluation"))
            vive_value = eval_map.get("vive", eval_map.get("vrmt"))
            if isinstance(ev3_value, pd.DataFrame):
                ev3_eval = ev3_value.copy()
            if isinstance(vive_value, pd.DataFrame):
                vive_eval = vive_value.copy()
        for stream, target in (("ev3", "ev3"), ("vive", "vive"), ("vrmt", "vive")):
            item = partition.get(stream)
            if isinstance(item, Mapping):
                cal = item.get("calibration", item.get("calib"))
                ev = item.get("evaluation", item.get("eval"))
                if target == "ev3":
                    ev3_cal = cal.copy() if isinstance(cal, pd.DataFrame) else ev3_cal
                    ev3_eval = ev.copy() if isinstance(ev, pd.DataFrame) else ev3_eval
                else:
                    vive_cal = cal.copy() if isinstance(cal, pd.DataFrame) else vive_cal
                    vive_eval = ev.copy() if isinstance(ev, pd.DataFrame) else vive_eval
    return ev3_cal, vive_cal, ev3_eval, vive_eval


def _source_keys(frame: pd.DataFrame) -> set[Any]:
    if frame.empty:
        return set()
    # ``support_key`` is the canonical post-alignment identity.  Fall back to
    # the persisted source row only before alignment.
    column = "support_key" if "support_key" in frame.columns else "source_row_index"
    if column not in frame.columns:
        # Exact support comparisons require persisted source identities.  A
        # positional range would fabricate identities and could pair unrelated
        # observations across arms.
        return set()
    from offstap.core.support import validate_arm_support
    validation = validate_arm_support(frame, [column])
    if not validation.valid:
        return set()
    return {row[0] for row in validation.canonical_keys.itertuples(index=False, name=None)}


def _attach_arm_provenance(spatial: Any, ev3_cal: pd.DataFrame, arm_name: str, temporal: Mapping[str, Any], clock: Mapping[str, Any]) -> Any:
    """Ensure each arm records an auditable, distinct calibration fit."""
    from offstap.core.spatial import SpatialFit
    if isinstance(spatial, SpatialFit):
        fit = spatial
    elif isinstance(spatial, Mapping):
        fit = SpatialFit(dict(spatial))
    elif spatial is not None and hasattr(spatial, "__dict__"):
        fit = SpatialFit(dict(vars(spatial)))
    else:
        fit = SpatialFit()
    keys = sorted(map(str, _source_keys(ev3_cal)))
    payload = json.dumps({"arm": arm_name, "keys": keys, "a": float(clock.get("scale_a", 1.0)),
                          "b": float(clock.get("offset_b", 0.0)),
                          "method": str(temporal.get("method_used", ""))}, sort_keys=True)
    key_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    fit_id = hashlib.sha256(("spatial-fit:" + payload).encode("utf-8")).hexdigest()
    fit["provenance"] = {"fit_source": "calibration_source", "spatial_fit_id": fit_id,
                          "fit_source_key_hash": key_hash}
    return fit


def build_h1_arms(partition: Any, cfg: Any) -> dict[str, Arm]:
    """Fit onset and refined H1 arms independently from calibration rows only."""
    ev3_cal, vive_cal, ev3_eval, vive_eval = _partition_frames(partition)
    # Frames supplied by lightweight callers may still contain both labels;
    # enforce the persisted partition contract at the API boundary.
    if "partition" in ev3_cal.columns:
        ev3_cal = ev3_cal[ev3_cal["partition"].astype(str).str.lower().eq("calibration")].copy()
    if "partition" in vive_cal.columns:
        vive_cal = vive_cal[vive_cal["partition"].astype(str).str.lower().eq("calibration")].copy()
    if "partition" in ev3_eval.columns:
        ev3_eval = ev3_eval[ev3_eval["partition"].astype(str).str.lower().eq("evaluation")].copy()
    if "partition" in vive_eval.columns:
        vive_eval = vive_eval[vive_eval["partition"].astype(str).str.lower().eq("evaluation")].copy()
    # Empty or rejected calibration partitions are represented as invalid
    # arms (rather than silently falling back to full-stream fitting).
    arms: dict[str, Arm] = {}
    # Invoke temporal fitting independently for each arm; no evaluation frame
    # is ever passed to this routine.
    for name in ("onset", "refined"):
        temporal_raw = estimate_temporal_alignment(ev3_cal.copy(), vive_cal.copy(), cfg)
        if isinstance(temporal_raw, Mapping):
            temporal = dict(temporal_raw)
        elif temporal_raw is not None and hasattr(temporal_raw, "__dict__"):
            temporal = dict(vars(temporal_raw))
        else:
            temporal = {}
        b_onset = float(temporal.get("b_onset", 0.0))
        b_refined = float(temporal.get("b_final", temporal.get("b_refined", b_onset)))
        b = b_onset if name == "onset" else b_refined
        clock = _ClockFit({"selected_model": "constant_offset", "offset_b": b, "scale_a": 1.0})
        spatial_cfg = cfg
        if not hasattr(cfg, "geometry") and isinstance(cfg, Mapping):
            try:
                spatial_cfg = Config(dict(cfg))
            except Exception:
                spatial_cfg = cfg
        try:
            spatial = estimate_spatial_transform(ev3_cal.copy(), vive_cal.copy(), spatial_cfg, temporal, clock)
        except TypeError:
            # Keep the public helper easy to monkeypatch in lightweight
            # contract tests while production fitting receives the mapping.
            spatial = estimate_spatial_transform(ev3_cal.copy(), vive_cal.copy(), spatial_cfg)
        spatial = _attach_arm_provenance(spatial, ev3_cal, name, temporal, clock)
        # Evaluation support is sourced exclusively from persisted EV3 rows.
        evaluation = ev3_eval.copy()
        support = _source_keys(evaluation)
        diagnostics = {
            "b_onset_s": b_onset, "b_refined_s": b_refined,
            "delta_b_s": b_refined - b_onset,
            "abs_delta_b_s": abs(b_refined - b_onset),
            "refinement_score": temporal.get("best_corr", np.nan),
            "second_peak_score": temporal.get("second_best_corr", np.nan),
            "peak_ratio": temporal.get("peak_confidence_ratio", np.nan),
            "overlap_s": temporal.get("overlap_duration_s", 0.0),
            "rejection_reason": temporal.get("rejection_reason", temporal.get("reason")),
            # Onset-only remains a valid baseline even when the refinement
            # correlation is rejected; the refined arm carries that rejection.
            "status": "ok" if name == "onset" else temporal.get("status", "ok"),
            "fit_source": "calibration_source",
        }
        diagnostics["rejection_reason"] = diagnostics.get("rejection_reason")
        arms[name] = Arm(name, temporal, clock, spatial, evaluation, support, diagnostics)
    return arms


def _rmse_on_support(arm: Arm, support: set[Any]) -> float:
    frame = arm.evaluation
    if frame.empty:
        return float("nan")
    key_col = "support_key" if "support_key" in frame.columns else "source_row_index"
    selected = _h3_filter_keys(frame, support) if key_col in frame.columns else frame
    if selected.empty:
        return float("nan")
    if "residual_norm" in selected.columns:
        values = pd.to_numeric(selected["residual_norm"], errors="coerce").to_numpy(float)
    elif {"odometry_x", "odometry_y", "vrmt_aligned_x", "vrmt_aligned_y"}.issubset(selected.columns):
        dx = pd.to_numeric(selected["odometry_x"], errors="coerce") - pd.to_numeric(selected["vrmt_aligned_x"], errors="coerce")
        dy = pd.to_numeric(selected["odometry_y"], errors="coerce") - pd.to_numeric(selected["vrmt_aligned_y"], errors="coerce")
        values = np.sqrt(dx.to_numpy(float) ** 2 + dy.to_numpy(float) ** 2)
    else:
        return float("nan")
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values ** 2))) if values.size else float("nan")


def _finite_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain rows with finite residual inputs before support intersection."""
    if frame.empty:
        return frame.copy()
    if "residual_norm" in frame.columns:
        values = pd.to_numeric(frame["residual_norm"], errors="coerce").to_numpy(float)
        return frame[np.isfinite(values)].copy()
    required = ["odometry_x", "odometry_y", "vrmt_aligned_x", "vrmt_aligned_y"]
    if set(required).issubset(frame.columns):
        numeric = frame[required].apply(pd.to_numeric, errors="coerce")
        return frame[np.isfinite(numeric.to_numpy(float)).all(axis=1)].copy()
    # Support identity is still auditable when a lightweight caller has not
    # materialized residual columns; RMSE scoring will fail closed separately.
    return frame.copy()


def _h3_get(value: Any, *names: str, default: Any = None) -> Any:
    """Read a field from mappings or lightweight fixture objects."""
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return default


def _h3_keys(value: Any) -> set[Any]:
    """Normalise persisted source keys without fabricating positional IDs."""
    if value is None:
        return set()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            value = [value]
    if isinstance(value, pd.Series):
        value = value.tolist()
    if isinstance(value, pd.DataFrame):
        value = value.get("source_row_index", pd.Series(dtype=object)).tolist()
    try:
        items = list(value)
    except TypeError:
        items = [value]
    keys: set[Any] = set()
    for item in items:
        if pd.isna(item):
            continue
        if isinstance(item, np.generic):
            item = item.item()
        # Source identities are integer keys in the persisted contract.  Do
        # not coerce arbitrary strings/fractions into apparently valid IDs.
        if isinstance(item, (int, np.integer)):
            keys.add(int(item))
        elif isinstance(item, float) and np.isfinite(item) and item.is_integer():
            keys.add(int(item))
        elif isinstance(item, str):
            text = item.strip()
            try:
                number = float(text)
                if np.isfinite(number) and number.is_integer() and abs(number) <= 2**53:
                    keys.add(int(number))
                else:
                    keys.add(item)
            except (TypeError, ValueError):
                keys.add(item)
        else:
            keys.add(item)
    return keys


def _h3_normalise_hash(value: Any) -> str:
    """Return a persisted hash or ``""`` for missing/non-finite values.

    In particular, never turn a pandas/NumPy NaN into the literal string
    ``"nan"``: that would make an invalid partition appear to have a hash.
    """
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "nat"}:
        return ""
    return text


def _h3_key_hash(keys: set[Any] | list[Any]) -> str:
    """Deterministic hash matching ``src.partition`` eligibility hashing."""
    canonical = sorted(_h3_keys(keys), key=lambda item: (0, item) if isinstance(item, int) else (1, str(item)))
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _h3_frame_keys(frame: pd.DataFrame) -> set[Any]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return set()
    col = "support_key" if "support_key" in frame.columns else "source_row_index"
    return _h3_keys(frame[col]) if col in frame.columns else set()


def _h3_filter_keys(frame: pd.DataFrame, keys: set[Any]) -> pd.DataFrame:
    """Filter rows by canonical source identity (including numeric strings)."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    # Persisted partition IDs are source-row identities.  Prefer that column
    # when present; aligned evaluation frames may expose only ``support_key``.
    col = "source_row_index" if "source_row_index" in frame.columns else "support_key"
    if col not in frame.columns:
        return frame.iloc[0:0].copy()
    mask = frame[col].map(lambda value: bool(_h3_keys([value]) & keys))
    return frame.loc[mask].copy()


def _h3_stream_hashes(ledger: Any, stream: str, trial_id: Any = None) -> set[str]:
    """Collect non-empty partition hashes for one stream from a ledger."""
    if not isinstance(ledger, pd.DataFrame) or ledger.empty:
        return set()
    rows = ledger.copy()
    if trial_id is not None and "trial_id" in rows.columns:
        rows = rows[rows["trial_id"].astype(str).eq(str(trial_id))]
    if "stream" in rows.columns:
        aliases = {str(stream).lower(), "vrmt"} if stream == "vive" else {str(stream).lower()}
        rows = rows[rows["stream"].astype(str).str.lower().isin(aliases)]
    hashes: set[str] = set()
    if "partition_hash" in rows.columns:
        for value in rows["partition_hash"].tolist():
            normalized = _h3_normalise_hash(value)
            if normalized:
                hashes.add(normalized)
    return hashes


def _h3_ledger_keysets(ledger: Any, trial_id: Any = None) -> tuple[set[Any], set[Any]]:
    """Extract EV3 calibration/evaluation keys from persisted CSV ledger rows."""
    if not isinstance(ledger, pd.DataFrame) or ledger.empty:
        return set(), set()
    rows = ledger.copy()
    if trial_id is not None and "trial_id" in rows.columns:
        rows = rows[rows["trial_id"].astype(str).eq(str(trial_id))]
    if "stream" in rows.columns:
        rows = rows[rows["stream"].astype(str).str.lower().eq("ev3")]
    cal: set[Any] = set()
    ev: set[Any] = set()
    for _, row in rows.iterrows():
        keys = _h3_keys(row.get("source_row_index", row.get("source_row_indices", [])))
        label = str(row.get("partition", "")).lower()
        if label == "calibration":
            cal |= keys
        elif label in {"evaluation", "eval", "heldout", "held_out", "test"}:
            ev |= keys
    return cal, ev


def _h3_stream_keysets(ledger: Any, stream: str, trial_id: Any = None) -> tuple[set[Any], set[Any], set[Any], bool]:
    """Parse persisted calibration/evaluation/eligible keys for one stream.

    The partition ledger is authoritative.  ``eligible`` is intentionally
    tracked with a presence flag so an omitted eligible-fitting field cannot be
    mistaken for a valid empty set (which would hide H3 leakage accounting).
    """
    if not isinstance(ledger, pd.DataFrame) or ledger.empty:
        return set(), set(), set(), False
    rows = ledger.copy()
    if trial_id is not None and "trial_id" in rows.columns:
        rows = rows[rows["trial_id"].astype(str).eq(str(trial_id))]
    if "stream" in rows.columns:
        rows = rows[rows["stream"].astype(str).str.lower().isin({str(stream).lower(), "vrmt" if stream == "vive" else str(stream).lower()})]
    cal: set[Any] = set()
    ev: set[Any] = set()
    eligible: set[Any] = set()
    eligible_present = False
    for _, row in rows.iterrows():
        keys = _h3_keys(row.get("source_row_index", row.get("source_row_indices", [])))
        label = str(row.get("partition", "")).lower()
        if label == "calibration":
            cal |= keys
        elif label in {"evaluation", "eval", "heldout", "held_out", "test"}:
            ev |= keys
        for field_name in (
            "eligible_evaluation_fitting_keys", "eligible_eval_fitting_keys",
            "eligible_evaluation_fitting_key", "evaluation_fitting_keys",
            "evaluation_fitting_key",
        ):
            if field_name in row.index and pd.notna(row[field_name]):
                eligible_present = True
                eligible |= _h3_keys(row[field_name])
    return cal, ev, eligible, eligible_present


def _h3_duration(trial: Any, partition: Any, cfg: Any = None) -> float:
    value = _h3_get(trial, "evaluation_duration_s", "evaluation_duration", default=None)
    if value is None:
        value = _h3_get(partition, "evaluation_duration_s", "evaluation_duration", default=None)
    if isinstance(value, pd.Series):
        value = value.iloc[0] if not value.empty else None
    if value is None and isinstance(partition, pd.DataFrame) and not partition.empty:
        rows = partition
        if "partition" in rows.columns:
            rows = rows[rows["partition"].astype(str).str.lower().isin({"evaluation", "eval", "heldout", "held_out", "test"})]
        if "evaluation_duration_s" in rows.columns and not rows.empty:
            value = rows.iloc[0]["evaluation_duration_s"]
    if value is None:
        start = _h3_get(partition, "evaluation_start_s", "evaluation_start", default=None)
        end = _h3_get(partition, "evaluation_end_s", "evaluation_end", default=None)
        if start is not None and end is not None:
            value = float(end) - float(start)
    if value is None:
        total = _h3_get(trial, "t_total_s", "total_duration_s", default=None)
        cal = _h3_get(trial, "t_cal_s", "calibration_duration_s", default=None)
        if total is not None and cal is not None:
            value = float(total) - float(cal)
    try:
        return max(0.0, float(value)) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _h3_min_eval(cfg: Any, trial: Any) -> float:
    value = _h3_get(trial, "minimum_evaluation_duration_s", default=None)
    if value is None:
        section = _h3_get(cfg, "calibration", default={})
        value = _h3_get(section, "minimum_evaluation_duration_s", default=0.0)
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _h3_partition(trial: Any) -> Any:
    return _h3_get(trial, "partition", "persisted_partition", "partition_result", default=trial)


def fit_h3_arm(trial: Any, mode: str = "split", cfg: Any = None) -> Arm:
    """Fit one H3 arm from the persisted dynamic partition.

    ``split`` uses calibration keys only.  ``full`` intentionally includes all
    persisted evaluation keys (including the eligible fitting subset), making
    leakage explicit in the returned provenance.  No local duration threshold
    or positional row identity is inferred.
    """
    mode = str(mode).lower()
    if mode not in {"split", "full"}:
        raise ValueError("mode must be 'split' or 'full'")
    if cfg is None:
        cfg = _h3_get(trial, "cfg", "config", default=None)
    partition = _h3_partition(trial)
    ev3_cal, vive_cal, ev3_eval, vive_eval = _partition_frames(partition)
    ev3_cal, vive_cal = _prepare_h3_fit_frames(ev3_cal, vive_cal)
    ev3_eval, vive_eval = _prepare_h3_fit_frames(ev3_eval, vive_eval)
    trial_id = _h3_get(trial, "trial_id", default=None)
    ledger = _h3_get(partition, "ledger", "partition_ledger", "rows", default=partition)
    # A CSV-loaded partition is itself the ledger.  Nested fixtures may expose
    # it under ``ledger`` while carrying stream frames separately.
    ev3_ledger_cal, ev3_ledger_eval, ev3_eligible, ev3_eligible_present = _h3_stream_keysets(ledger, "ev3", trial_id)
    vive_ledger_cal, vive_ledger_eval, _, _ = _h3_stream_keysets(ledger, "vive", trial_id)
    ledger_cal, ledger_eval = _h3_ledger_keysets(ledger, trial_id)
    # Explicit fixture fields take precedence over frames and are the persisted
    # support ledger used to prove membership.
    calibration_keys = _h3_keys(_h3_get(trial, "calibration_keys", "calibration_source_keys", "calibration_fitting_keys", "calibration_source_rows", default=None))
    if not calibration_keys:
        calibration_keys = _h3_keys(_h3_get(partition, "calibration_keys", "calibration_source_keys", "calibration_fitting_keys", "calibration_source_rows", default=None))
    if not calibration_keys:
        calibration_keys = ev3_ledger_cal or ledger_cal
    evaluation_keys = _h3_keys(_h3_get(trial, "evaluation_keys", "evaluation_source_keys", default=None))
    if not evaluation_keys:
        evaluation_keys = _h3_keys(_h3_get(partition, "evaluation_keys", "evaluation_source_keys", default=None))
    if not evaluation_keys:
        evaluation_keys = ev3_ledger_eval or ledger_eval
    vive_calibration_keys = _h3_keys(_h3_get(trial, "vive_calibration_keys", "vive_calibration_source_keys", default=None))
    if not vive_calibration_keys:
        vive_calibration_keys = _h3_keys(_h3_get(partition, "vive_calibration_keys", "vive_calibration_source_keys", default=None))
    if not vive_calibration_keys:
        # A missing Vive ledger must never inherit EV3 identities.  For
        # lightweight in-memory callers, derive only from the Vive frame.
        vive_calibration_keys = vive_ledger_cal or _h3_frame_keys(vive_cal)
    vive_evaluation_keys = _h3_keys(_h3_get(trial, "vive_evaluation_keys", "vive_evaluation_source_keys", default=None))
    if not vive_evaluation_keys:
        vive_evaluation_keys = _h3_keys(_h3_get(partition, "vive_evaluation_keys", "vive_evaluation_source_keys", default=None))
    if not vive_evaluation_keys:
        vive_evaluation_keys = vive_ledger_eval or _h3_frame_keys(vive_eval)
    eligible_eval = _h3_keys(_h3_get(trial, "eligible_evaluation_fitting_keys", "eligible_eval_fitting_keys", "eligible_evaluation_keys", "evaluation_fitting_keys", default=None))
    if not eligible_eval:
        eligible_eval = _h3_keys(_h3_get(partition, "eligible_evaluation_fitting_keys", "eligible_eval_fitting_keys", "eligible_evaluation_keys", "evaluation_fitting_keys", default=None))
    eligible_present = bool(eligible_eval) or ev3_eligible_present
    if not eligible_eval and ev3_eligible_present:
        eligible_eval = ev3_eligible
    partition_hash_value = _h3_get(trial, "partition_hash", default=None)
    if partition_hash_value is None and isinstance(partition, pd.DataFrame) and "partition_hash" in partition.columns and not partition.empty:
        partition_hash_value = partition.iloc[0]["partition_hash"]
    if partition_hash_value is None:
        partition_hash_value = _h3_get(partition, "partition_hash", default="")
    partition_hash = _h3_normalise_hash(partition_hash_value)
    ev3_hashes = _h3_stream_hashes(ledger, "ev3", trial_id)
    vive_hashes = _h3_stream_hashes(ledger, "vive", trial_id)
    # Accept explicit per-stream provenance fields used by object callers.
    for names, target in (
        (("ev3_partition_hash", "ev3_partition_hashes"), ev3_hashes),
        (("vive_partition_hash", "vive_partition_hashes"), vive_hashes),
    ):
        for owner in (trial, partition):
            value = _h3_get(owner, *names, default=None)
            if isinstance(value, (list, tuple, set)):
                target.update(filter(None, (_h3_normalise_hash(item) for item in value)))
            else:
                normalized = _h3_normalise_hash(value)
                if normalized:
                    target.add(normalized)
    # Nested partition objects may carry stream-specific hashes as attrs.
    for stream_name, stream_obj in (("ev3", ev3_cal), ("vive", vive_cal)):
        candidate = _h3_normalise_hash(_h3_get(stream_obj, "partition_hash", default=None))
        if candidate:
            (ev3_hashes if stream_name == "ev3" else vive_hashes).add(candidate)
    eval_duration = _h3_duration(trial, partition, cfg)
    min_eval = _h3_min_eval(cfg, trial)
    fit_keys = set(calibration_keys) if mode == "split" else (set(calibration_keys) | set(evaluation_keys))

    # All identity and partition gates are evaluated before any fitting.  This
    # prevents accidental full-stream or positional fallbacks from changing
    # the H3 estimand when persisted provenance is incomplete.
    failure = None
    if not partition_hash or not calibration_keys or not evaluation_keys or not eligible_present:
        failure = "missing_persisted_h3_keys_or_partition_hash"
    elif set(calibration_keys) & set(evaluation_keys):
        failure = "persisted_calibration_evaluation_overlap"
    elif set(vive_calibration_keys) & set(vive_evaluation_keys):
        failure = "persisted_vive_calibration_evaluation_overlap"
    elif any(stream_hashes and (len(stream_hashes) != 1 or partition_hash not in stream_hashes)
             for stream_hashes in (ev3_hashes, vive_hashes)):
        failure = "partition_hash_mismatch"
    elif not set(eligible_eval).issubset(set(evaluation_keys)):
        failure = "invalid_eligible_evaluation_fitting_keys"
    from offstap.core.support import validate_arm_support
    for frame, expected in ((ev3_cal, calibration_keys), (ev3_eval, evaluation_keys),
                            (vive_cal, vive_calibration_keys),
                            (vive_eval, vive_evaluation_keys)):
        if failure:
            break
        if frame.empty or not expected:
            failure = "missing_persisted_stream_ids"
            break
        key_col = "source_row_index" if "source_row_index" in frame.columns else "support_key"
        validation = validate_arm_support(frame, [key_col])
        frame_keys = _h3_keys(frame[key_col]) if key_col in frame.columns else set()
        if not validation.valid or not set(expected).issubset(frame_keys):
            failure = "missing_persisted_stream_ids"
            break

    # Restrict fitting frames by persisted source identity.  Full mode uses
    # every available persisted row; split mode cannot fall back to full data.
    fit_ev3 = pd.concat([ev3_cal, ev3_eval], ignore_index=True) if mode == "full" else ev3_cal.copy()
    fit_vive = pd.concat([vive_cal, vive_eval], ignore_index=True) if mode == "full" else vive_cal.copy()
    def _filter_keys(frame: pd.DataFrame, keys: set[Any]) -> pd.DataFrame:
        return _h3_filter_keys(frame, keys)
    fit_ev3 = _filter_keys(fit_ev3, fit_keys)
    vive_fit_keys = set(vive_calibration_keys) if mode == "split" else (set(vive_calibration_keys) | set(vive_evaluation_keys))
    fit_vive = _filter_keys(fit_vive, vive_fit_keys)
    temporal: dict[str, Any] = {}
    spatial: Any = None
    clock = _ClockFit({"selected_model": "constant_offset", "offset_b": 0.0, "scale_a": 1.0})
    if failure is None and eval_duration < min_eval:
        failure = "insufficient_evaluation_duration"
    elif failure is None and (fit_ev3.empty or fit_vive.empty):
        failure = "missing_or_empty_fitting_partition"
    elif failure is None:
        try:
            raw = estimate_temporal_alignment(fit_ev3.copy(), fit_vive.copy(), cfg)
            temporal = dict(raw) if isinstance(raw, Mapping) else dict(vars(raw)) if hasattr(raw, "__dict__") else {}
            temporal_status = str(temporal.get("status", "")).strip().lower()
            if temporal_status and temporal_status not in {"ok", "success", "valid", "accepted"}:
                failure = "invalid_temporal_status"
                raise RuntimeError(failure)
            clock = _ClockFit({"selected_model": "constant_offset", "offset_b": temporal.get("b_final", temporal.get("b_onset", 0.0)), "scale_a": 1.0})
            try:
                spatial = estimate_spatial_transform(fit_ev3.copy(), fit_vive.copy(), cfg, temporal, clock)
            except TypeError:
                spatial = estimate_spatial_transform(fit_ev3.copy(), fit_vive.copy(), cfg)
        except Exception as exc:  # fail closed and preserve reason
            failure = "invalid_temporal_status" if str(exc) == "invalid_temporal_status" else f"fit_failed:{exc}"
            clock = _ClockFit({"selected_model": "constant_offset", "offset_b": 0.0, "scale_a": 1.0})
    evaluation = _h3_get(
        trial,
        f"{mode}_evaluation",
        f"{mode}_evaluation_frame",
        "evaluation",
        "evaluation_frame",
        "evaluation_data",
        default=ev3_eval,
    )
    if isinstance(evaluation, Mapping):
        evaluation = evaluation.get(mode, evaluation.get("evaluation", ev3_eval))
    evaluation = evaluation.copy() if isinstance(evaluation, pd.DataFrame) else pd.DataFrame()
    evaluation = _filter_keys(evaluation, evaluation_keys) if not evaluation.empty else evaluation
    support = _h3_frame_keys(evaluation) & evaluation_keys if evaluation_keys else set()
    diagnostics = {
        "status": "FAILED" if failure else "ok",
        "reason": failure,
        "fit_source": "calibration_only" if mode == "split" else "full_trajectory_intentionally_leaky",
        "fit_source_count": len(fit_keys),
        "fit_source_key_hash": _h3_key_hash(fit_keys) if fit_keys else "",
        "eligible_evaluation_fitting_key_hash": _h3_key_hash(eligible_eval) if eligible_present else "",
        "eligible_evaluation_fitting_key_hash_provenance": "persisted_partition_ledger" if eligible_present else "",
        "partition_hash_provenance": "persisted_partition_ledger",
        "evaluation_key_count": len(evaluation_keys),
        "eligible_evaluation_fitting_key_count": len(eligible_eval),
        "split_excludes_eligible_evaluation": not bool(set(fit_keys) & eligible_eval) if mode == "split" else None,
        "full_includes_eligible_evaluation": bool(set(fit_keys) & eligible_eval) if mode == "full" else None,
        "evaluation_duration_s": eval_duration,
        "minimum_evaluation_duration_s": min_eval,
        "temporal_status": str(temporal.get("status", "")),
        "temporal_reason": str(temporal.get("rejection_reason", "")),
    }
    return Arm(mode, temporal, clock, spatial, evaluation, support, diagnostics, fit_keys, partition_hash, mode)


def compare_h3_arms(split_arm: Arm, full_arm: Arm) -> H3Comparison:
    """Score split/full H3 arms on identical finite held-out source support."""
    if split_arm.partition_hash and full_arm.partition_hash and split_arm.partition_hash != full_arm.partition_hash:
        return H3Comparison(split_arm, full_arm, diagnostics={"status": "invalid", "reason": "partition_hash_mismatch"})
    if split_arm.diagnostics.get("status") != "ok" or full_arm.diagnostics.get("status") != "ok":
        reason = split_arm.diagnostics.get("reason") or full_arm.diagnostics.get("reason") or "invalid_fit"
        return H3Comparison(split_arm, full_arm, diagnostics={"status": "invalid", "reason": reason})
    for arm in (split_arm, full_arm):
        duration = arm.diagnostics.get("evaluation_duration_s")
        minimum = arm.diagnostics.get("minimum_evaluation_duration_s", 0.0)
        try:
            if duration is not None and float(duration) < float(minimum):
                return H3Comparison(split_arm, full_arm, diagnostics={"status": "invalid", "reason": "insufficient_evaluation_duration"})
        except (TypeError, ValueError):
            return H3Comparison(split_arm, full_arm, diagnostics={"status": "invalid", "reason": "invalid_evaluation_duration"})
    split_finite = _finite_metric_rows(split_arm.evaluation)
    full_finite = _finite_metric_rows(full_arm.evaluation)
    key_col = "support_key" if "support_key" in split_finite.columns and "support_key" in full_finite.columns else "source_row_index"
    from offstap.core.support import intersect_support
    result = intersect_support({"split": split_finite, "full": full_finite}, [key_col])
    common = {row[0] for row in result.common_keys.itertuples(index=False, name=None)} if result.valid else set()
    split_rmse = _rmse_on_support(split_arm, common)
    full_rmse = _rmse_on_support(full_arm, common)
    delta = split_rmse - full_rmse if np.isfinite(split_rmse) and np.isfinite(full_rmse) else float("nan")
    support_hash = result.common_support_hash_sha256 if common and result.valid else ""
    valid = bool(common) and np.isfinite(delta)
    diagnostics = {"status": "valid" if valid else "invalid", "reason": None if valid else (result.status if not result.valid else "insufficient_evaluation_support"), "common_support_count": len(common), "common_support_hash": support_hash, "duplicate_key_count": sum(result.duplicate_key_counts.values())}
    return H3Comparison(split_arm, full_arm, common, split_rmse, full_rmse, delta, diagnostics)


def compare_h1_arms(arms: Mapping[str, Arm]) -> H1Comparison:
    """Score both H1 arms on the exact held-out support intersection."""
    onset, refined = arms["onset"], arms["refined"]
    # Revalidate the exact canonical key column so duplicate, non-integral, or
    # unsafe identities invalidate the comparison rather than being rounded.
    onset_finite = _finite_metric_rows(onset.evaluation)
    refined_finite = _finite_metric_rows(refined.evaluation)
    key_column = "support_key" if "support_key" in onset_finite.columns and "support_key" in refined_finite.columns else "source_row_index"
    from offstap.core.support import intersect_support
    support_result = intersect_support({"onset": onset_finite, "refined": refined_finite}, [key_column])
    common = {row[0] for row in support_result.common_keys.itertuples(index=False, name=None)} if support_result.valid else set()
    ro = _rmse_on_support(onset, common)
    rr = _rmse_on_support(refined, common)
    delta = ro - rr if np.isfinite(ro) and np.isfinite(rr) else float("nan")
    diagnostics = {
        "common_support_count": len(common),
        # Use the canonical support-contract hash produced by intersect_support
        # rather than re-serializing keys in this consumer.
        "common_support_hash": support_result.common_support_hash_sha256 if common else "",
        "common_support_hash_sha256": support_result.common_support_hash_sha256 if common else "",
        "onset_support_count": len(onset.support), "refined_support_count": len(refined.support),
        "duplicate_key_count": sum(support_result.duplicate_key_counts.values()),
        "support_status": support_result.status,
        "status": "valid" if common and np.isfinite(delta) and refined.diagnostics.get("status") == "ok" else "invalid",
        "reason": (None if common and np.isfinite(delta) and refined.diagnostics.get("status") == "ok"
                   else (refined.diagnostics.get("status") if refined.diagnostics.get("status") not in (None, "ok") else "insufficient_overlap")),
    }
    diagnostics["rejection_reason"] = diagnostics["reason"]
    return H1Comparison(onset, refined, common, ro, rr, delta, diagnostics)


def _h2_frame(value: Any) -> pd.DataFrame:
    """Convert a partition component to a defensive DataFrame copy."""
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value) if value is not None else pd.DataFrame()


def _h2_eval_frame(partition: Any) -> pd.DataFrame:
    """Extract the persisted EV3 evaluation frame accepted by H2 callers."""
    _, _, ev3_eval, _ = _partition_frames(partition)
    if ev3_eval.empty and isinstance(partition, pd.DataFrame):
        ev3_eval = partition.copy()
    # H2 is held-out by construction.  Do not silently treat an unlabelled
    # frame as evaluation data or fabricate identities from row positions.
    if "partition" not in ev3_eval.columns or "source_row_index" not in ev3_eval.columns:
        return pd.DataFrame()
    ev3_eval = ev3_eval[ev3_eval["partition"].astype(str).str.lower().eq("evaluation")].copy()
    return ev3_eval.reset_index(drop=True)


def _h2_get(mapping: Any, *names: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        for name in names:
            if name in mapping:
                return mapping[name]
    for name in names:
        value = getattr(mapping, name, None)
        if value is not None:
            return value
    return default


def build_h2_comparison(frozen_params: Any, partition: Any,
                        methods: Any = ("zoh", "linear", "pchip", "spline")) -> pd.DataFrame:
    """Build the corrected four-arm Vive-reference H2 comparison.

    ``frozen_params`` contains the already-fitted temporal/clock/spatial
    parameters and cleaned Vive samples.  The EV3 evaluation frame is read
    from ``partition`` and is never resampled.  Every method is queried at the
    same mapped EV3 timestamps and scored on the exact intersection of finite
    source identities across all four arms.
    """
    methods = tuple(str(m).lower() for m in methods)
    if len(methods) != 4 or set(methods) != {"zoh", "linear", "pchip", "spline"}:
        raise ValueError("H2 requires exactly the four Vive operators: zoh, linear, pchip, spline")
    ev3 = _h2_eval_frame(partition)
    if ev3.empty:
        return pd.DataFrame(columns=["trial_id", "method", "vive_operator", "phase", "ev3_source_row_index", "source_timestamp_ns", "partition", "query_time_s", "mapped_query_time_s", "rmse_mm", "common_support_hash", "common_support_hash_sha256", "common_support_count", "extrapolated_count", "arm_extrapolated_count", "status", "fit_or_resampling_status", "reason"])

    # Frozen inputs may be a mapping keyed by trial; support both that form and
    # one shared fixture.  No fitting is performed here.
    fp = frozen_params if frozen_params is not None else {}
    shared_vive = _h2_get(fp, "vive_samples", "vrmt_samples", "vive", "vrmt")
    trials = list(ev3.groupby("trial_id", sort=False)) if "trial_id" in ev3.columns else [("", ev3)]
    rows: list[dict[str, Any]] = []
    for trial_id, ev in trials:
        trial_fp = fp.get(str(trial_id), fp.get(trial_id, fp)) if isinstance(fp, Mapping) else fp
        vive = _h2_get(trial_fp, "vive_samples", "vrmt_samples", "vive", "vrmt", default=shared_vive)
        if vive is None or (isinstance(vive, pd.DataFrame) and vive.empty):
            continue
        key_col = "source_row_index" if "source_row_index" in ev.columns else None
        if key_col is None:
            # Never fabricate positional identities for paired H2 support.
            continue
        keys = ev[key_col].to_numpy(copy=True)
        from offstap.core.support import validate_arm_support
        support_validation = validate_arm_support(ev, [key_col])
        # Canonical H2 identities are persisted integer source-row keys.  A
        # textual key, fractional value, NaN, or duplicate invalidates the
        # trial rather than being coerced or silently dropped.
        numeric_keys = pd.to_numeric(ev[key_col], errors="coerce")
        if (not support_validation.valid or numeric_keys.isna().any()
                or any(isinstance(value, (str, bytes)) for value in keys)
                or not np.isfinite(numeric_keys.to_numpy(float)).all()
                or not np.equal(numeric_keys.to_numpy(float), np.floor(numeric_keys.to_numpy(float))).all()):
            continue
        # Persisted mapped query times take precedence; otherwise apply the
        # frozen affine mapping after reading original EV3 times.
        q = _h2_get(trial_fp, "mapped_query_times", "query_times", "t_common_s")
        if q is None:
            t_col = next((c for c in ("local_time_s", "ev3_time_original_s", "time_s") if c in ev.columns), None)
            if t_col is None:
                continue
            base_t = pd.to_numeric(ev[t_col], errors="coerce").to_numpy(float)
            clock = _h2_get(trial_fp, "clock", "clock_params", "temporal", default={})
            a = float(_h2_get(clock, "scale_a", "a_selected", "a", default=1.0))
            b = float(_h2_get(clock, "offset_b", "b_selected", "b", default=0.0))
            q = a * base_t + b
        q = np.asarray(q, dtype=float).reshape(-1)
        if len(q) != len(ev):
            continue
        x_col = next((c for c in ("odometry_x", "ev3_x", "x") if c in ev.columns), None)
        y_col = next((c for c in ("odometry_y", "ev3_y", "y") if c in ev.columns), None)
        if x_col is None or y_col is None:
            continue
        ex = pd.to_numeric(ev[x_col], errors="coerce").to_numpy(float)
        ey = pd.to_numeric(ev[y_col], errors="coerce").to_numpy(float)
        queries = {m: query_reference(m, vive, q) for m in methods}
        finite_common = np.isfinite(ex) & np.isfinite(ey)
        for result in queries.values():
            finite_common &= result.valid_mask
        candidate_keys = keys[finite_common]
        from offstap.core.support import intersect_support
        arm_support = {m: pd.DataFrame({"source_row_index": candidate_keys}) for m in methods}
        common_result = intersect_support(arm_support, ["source_row_index"])
        if not common_result.valid:
            continue
        common_keys = common_result.common_keys["source_row_index"].to_numpy(copy=True)
        if not len(common_keys):
            continue
        support_hash = common_result.common_support_hash_sha256
        spatial = _h2_get(trial_fp, "spatial", "spatial_params", default=None)
        for method in methods:
            result = queries[method]
            rx, ry = result.x.copy(), result.y.copy()
            transform_error = None
            if spatial is not None and spatial:
                try:
                    from offstap.core.spatial import apply_spatial_transform
                    rx, ry = apply_spatial_transform(rx, ry, spatial)
                except (TypeError, ValueError, KeyError) as exc:
                    transform_error = str(exc)
            elif spatial is None:
                transform_error = "missing_frozen_spatial_parameters"
            support_idx = np.flatnonzero(finite_common)
            if transform_error is None:
                residual = np.sqrt((ex - rx) ** 2 + (ey - ry) ** 2)
                rmse_mm = float(np.sqrt(np.mean(residual[finite_common] ** 2)) * 1000.0)
                status, fit_status, reason = "valid", "success", ""
            else:
                rmse_mm = np.nan
                status, fit_status, reason = "FAILED", "failed", transform_error
            # One aggregate row per trial/method/phase is the statistical H2
            # grain.  The tuple of persisted identities is retained intact so
            # downstream audits can prove exact support without duplicate pivot
            # keys; per-key diagnostics can be reconstructed from the tuple.
            rows.append({
                    "trial_id": trial_id,
                    "method": method,
                    "vive_operator": method,
                    "phase": "all",
                    "ev3_source_row_index": tuple(keys[support_idx].tolist()),
                    "source_timestamp_ns": tuple(ev.iloc[support_idx]["source_timestamp_ns"].tolist()) if "source_timestamp_ns" in ev.columns else tuple(),
                    "partition": "evaluation",
                    "query_time_s": tuple(q[support_idx].tolist()),
                    "mapped_query_time_s": tuple(q[support_idx].tolist()),
                    "rmse_mm": rmse_mm,
                    "common_support_hash": support_hash,
                    "common_support_hash_sha256": support_hash,
                    "common_support_count": int(len(common_keys)),
                    # Extrapolated query points are excluded before the common
                    # support is formed; therefore scored rows contain none.
                    "extrapolated_count": 0,
                    "arm_extrapolated_count": int(result.extrapolated_count),
                    "status": status,
                    "fit_or_resampling_status": fit_status,
                    "reason": reason,
                })
    return pd.DataFrame(rows)

def main():
    cfg = load_config()
    prov = ProvenanceTracker(cfg.paths.get("output_provenance", "outputs/provenance"))

    base_out = cfg.paths.get("output_base", "outputs")
    metrics_path = os.path.join(base_out, "14_metrics", "evaluation_metrics.csv")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Error: {metrics_path} not found.")

    df = pd.read_csv(metrics_path)

    # Normally we'd take two configs from CLI arguments.
    # If only one configuration exists, record that no comparison was possible.
    configs = df["config_id"].unique() if "config_id" in df.columns else ["default"]

    comp_res = {}
    if len(configs) >= 2:
        for metric in ["ate_rmse", "rpe_rmse", "ate_mae", "ate_trimmed_rmse", "ate_huber_loss"]:
            if metric in df.columns:
                comp_res[metric] = compare_configurations(df, configs[0], configs[1], metric)
    else:
        comp_res["status"] = "only_one_config_tested_so_no_comparison_made"

    comp_dir = os.path.join(base_out, "16_config_comparison")
    os.makedirs(comp_dir, exist_ok=True)
    comp_path = os.path.join(comp_dir, "configuration_comparison.json")
    with open(comp_path, 'w') as f:
        json.dump(comp_res, f, indent=2)

    per_trial_path = os.path.join(base_out, "14_metrics", "per_trial_metrics.csv")
    if not os.path.exists(per_trial_path):
        raise FileNotFoundError(f"Error: {per_trial_path} not found.")

    df_trial = pd.read_csv(per_trial_path)

    # H1/H2/H3 are computed from real cleaned streams, aligned trajectories, and frozen parameters.

    if not df_trial.empty:
        trials = df_trial["trial_id"].tolist()

        # Corrected H2: the factor is Vive representation/query policy.  EV3
        # evaluation rows, mapped query times, and all fitted parameters are
        # frozen once per trial; no EV3-resampling or native continuous-time
        # arm enters the four-arm comparison.
        h2_dir = os.path.join(base_out, "15_hypotheses", "H2")
        os.makedirs(h2_dir, exist_ok=True)
        print("Running corrected H2 (four Vive reference operators) on accepted trials...")
        partition_path = os.path.join(base_out, "08_calibration_evaluation_split", "partition.csv")
        h2_parts = {}
        h2_failures = {}
        if os.path.exists(partition_path):
            ptab = pd.read_csv(partition_path)
            for tid, group in ptab.groupby("trial_id"):
                ev = group[(group["stream"].astype(str).str.lower() == "ev3") & (group["partition"].astype(str).str.lower() == "evaluation")]
                if not ev.empty and str(ev.iloc[0].get("status", "success")).lower() == "success":
                    ids = json.loads(ev.iloc[0]["source_row_index"]) if isinstance(ev.iloc[0].get("source_row_index"), str) else []
                    epath = os.path.join(base_out, "03_cleaned", "ev3", f"{tid}_ev3_cleaned.parquet")
                    vpath = os.path.join(base_out, "03_cleaned", "vrmt", f"{tid}_vrmt_cleaned.parquet")
                    if os.path.exists(epath) and os.path.exists(vpath):
                        eframe = pd.read_parquet(epath)
                        if "source_row_index" in eframe.columns:
                            eframe = eframe[eframe["source_row_index"].isin(ids)].copy()
                            # Partition membership is persisted in the ledger,
                            # not necessarily in cleaned Parquet; attach the
                            # already-verified label at this boundary.
                            eframe["partition"] = "evaluation"
                            h2_parts[str(tid)] = {"evaluation": {"ev3": eframe}}
        h2_frozen = {}
        for tid in trials:
            epath = os.path.join(base_out, "03_cleaned", "ev3", f"{tid}_ev3_cleaned.parquet")
            vpath = os.path.join(base_out, "03_cleaned", "vrmt", f"{tid}_vrmt_cleaned.parquet")
            if not (os.path.exists(epath) and os.path.exists(vpath)):
                continue
            ev_full, vive_full = pd.read_parquet(epath), pd.read_parquet(vpath)
            # Project raw Vive x/y/z into the configured planar frame once
            # before freezing H2.  In particular, direct-map configurations may
            # use x/z rather than the raw vertical y column; every operator
            # must receive these identical projected samples.
            try:
                from offstap.core.spatial import get_planar_trajectory
                px, py, _, _ = get_planar_trajectory(cfg, vive_full)
                vive_reference = vive_full.copy()
                vive_reference["planar_x"], vive_reference["planar_y"] = px, py
            except (KeyError, ValueError, TypeError) as exc:
                # Never query raw 3-D coordinates as if they were the planar
                # reference.  Preserve an explicit rejection for auditability.
                h2_failures[str(tid)] = f"vive_projection_failed: {exc}"
                continue
            # Parameter fitting is calibration-only; temporal mapping precedes
            # the spatial fit and is frozen before any H2 query occurs.
            cal_ids = set()
            if os.path.exists(partition_path):
                grp = pd.read_csv(partition_path)
                grp = grp[(grp["trial_id"].astype(str) == str(tid)) & (grp["stream"].astype(str).str.lower() == "ev3") & (grp["partition"].astype(str).str.lower() == "calibration")]
                if not grp.empty and isinstance(grp.iloc[0].get("source_row_index"), str):
                    cal_ids = set(json.loads(grp.iloc[0]["source_row_index"]))
            ev_cal = ev_full[ev_full.get("source_row_index", pd.Series(dtype=object)).isin(cal_ids)] if cal_ids and "source_row_index" in ev_full.columns else ev_full.iloc[0:0]
            if ev_cal.empty:
                h2_failures[str(tid)] = "missing_or_empty_persisted_ev3_calibration_partition"
                continue
            vive_cal = None
            if os.path.exists(partition_path):
                vg = pd.read_csv(partition_path)
                vg = vg[(vg["trial_id"].astype(str) == str(tid)) & (vg["stream"].astype(str).str.lower().isin(("vive", "vrmt"))) & (vg["partition"].astype(str).str.lower() == "calibration")]
                if not vg.empty and isinstance(vg.iloc[0].get("source_row_index"), str) and "source_row_index" in vive_full.columns:
                    vive_cal = vive_full[vive_full["source_row_index"].isin(json.loads(vg.iloc[0]["source_row_index"]))].copy()
            if vive_cal is None or vive_cal.empty:
                h2_failures[str(tid)] = "missing_or_empty_persisted_vive_calibration_partition"
                continue
            temp = estimate_temporal_alignment(ev_cal, vive_cal, cfg)
            clock = {"selected_model": "constant_offset", "offset_b": float(temp.get("b_final", temp.get("b_onset", 0.0))), "scale_a": 1.0}
            sp = estimate_spatial_transform(ev_cal, vive_cal, cfg, temp, clock)
            h2_frozen[str(tid)] = {"vive_samples": vive_reference, "clock": clock, "spatial": sp}
        h2_rows = []
        for tid, part in h2_parts.items():
            trial_rows = build_h2_comparison(h2_frozen.get(tid, {}), part, ("zoh", "linear", "pchip", "spline"))
            if not trial_rows.empty:
                h2_rows.append(trial_rows)
        for tid, reason in h2_failures.items():
            h2_rows.append(pd.DataFrame([{
                "trial_id": tid, "method": method, "vive_operator": method,
                "phase": "all", "ev3_source_row_index": tuple(),
                "source_timestamp_ns": tuple(), "partition": "evaluation",
                "query_time_s": tuple(), "mapped_query_time_s": tuple(),
                "rmse_mm": np.nan, "common_support_hash": "",
                "common_support_hash_sha256": "", "common_support_count": 0,
                "extrapolated_count": 0, "arm_extrapolated_count": 0,
                "status": "FAILED", "fit_or_resampling_status": "failed",
                "reason": reason,
            } for method in ("zoh", "linear", "pchip", "spline")]))
        h2_df = pd.concat(h2_rows, ignore_index=True) if h2_rows else pd.DataFrame()
        if not h2_df.empty:
            h2_df.to_csv(os.path.join(h2_dir, "h2_trial_comparison.csv"), index=False)
            h2_df.groupby("method", as_index=False)["rmse_mm"].mean().rename(columns={"rmse_mm": "mean_rmse_mm"}).to_csv(os.path.join(h2_dir, "h2_summary_by_method.csv"), index=False)

        # Corrected H3 Logic: every arm is built by ``fit_h3_arm`` from the
        # persisted dynamic partition, then scored by ``compare_h3_arms``.
        h3_dir = os.path.join(base_out, "15_hypotheses", "H3")
        os.makedirs(h3_dir, exist_ok=True)
        h3_metrics = []
        print("Running corrected H3 Logic (persisted split vs intentionally leaky full fit)...")
        partition_path = os.path.join(base_out, "08_calibration_evaluation_split", "partition.csv")
        if not os.path.exists(partition_path):
            raise RuntimeError("FAILED: persisted partition.csv is missing; H3 cannot construct its estimand")
        partition_df = pd.read_csv(partition_path)
        partition_rows = {}
        for tid, group in partition_df.groupby("trial_id"):
            group = group.copy()
            if "status" in group.columns and not group["status"].astype(str).str.lower().isin({"success", "ok", "valid", "accepted"}).all():
                continue
            def _row_keys(frame, label):
                rows = frame[frame["partition"].astype(str).str.lower().eq(label)]
                if rows.empty or "source_row_index" not in rows.columns:
                    return set()
                value = rows.iloc[0]["source_row_index"]
                return _h3_keys(value)
            ev3 = group[group["stream"].astype(str).str.lower().eq("ev3")]
            vive = group[group["stream"].astype(str).str.lower().isin({"vive", "vrmt"})]
            cal, ev = _row_keys(ev3, "calibration"), _row_keys(ev3, "evaluation")
            vcal, vev = _row_keys(vive, "calibration"), _row_keys(vive, "evaluation")
            eval_row = ev3[ev3["partition"].astype(str).str.lower().eq("evaluation")]
            cal_row = ev3[ev3["partition"].astype(str).str.lower().eq("calibration")]
            if not cal_row.empty and not eval_row.empty:
                row = eval_row.iloc[0]
                def _row_eligible(rows):
                    eligible = set()
                    present = False
                    if rows.empty:
                        return eligible, present
                    for _, candidate in rows.iterrows():
                        for col in ("eligible_evaluation_fitting_keys", "eligible_eval_fitting_keys", "eligible_evaluation_fitting_key", "evaluation_fitting_keys", "evaluation_fitting_key"):
                            if col in candidate.index and pd.notna(candidate[col]):
                                present = True
                                eligible |= _h3_keys(candidate[col])
                    return eligible, present
                elig, elig_present = _row_eligible(eval_row)
                vive_elig, vive_elig_present = _row_eligible(vive[vive["partition"].astype(str).str.lower().eq("evaluation")])
                ev3_hashes = {_h3_normalise_hash(v) for v in ev3.get("partition_hash", pd.Series(dtype=object)).tolist()}
                vive_hashes = {_h3_normalise_hash(v) for v in vive.get("partition_hash", pd.Series(dtype=object)).tolist()}
                ev3_hashes.discard("")
                vive_hashes.discard("")
                for col in ("eligible_evaluation_fitting_keys", "eligible_eval_fitting_keys", "eligible_evaluation_fitting_key", "evaluation_fitting_keys", "evaluation_fitting_key"):
                    # retained for compatibility with ledgers carrying only
                    # one stream's eligibility field
                    if col in row.index and pd.notna(row[col]):
                        elig |= _h3_keys(row[col])
                partition_rows[str(tid)] = {
                    "calibration_keys": cal, "evaluation_keys": ev,
                    "vive_calibration_keys": vcal, "vive_evaluation_keys": vev,
                    "eligible_evaluation_fitting_keys": elig,
                    "vive_eligible_evaluation_fitting_keys": vive_elig,
                    "eligible_keys_present": bool(elig_present and vive_elig_present),
                    "eligible_evaluation_fitting_key_hash": _h3_normalise_hash(row.get("eligible_evaluation_fitting_key_hash", row.get("eligible_evaluation_fitting_keys_hash"))),
                    "vive_eligible_evaluation_fitting_key_hash": _h3_normalise_hash(vive[vive["partition"].astype(str).str.lower().eq("evaluation")].iloc[0].get("eligible_evaluation_fitting_key_hash", vive[vive["partition"].astype(str).str.lower().eq("evaluation")].iloc[0].get("eligible_evaluation_fitting_keys_hash"))) if vive_elig_present and not vive[vive["partition"].astype(str).str.lower().eq("evaluation")].empty else "",
                    "ev3_partition_hashes": ev3_hashes,
                    "vive_partition_hashes": vive_hashes,
                    "calibration_end_s": float(cal_row.iloc[0].get("calibration_end_s", 0.0)),
                    "evaluation_duration_s": float(row.get("evaluation_duration_s", row.get("evaluation_end_s", 0.0) - cal_row.iloc[0].get("calibration_end_s", 0.0))),
                    "partition_hash": _h3_normalise_hash(row.get("partition_hash", cal_row.iloc[0].get("partition_hash", ""))),
                    "ledger": group,
                }

        def _record_h3_failure(trial_id, reason, info=None):
            info = info or {}
            for mode in ("calibration_evaluation_split", "full_trajectory_calibration"):
                h3_metrics.append({
                    "trial_id": trial_id,
                    "calibration_mode": mode,
                    "status": "FAILED",
                    "reason": reason,
                    "valid_for_paper": False,
                    "partition_hash": info.get("partition_hash", ""),
                    "partition_hash_provenance": "persisted_partition_ledger",
                    "calibration_interval_s": info.get("calibration_interval_s", np.nan),
                    "evaluation_interval_s": info.get("evaluation_interval_s", np.nan),
                    "heldout_rmse_mm": np.nan,
                    "delta_rmse_mm": np.nan,
                    "common_support_count": 0,
                    "common_support_hash": "",
                    "common_support_hash_sha256": "",
                    "evaluation_support_count": 0,
                    "evaluation_support_hash": "",
                    "fit_source_key_count": np.nan,
                    "fit_source_key_hash": "",
                    "eligible_evaluation_fitting_key_count": info.get("eligible_evaluation_fitting_key_count", np.nan),
                    "eligible_evaluation_fitting_key_hash": info.get("eligible_evaluation_fitting_key_hash", ""),
                    "eligible_evaluation_fitting_key_hash_provenance": "persisted_partition_ledger",
                    "split_excludes_evaluation_fitting": np.nan,
                    "full_includes_evaluation_fitting": np.nan,
                    "leakage_status": "",
                    "fit_status": info.get(f"{mode}_fit_status", ""),
                    "fit_reason": info.get(f"{mode}_fit_reason", reason),
                })

        import pickle
        from offstap.core.alignment import generate_aligned_trajectory
        from offstap.core.trajectory_reference import ContinuousTrajectory
        for tid in trials:
            persisted = partition_rows.get(str(tid))
            if persisted is None:
                _record_h3_failure(tid, "missing_or_rejected_persisted_partition")
                continue
            epath = os.path.join(base_out, "03_cleaned", "ev3", f"{tid}_ev3_cleaned.parquet")
            vpath = os.path.join(base_out, "03_cleaned", "vrmt", f"{tid}_vrmt_cleaned.parquet")
            if not (os.path.exists(epath) and os.path.exists(vpath)):
                _record_h3_failure(tid, "missing_cleaned_stream_parquet")
                continue
            ev3_full, vive_full = pd.read_parquet(epath), pd.read_parquet(vpath)
            if "source_row_index" not in ev3_full.columns or "source_row_index" not in vive_full.columns:
                _record_h3_failure(tid, "missing_persisted_stream_ids")
                continue
            keys = persisted
            if not keys["calibration_keys"] or not keys["evaluation_keys"] or not keys["vive_calibration_keys"] or not keys["vive_evaluation_keys"] or not keys["eligible_keys_present"]:
                _record_h3_failure(tid, "missing_persisted_h3_keys_or_partition_hash", {"partition_hash": keys.get("partition_hash", "")})
                continue
            partition = {"ev3_calibration": _h3_filter_keys(ev3_full, keys["calibration_keys"]), "vive_calibration": _h3_filter_keys(vive_full, keys["vive_calibration_keys"]), "ev3_evaluation": _h3_filter_keys(ev3_full, keys["evaluation_keys"]), "vive_evaluation": _h3_filter_keys(vive_full, keys["vive_evaluation_keys"]), "partition_hash": keys["partition_hash"], "evaluation_duration_s": keys["evaluation_duration_s"], "ledger": keys.get("ledger")}
            trial = {"trial_id": str(tid), "partition": partition, "calibration_keys": keys["calibration_keys"], "evaluation_keys": keys["evaluation_keys"], "vive_calibration_keys": keys["vive_calibration_keys"], "vive_evaluation_keys": keys["vive_evaluation_keys"], "eligible_evaluation_fitting_keys": keys["eligible_evaluation_fitting_keys"], "vive_eligible_evaluation_fitting_keys": keys.get("vive_eligible_evaluation_fitting_keys", set()), "minimum_evaluation_duration_s": _h3_min_eval(cfg, {}) if cfg is not None else 0.0, "partition_hash": keys["partition_hash"], "eligible_keys_present": keys["eligible_keys_present"]}
            split_arm, full_arm = fit_h3_arm(trial, "split", cfg), fit_h3_arm(trial, "full", cfg)
            spline_file = os.path.join(base_out, "11_continuous_time_reference", "models", f"{tid}_spline.pkl")
            if split_arm.diagnostics.get("status") != "ok" or full_arm.diagnostics.get("status") != "ok":
                _record_h3_failure(tid, split_arm.diagnostics.get("reason") or full_arm.diagnostics.get("reason") or "invalid_fit", {
                    "partition_hash": keys["partition_hash"],
                    "evaluation_interval_s": keys.get("evaluation_duration_s", np.nan),
                    "calibration_interval_s": keys.get("calibration_end_s", np.nan),
                    "eligible_evaluation_fitting_key_count": len(keys.get("eligible_evaluation_fitting_keys", set())),
                    "eligible_evaluation_fitting_key_hash": keys.get("eligible_evaluation_fitting_key_hash", ""),
                    "calibration_evaluation_split_fit_status": split_arm.diagnostics.get("status", ""),
                    "calibration_evaluation_split_fit_reason": split_arm.diagnostics.get("reason", ""),
                    "full_trajectory_calibration_fit_status": full_arm.diagnostics.get("status", ""),
                    "full_trajectory_calibration_fit_reason": full_arm.diagnostics.get("reason", ""),
                })
                continue
            if not os.path.exists(spline_file):
                _record_h3_failure(tid, "missing_reference_spline", {
                    "partition_hash": keys["partition_hash"],
                    "evaluation_interval_s": keys.get("evaluation_duration_s", np.nan),
                    "calibration_interval_s": keys.get("calibration_end_s", np.nan),
                })
                continue
            with open(spline_file, "rb") as handle:
                spline_data = pickle.load(handle)
            if not spline_data.get("valid", False):
                _record_h3_failure(tid, "invalid_reference_spline", {
                    "partition_hash": keys["partition_hash"],
                    "evaluation_interval_s": keys.get("evaluation_duration_s", np.nan),
                    "calibration_interval_s": keys.get("calibration_end_s", np.nan),
                })
                continue
            spline = ContinuousTrajectory(spline_data["t"], spline_data["x"], spline_data["y"])
            for arm in (split_arm, full_arm):
                aligned = generate_aligned_trajectory(ev3_full, vive_full, spline, arm.temporal, arm.clock, arm.spatial, cfg)
                arm.evaluation = _h3_filter_keys(aligned, keys["evaluation_keys"]) if isinstance(aligned, pd.DataFrame) else pd.DataFrame()
                arm.support = _h3_frame_keys(arm.evaluation)
            comparison = compare_h3_arms(split_arm, full_arm)
            common = comparison.common_support
            base = {
                "trial_id": tid,
                "evaluation_interval_s": keys["evaluation_duration_s"],
                "calibration_interval_s": keys["calibration_end_s"],
                "partition_hash": keys["partition_hash"],
                "partition_hash_provenance": "persisted_partition_ledger",
                "common_support_count": len(common),
                "common_support_hash": comparison.common_support_hash,
                "common_support_hash_sha256": comparison.common_support_hash,
                "evaluation_support_count": len(common),
                "evaluation_support_hash": comparison.common_support_hash,
                "status": comparison.status,
                "reason": comparison.reason or "",
                "valid_for_paper": comparison.status == "valid",
                "delta_rmse_mm": comparison.delta_rmse_mm,
                "eligible_evaluation_fitting_key_count": len(keys["eligible_evaluation_fitting_keys"]),
                "eligible_evaluation_fitting_key_hash": keys.get("eligible_evaluation_fitting_key_hash", ""),
                "eligible_evaluation_fitting_key_hash_provenance": "persisted_partition_ledger",
                "vive_eligible_evaluation_fitting_key_hash": keys.get("vive_eligible_evaluation_fitting_key_hash", ""),
            }
            h3_metrics.extend([
                {**base, "calibration_mode": "calibration_evaluation_split", "heldout_rmse_mm": comparison.rmse_split_mm,
                 "fit_source_key_count": split_arm.fit_key_count, "fit_source_key_hash": split_arm.diagnostics.get("fit_source_key_hash", ""),
                 "split_excludes_evaluation_fitting": split_arm.diagnostics.get("split_excludes_eligible_evaluation", False),
                 "full_includes_evaluation_fitting": np.nan, "leakage_status": split_arm.leakage_status,
                 "fit_status": split_arm.diagnostics.get("status", ""), "fit_reason": split_arm.diagnostics.get("reason", "")},
                {**base, "calibration_mode": "full_trajectory_calibration", "heldout_rmse_mm": comparison.rmse_full_mm,
                 "fit_source_key_count": full_arm.fit_key_count, "fit_source_key_hash": full_arm.diagnostics.get("fit_source_key_hash", ""),
                 "split_excludes_evaluation_fitting": np.nan, "full_includes_evaluation_fitting": full_arm.diagnostics.get("full_includes_eligible_evaluation", False),
                 "leakage_status": full_arm.leakage_status,
                 "fit_status": full_arm.diagnostics.get("status", ""), "fit_reason": full_arm.diagnostics.get("reason", "")},
            ])
        if h3_metrics:
            pd.DataFrame(h3_metrics).to_csv(os.path.join(h3_dir, "h3_calibration_mode_comparison.csv"), index=False)

        # Real H1 Logic (Onset vs Crosscorr)
        h1_dir = os.path.join(base_out, "15_hypotheses", "H1")
        os.makedirs(h1_dir, exist_ok=True)
        h1_metrics = []
        h1_tier_map = {}
        h1_manifest_path = os.path.join(base_out, "01_manifest", "pair_manifest.csv")
        if os.path.exists(h1_manifest_path):
            _manifest = pd.read_csv(h1_manifest_path)
            if {"trial_id", "dataset_tier"}.issubset(_manifest.columns):
                h1_tier_map = dict(zip(_manifest["trial_id"].astype(str), _manifest["dataset_tier"].astype(str)))

        print("Running Real H1 Logic (Onset vs Crosscorr)...")

        # Load temporal report
        crosscorr_report_path = os.path.join(base_out, "09_temporal", "crosscorr_report.csv")
        df_cc = pd.DataFrame()
        if os.path.exists(crosscorr_report_path):
            df_cc = pd.read_csv(crosscorr_report_path).set_index("trial_id")

        params_path = os.path.join(base_out, "alignment_parameters.json")
        all_params = {}
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                all_params = json.load(f)

        for tid in trials:
            persisted_partition = partition_rows.get(str(tid))
            if persisted_partition is None:
                h1_metrics.append({"trial_id": tid, "status": "FAILED", "reason": "missing_or_rejected_persisted_partition", "crosscorr_status": "failed", "crosscorr_accepted": 0, "delta_rmse_mm": np.nan, "valid_for_paper": False})
                continue
            # Evaluation membership is authoritative from the persisted
            # partition ledger.  ``evaluation_source_rows`` was a removed
            # pre-canonical alias; do not resurrect it as a fallback.
            evaluation_keys = persisted_partition.get("evaluation_keys")
            if not evaluation_keys:
                h1_metrics.append({"trial_id": tid, "status": "FAILED", "reason": "missing_persisted_evaluation_keys", "crosscorr_status": "failed", "crosscorr_accepted": 0, "delta_rmse_mm": np.nan, "valid_for_paper": False})
                continue
            evaluation_keys = set(evaluation_keys)
            e_c_path = os.path.join(base_out, "03_cleaned", "ev3", f"{tid}_ev3_cleaned.parquet")
            v_c_path = os.path.join(base_out, "03_cleaned", "vrmt", f"{tid}_vrmt_cleaned.parquet")
            if not os.path.exists(e_c_path) or not os.path.exists(v_c_path):
                h1_metrics.append({"trial_id": tid, "status": "FAILED", "reason": "missing_cleaned_stream_parquet", "crosscorr_status": "failed", "crosscorr_accepted": 0, "delta_rmse_mm": np.nan, "valid_for_paper": False})
                continue

            # Corrected H1 path: construct both temporal arms from the
            # persisted calibration rows and fit spatial parameters
            # independently under each mapping.  The legacy branch below is
            # intentionally unreachable for accepted trials; it is retained
            # only as historical context until the next cleanup pass.
            cal_dir = os.path.join(base_out, "08_calibration_evaluation_split", "calibration_data")
            e_cal_path = os.path.join(cal_dir, f"{tid}_ev3_calib.parquet")
            v_cal_path = os.path.join(cal_dir, f"{tid}_vrmt_calib.parquet")
            if os.path.exists(e_cal_path) and os.path.exists(v_cal_path):
                try:
                    ev3_cal = pd.read_parquet(e_cal_path)
                    vive_cal = pd.read_parquet(v_cal_path)
                    ev3_eval = pd.read_parquet(e_c_path)
                    ev3_eval = ev3_eval[ev3_eval["source_row_index"].isin(evaluation_keys)]
                    partition_obj = {"ev3_calibration": ev3_cal, "vive_calibration": vive_cal,
                                     "ev3_evaluation": ev3_eval,
                                     "vive_evaluation": pd.DataFrame()}
                    arms = build_h1_arms(partition_obj, cfg)
                    # Attach held-out residual frames using the persisted
                    # continuous reference when available.  This evaluation
                    # step consumes no rows for fitting.
                    spline_file = os.path.join(base_out, "11_continuous_time_reference", "models", f"{tid}_spline.pkl")
                    if os.path.exists(spline_file):
                        import pickle
                        from offstap.core.trajectory_reference import ContinuousTrajectory
                        from offstap.core.alignment import generate_aligned_trajectory
                        with open(spline_file, "rb") as handle:
                            spline_data = pickle.load(handle)
                        if spline_data.get("valid", False):
                            spline = ContinuousTrajectory(spline_data["t"], spline_data["x"], spline_data["y"])
                            for arm in arms.values():
                                aligned = generate_aligned_trajectory(
                                    pd.read_parquet(e_c_path), pd.read_parquet(v_c_path), spline,
                                    arm.temporal, arm.clock, arm.spatial, cfg,
                                )
                                arm.evaluation = aligned[aligned["support_key"].isin(evaluation_keys)] if "support_key" in aligned.columns else aligned.iloc[0:0]
                                arm.support = _source_keys(arm.evaluation)
                    comparison = compare_h1_arms(arms)
                    onset_diag, refined_diag = arms["onset"].diagnostics, arms["refined"].diagnostics
                    h1_metrics.append({
                        "trial_id": tid,
                        "tier": h1_tier_map.get(str(tid), "unknown"),
                        "rmse_onset_mm": comparison.onset_rmse * 1000.0 if np.isfinite(comparison.onset_rmse) else np.nan,
                        "rmse_refined_mm": comparison.refined_rmse * 1000.0 if np.isfinite(comparison.refined_rmse) else np.nan,
                        "delta_rmse_mm": comparison.delta_rmse * 1000.0 if np.isfinite(comparison.delta_rmse) else np.nan,
                        "num_common_samples": len(comparison.common_support),
                        "support_hash": comparison.diagnostics.get("common_support_hash", ""),
                        "common_support_count": len(comparison.common_support),
                        "common_support_hash": comparison.diagnostics.get("common_support_hash", ""),
                        "common_support_hash_sha256": comparison.diagnostics.get("common_support_hash", ""),
                        "b_onset_s": onset_diag.get("b_onset_s", np.nan),
                        "b_refined_s": refined_diag.get("b_refined_s", np.nan),
                        "delta_b_s": refined_diag.get("b_refined_s", np.nan) - onset_diag.get("b_onset_s", np.nan),
                        "abs_delta_b_s": abs(refined_diag.get("b_refined_s", np.nan) - onset_diag.get("b_onset_s", np.nan)),
                        "epsilon_t_s": onset_diag.get("b_onset_s", np.nan) - refined_diag.get("b_refined_s", np.nan),
                        "crosscorr_peak_confidence_ratio": refined_diag.get("peak_ratio", np.nan),
                        "refinement_score": refined_diag.get("refinement_score", np.nan),
                        "second_peak_score": refined_diag.get("second_peak_score", np.nan),
                        "peak_ratio": refined_diag.get("peak_ratio", np.nan),
                        "overlap_s": refined_diag.get("overlap_s", np.nan),
                        "rejection_reason": refined_diag.get("rejection_reason"),
                        "crosscorr_status": refined_diag.get("status", "invalid"),
                        "crosscorr_accepted": 1 if refined_diag.get("status") == "ok" else 0,
                        "status": comparison.diagnostics.get("status", "invalid"),
                        "reason": comparison.diagnostics.get("reason"),
                        "valid_for_paper": comparison.diagnostics.get("status") == "valid",
                        "onset_support_count": len(arms["onset"].support),
                        "refined_support_count": len(arms["refined"].support),
                    })
                except Exception as exc:
                    h1_metrics.append({"trial_id": tid, "status": "FAILED", "reason": f"h1_arm_fit_failed:{exc}", "rejection_reason": f"h1_arm_fit_failed:{exc}", "crosscorr_status": "failed", "crosscorr_accepted": 0, "delta_b_s": np.nan, "abs_delta_b_s": np.nan, "delta_rmse_mm": np.nan, "valid_for_paper": False})
                continue

            # Never fall through to the historical full-stream fitting path:
            # missing persisted calibration is an explicit H1 failure.
            h1_metrics.append({"trial_id": tid, "status": "FAILED", "reason": "missing_persisted_calibration_parquet", "rejection_reason": "missing_persisted_calibration_parquet", "crosscorr_status": "failed", "crosscorr_accepted": 0, "delta_b_s": np.nan, "abs_delta_b_s": np.nan, "delta_rmse_mm": np.nan, "valid_for_paper": False})
            continue

            df_ev3_full = pd.read_parquet(e_c_path)
            df_ev3_full = df_ev3_full.copy()
            if "source_row_index" not in df_ev3_full.columns:
                h1_metrics.append({"trial_id": tid, "status": "FAILED", "reason": "missing_source_row_index", "crosscorr_status": "failed", "crosscorr_accepted": 0, "delta_rmse_mm": np.nan, "valid_for_paper": False})
                continue
            df_ev3_full["support_key"] = pd.to_numeric(df_ev3_full["source_row_index"], errors="coerce")
            df_vrmt_full = pd.read_parquet(v_c_path)

            # Load stats from crosscorr report
            b_onset = 0.0
            b_final = 0.0
            cc_status = "missing_input"
            cc_peak_conf = 0.0

            if not df_cc.empty and tid in df_cc.index:
                row = df_cc.loc[tid]
                b_onset = float(row.get("b_onset", 0.0))
                b_final = float(row.get("b_final", 0.0))
                cc_status = str(row.get("status", "missing_input"))
                cc_peak_conf = float(row.get("peak_confidence_ratio", 0.0))

            # Use frozen spatial parameters when available; otherwise estimate from the full streams.
            spatial_full = all_params.get(tid, {}).get("spatial", None)
            if spatial_full is None:
                spatial_full = estimate_spatial_transform(df_ev3_full, df_vrmt_full, cfg)

            # Real evaluation for H1
            import pickle
            from offstap.core.alignment import generate_aligned_trajectory
            from offstap.core.metrics import evaluate_trajectory_metrics
            from offstap.core.trajectory_reference import ContinuousTrajectory

            spline_file = os.path.join(base_out, "11_continuous_time_reference", "models", f"{tid}_spline.pkl")
            if not os.path.exists(spline_file):
                h1_metrics.append({"trial_id": tid, "status": "FAILED", "reason": "missing_reference_spline", "crosscorr_status": "failed", "crosscorr_accepted": 0, "delta_rmse_mm": np.nan, "valid_for_paper": False})
                continue
            with open(spline_file, 'rb') as f:
                spline_data = pickle.load(f)
            if spline_data.get('valid', False):
                spline_traj = ContinuousTrajectory(spline_data['t'], spline_data['x'], spline_data['y'])
            else:
                h1_metrics.append({"trial_id": tid, "status": "FAILED", "reason": "invalid_reference_spline", "crosscorr_status": "failed", "crosscorr_accepted": 0, "delta_rmse_mm": np.nan, "valid_for_paper": False})
                continue

            temp_onset_dict = {"b_onset": b_onset, "b_final": b_onset, "valid": True, "status": "ok"}
            clock_onset = {"selected_model": "constant_offset", "offset_b": b_onset, "scale_a": 1.0}
            df_align_onset = generate_aligned_trajectory(df_ev3_full, df_vrmt_full, spline_traj, temp_onset_dict, clock_onset, spatial_full, cfg)

            temp_cc_dict = {"b_onset": b_onset, "b_final": b_final, "valid": (cc_status == "ok"), "status": cc_status, "peak_confidence": cc_peak_conf}
            clock_cc = {"selected_model": "constant_offset", "offset_b": b_final, "scale_a": 1.0}
            df_align_cc = generate_aligned_trajectory(df_ev3_full, df_vrmt_full, spline_traj, temp_cc_dict, clock_cc, spatial_full, cfg)
            required = ["odometry_x", "odometry_y", "vrmt_aligned_x", "vrmt_aligned_y"]
            onset_finite = df_align_onset[required].apply(np.isfinite).all(axis=1) if not df_align_onset.empty else pd.Series(dtype=bool)
            cc_finite = df_align_cc[required].apply(np.isfinite).all(axis=1) if not df_align_cc.empty else pd.Series(dtype=bool)
            onset_keys = set(df_align_onset.loc[onset_finite, "support_key"].astype(int)) & evaluation_keys if len(onset_finite) else set()
            cc_keys = set(df_align_cc.loc[cc_finite, "support_key"].astype(int)) & evaluation_keys if len(cc_finite) else set()
            common_keys = sorted(onset_keys & cc_keys)
            df_align_onset = df_align_onset[df_align_onset["support_key"].isin(common_keys)].sort_values("support_key")
            df_align_cc = df_align_cc[df_align_cc["support_key"].isin(common_keys)].sort_values("support_key")
            # Both arms have already been restricted to the persisted common
            # evaluation support (`common_keys`) above.
            metrics_onset = evaluate_trajectory_metrics(df_align_onset, cfg, evaluation_only=False)
            metrics_cc = evaluate_trajectory_metrics(df_align_cc, cfg, evaluation_only=False)
            onset_rmse = metrics_onset.get("ate_rmse", np.nan) * 1000.0 if not np.isnan(metrics_onset.get("ate_rmse", np.nan)) else np.nan
            cc_rmse = metrics_cc.get("ate_rmse", np.nan) * 1000.0 if not np.isnan(metrics_cc.get("ate_rmse", np.nan)) else np.nan
            support_hash = hashlib.sha256(
                ",".join(str(k) for k in common_keys).encode("utf-8")
            ).hexdigest() if common_keys else ""

            delta = onset_rmse - cc_rmse if np.isfinite(onset_rmse) and np.isfinite(cc_rmse) else np.nan

            # Zero paired differences and zero-valued offsets are valid data.
            # Excluding them changes both n and the Wilcoxon zero convention.
            is_valid_for_paper = bool(
                metrics_cc.get("status") == "success" and
                np.isfinite(cc_rmse) and np.isfinite(onset_rmse) and
                cc_status == "ok" and
                np.isfinite(delta)
            )

            h1_metrics.append({
                "trial_id": tid,
                "scenario_id": tid.split('_')[0] + '_' + tid.split('_')[1] if len(tid.split('_')) > 1 else 'unknown',
                "drive_id": tid.split('_')[0],
                "route_id": tid.split('_')[1] if len(tid.split('_')) > 1 else 'unknown',
                "repetition_id": 1,
                "tier": h1_tier_map.get(str(tid), "unknown"),
                "rmse_onset_mm": onset_rmse,
                "rmse_refined_mm": cc_rmse,
                "delta_rmse_mm": delta,
                "num_common_samples": len(common_keys),
                "support_hash": support_hash,
                "common_support_count": len(common_keys),
                "common_support_hash": support_hash,
                "common_support_hash_sha256": support_hash,
                "b_onset_s": b_onset,
                "b_refined_s": b_final,
                "epsilon_t_s": b_onset - b_final,
                "crosscorr_peak_confidence_ratio": cc_peak_conf,
                "crosscorr_status": cc_status,
                "crosscorr_accepted": 1 if cc_status == "ok" else 0,
                "status": "valid" if is_valid_for_paper else "invalid",
                "valid_for_paper": is_valid_for_paper
            })

        if h1_metrics:
            h1_df = pd.DataFrame(h1_metrics)
            # Ensure every branch, including early failures, persists the
            # canonical H1 schema.  No historical residual_* aliases are
            # emitted or consulted by downstream statistics.
            for column in ("tier", "rmse_onset_mm", "rmse_refined_mm", "common_support_count", "common_support_hash", "common_support_hash_sha256"):
                if column not in h1_df.columns:
                    h1_df[column] = np.nan
            h1_df["tier"] = h1_df["tier"].fillna(h1_df["trial_id"].astype(str).map(h1_tier_map)).fillna("unknown")
            # Keep diagnostics columns present even when every trial fails at
            # an early gate (e.g. missing persisted calibration).
            for column in ("delta_b_s", "abs_delta_b_s", "refinement_score",
                           "second_peak_score", "peak_ratio", "overlap_s",
                           "rejection_reason"):
                if column not in h1_df.columns:
                    h1_df[column] = np.nan
            h1_df.to_csv(os.path.join(h1_dir, "h1_trial_pairs.csv"), index=False)

            h1_sum = {
                "mean_delta_rmse": h1_df[h1_df["valid_for_paper"] == True]["delta_rmse_mm"].mean() if not h1_df[h1_df["valid_for_paper"] == True].empty else np.nan,
                "accepted_for_h1": len(h1_df[h1_df["valid_for_paper"] == True]),
                "excluded_ambiguous_count": len(h1_df[h1_df["crosscorr_status"] == "ambiguous"])
            }
            pd.DataFrame([h1_sum]).to_csv(os.path.join(h1_dir, "h1_summary.csv"), index=False)

    prov.record_stage(
        stage_name="09_compare_configs",
        config_id=cfg.config_id,
        input_files=[metrics_path],
        output_files=[comp_path],
        additional_meta={"configs_found": len(configs)}
    )
    print("Configuration comparisons complete.")

if __name__ == "__main__":
    main()
