"""Canonical, revision-preserving Stage 14 cascade producer.

This module assembles a machine-readable cascade table exclusively from
persisted upstream artifacts.  It deliberately does not fit, align, query, or
score trajectories.  Stage-specific artifacts that are not persisted are
represented as reason-coded unavailable rows rather than inferred values.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STAGES = [
    {"stage_id": "S0", "stage_name": "no_alignment", "spatial_transform_applied": False, "temporal_offset_applied": False, "clock_model_applied": False, "sampling_or_representation_method": "none", "calibration_mode": "none", "scientific_role": "descriptive_system_level", "manuscript_continuity_status": "conditional", "attribution_role": "none", "intended_purpose": "unaligned_reference"},
    {"stage_id": "S1", "stage_name": "spatial_alignment_only", "spatial_transform_applied": True, "temporal_offset_applied": False, "clock_model_applied": False, "sampling_or_representation_method": "none", "calibration_mode": "split", "scientific_role": "descriptive_system_level", "manuscript_continuity_status": "conditional", "attribution_role": "none", "intended_purpose": "spatial_stage"},
    {"stage_id": "S2", "stage_name": "spatial_plus_temporal_alignment", "spatial_transform_applied": True, "temporal_offset_applied": True, "clock_model_applied": False, "sampling_or_representation_method": "none", "calibration_mode": "split", "scientific_role": "descriptive_system_level", "manuscript_continuity_status": "conditional", "attribution_role": "none", "intended_purpose": "temporal_stage"},
    {"stage_id": "S3", "stage_name": "spatial_temporal_common_sampling", "spatial_transform_applied": True, "temporal_offset_applied": True, "clock_model_applied": False, "sampling_or_representation_method": "common_grid", "calibration_mode": "split", "scientific_role": "descriptive_system_level", "manuscript_continuity_status": "conditional", "attribution_role": "none", "intended_purpose": "sampling_stage"},
    {"stage_id": "S4", "stage_name": "final_selected_pipeline", "spatial_transform_applied": True, "temporal_offset_applied": True, "clock_model_applied": True, "sampling_or_representation_method": "continuous_time", "calibration_mode": "split", "scientific_role": "descriptive_system_level", "manuscript_continuity_status": "conditional", "attribution_role": "none", "intended_purpose": "final_pipeline"},
]


def _load_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _float_or_nan(value: Any) -> float:
    try:
        return float(value) if value is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def _canonical_support_hash(value: Any) -> tuple[str, bool]:
    """Validate/hash support keys through the shared support contract."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "", False
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, json.JSONDecodeError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return "", False
    if not isinstance(parsed, (list, tuple)):
        return "", False
    from offstap.core.support import intersect_support
    evidence = intersect_support({"canonical": pd.DataFrame({"source_row_index": list(parsed)})}, ["source_row_index"])
    return evidence.hash_sha256, evidence.valid


def produce_cascade_table(base_out: str | Path, *, write: bool = True, gate_ledger_path: str | Path | None = None) -> pd.DataFrame:
    """Build and optionally persist the frozen Stage 14 per-trial table.

    S4 is mapped to the canonical held-out ``per_trial_metrics.csv`` artifact.
    S0--S3 remain explicit ``unavailable`` rows until dedicated canonical
    stage artifacts are produced upstream; no metric is copied between stages.
    """
    base = Path(base_out)
    metrics_path = base / "14_metrics" / "per_trial_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Canonical metrics artifact is required: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    if "trial_id" not in metrics.columns:
        raise ValueError("Canonical metrics artifact lacks trial_id")

    manifest = _load_optional(base / "01_manifest" / "pair_manifest.csv")
    manifest_by_trial = manifest.set_index("trial_id", drop=False) if "trial_id" in manifest.columns else pd.DataFrame()
    partition = _load_optional(base / "08_calibration_evaluation_split" / "partition.csv")
    partition_by_trial = {}
    if not partition.empty and "trial_id" in partition.columns:
        for tid, frame in partition.groupby("trial_id"):
            cal = frame[frame.get("partition", "").astype(str).str.lower().eq("calibration")] if "partition" in frame else frame.iloc[0:0]
            partition_by_trial[str(tid)] = {
                "calibration_provenance": ";".join(sorted(set(cal.get("partition_hash", pd.Series(dtype=str)).dropna().astype(str)))),
                "partition_status": ";".join(sorted(set(frame.get("status", pd.Series(dtype=str)).dropna().astype(str)))),
                }

    gate_path = Path(gate_ledger_path) if gate_ledger_path else base / "19_tables" / "authoritative_population_gate_ledger.csv"
    gate = _load_optional(gate_path)
    gate_by_trial: dict[str, Any] = {}
    gate_selector_ok: dict[str, bool] = {}
    if "trial_id" in gate.columns:
        for tid, group in gate.groupby(gate["trial_id"].astype(str)):
            selected = group
            if "gate" in group.columns:
                selected = group[group["gate"].astype(str).eq("final_OFF_STAP_valid")]
                gate_selector_ok[str(tid)] = not selected.empty
            else:
                gate_selector_ok[str(tid)] = False
            gate_by_trial[str(tid)] = selected.iloc[0] if not selected.empty else None
    spatial = _load_optional(base / "07_spatial" / "spatial_diagnostics.csv")
    temporal = _load_optional(base / "09_temporal" / "crosscorr_report.csv")
    clock = _load_optional(base / "10_clock_model" / "clock_model_comparison.csv")
    evidence_by_trial = {
        "spatial": set(spatial.get("trial_id", pd.Series(dtype=str)).astype(str)),
        "temporal": set(temporal.get("trial_id", pd.Series(dtype=str)).astype(str)),
        "clock": set(clock.get("trial_id", pd.Series(dtype=str)).astype(str)),
    }

    rows: list[dict[str, Any]] = []
    for _, metric in metrics.iterrows():
        tid = str(metric["trial_id"])
        mrow = manifest_by_trial.loc[tid] if isinstance(manifest_by_trial, pd.DataFrame) and tid in manifest_by_trial.index else {}
        tier = str(mrow.get("dataset_tier", "UNRESOLVED")) if hasattr(mrow, "get") else "UNRESOLVED"
        partition_meta = partition_by_trial.get(tid, {})
        persisted_support_hash = str(metric.get("support_hash", "")).strip()
        expected_support_hash, support_valid = _canonical_support_hash(metric.get("support_keys"))
        partition_rows = partition[partition["trial_id"].astype(str).eq(tid)] if "trial_id" in partition.columns else pd.DataFrame()
        partition_support_hashes = set()
        for col in ("support_hash", "evaluation_support_hash", "evaluation_support_hash_sha256"):
            if col in partition_rows.columns:
                partition_support_hashes.update(partition_rows[col].dropna().astype(str).str.strip())
        metric_status = str(metric.get("metric_status", "invalid"))
        finite = bool(np.isfinite(pd.to_numeric(pd.Series([metric.get("rmse_mm", np.nan)]), errors="coerce").iloc[0]))
        gate_row = gate_by_trial.get(tid)
        gate_status = str(gate_row.get("gate_status", "")) if hasattr(gate_row, "get") else ""
        gate_reason = str(gate_row.get("reason", gate_row.get("gate_reason", ""))) if hasattr(gate_row, "get") else ""
        if tier == "UNRESOLVED" and hasattr(gate_row, "get"):
            tier = str(gate_row.get("dataset_tier", "UNRESOLVED"))
        accepted_gate_statuses = {"eligible", "valid", "accepted", "passed", "final off-stap valid", "final_off_stap_valid"}
        gate_eligible = gate_status.lower() in accepted_gate_statuses
        if gate.empty:
            s4_reason = "missing_authoritative_gate_ledger"
        elif not gate_selector_ok.get(tid, False):
            s4_reason = "missing_final_gate_selector"
        elif gate_row is None:
            s4_reason = "missing_trial_gate_record"
        elif gate_status.lower() not in accepted_gate_statuses:
            s4_reason = "gate_status_unrecognized"
        elif gate_status.lower() in {"rejected", "ineligible", "failed", "invalid"}:
            s4_reason = gate_reason or "trial_gate_ineligible"
        elif not all(tid in evidence_by_trial[name] for name in ("spatial", "temporal", "clock")):
            s4_reason = "missing_canonical_provenance_artifact"
        elif not persisted_support_hash:
            s4_reason = "support_hash_unavailable"
        elif not support_valid:
            s4_reason = "support_validation_failed"
        elif persisted_support_hash != expected_support_hash:
            s4_reason = "support_hash_mismatch"
        elif partition_support_hashes and persisted_support_hash not in partition_support_hashes:
            s4_reason = "support_hash_mismatch"
        elif metric_status != "success" or not finite:
            s4_reason = str(metric.get("reason_if_invalid", "canonical_metric_invalid"))
        else:
            s4_reason = ""
        s4_valid = not s4_reason
        for stage in STAGES:
            is_s4 = stage["stage_id"] == "S4"
            row: dict[str, Any] = {
                "trial_id": tid,
                "tier": tier,
                "drive_id": str(mrow.get("drive_id", "UNRESOLVED")) if hasattr(mrow, "get") else "UNRESOLVED",
                "route_id": str(mrow.get("route_id", "UNRESOLVED")) if hasattr(mrow, "get") else "UNRESOLVED",
                "repetition_id": mrow.get("repetition_id", "UNRESOLVED") if hasattr(mrow, "get") else "UNRESOLVED",
                "stage_id": stage["stage_id"],
                "stage_name": stage["stage_name"],
                "scientific_role": stage["scientific_role"],
                "manuscript_continuity_status": "conditional",
                "attribution_role": "none",
                "metric_validity_status": "valid" if is_s4 and s4_valid else "invalid",
                "invalidity_reason": "" if is_s4 and s4_valid else (s4_reason if is_s4 else "canonical_stage_artifact_unavailable"),
                "rmse_mm": _float_or_nan(metric.get("rmse_mm")) if is_s4 and s4_valid else np.nan,
                "median_error_mm": _float_or_nan(metric.get("median_error_mm")) if is_s4 and s4_valid else np.nan,
                "p95_error_mm": _float_or_nan(metric.get("p95_error_mm")) if is_s4 and s4_valid else np.nan,
                "ate_rmse_mm": _float_or_nan(metric.get("ate_rmse_mm")) if is_s4 and s4_valid else np.nan,
                "rpe_rmse_mm": _float_or_nan(metric.get("rpe_rmse_mm")) if is_s4 and s4_valid else np.nan,
                "support_count": int(metric.get("support_count", 0)) if is_s4 and s4_valid else 0,
                "support_hash": persisted_support_hash if is_s4 and s4_valid else "",
                "calibration_provenance": partition_meta.get("calibration_provenance", ""),
                "evaluation_provenance": str(metrics_path) if is_s4 and s4_valid else "not_available",
                "temporal_provenance": str(base / "09_temporal" / "crosscorr_report.csv") if is_s4 and s4_valid else "not_available",
                "spatial_provenance": str(base / "07_spatial" / "spatial_diagnostics.csv") if is_s4 and s4_valid else "not_available",
                "representation_provenance": str(metric.get("representation_method", "")) if is_s4 else "not_available",
                "support_provenance": "canonical_metric_support_hash" if is_s4 and s4_valid else "not_available",
                "gate_status": gate_status or "missing",
                "gate_reason": gate_reason,
                "gate_eligible": gate_eligible,
                "gate_provenance_status": "verified" if gate_row is not None else "unavailable",
            }
            rows.append(row)
    out = pd.DataFrame(rows)
    if write:
        out_dir = base / "14_metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_dir / "cascade_per_trial_metrics.csv", index=False)
        pd.DataFrame(STAGES).to_csv(out_dir / "cascade_stage_registry.csv", index=False)
    return out
