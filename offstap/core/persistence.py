"""JSON-safe persistence helpers for alignment parameter artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from offstap.core.temporal import CorrelationResult


CLOCK_EXPORT_FIELDS = (
    "trial_id",
    "selected_model",
    "offset_b",
    "scale_a",
    "candidate_scale_a",
    "candidate_offset_b",
    "temporal_refinement_delta_s",
    "affine_improvement_ratio",
    "constant_mse",
    "affine_mse",
    "candidate_hit_bound",
    "reason",
)

FIGURE_SOURCE_FIELDS = (
    "figure_id",
    "panel_id",
    "extract_id",
    "source_artifact",
    "source_payload_sha256",
    "source_row_index",
    "source_row_identity",
    "source_column",
    "value_text",
    "value_type",
    "unit",
    "population",
    "eligibility",
    "support_hash",
    "provenance_json",
)

CLOCK_NON_NULL_FIELDS = (
    "trial_id",
    "selected_model",
    "offset_b",
    "scale_a",
    "temporal_refinement_delta_s",
    "reason",
)


def _json_safe(value: Any) -> Any:
    """Convert supported scientific values to strict JSON-native values."""
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    raise TypeError(f"Unsupported value for JSON persistence: {type(value).__name__}")


def correlation_result_to_json(result: CorrelationResult) -> dict[str, Any]:
    """Persist the frozen, machine-readable correlation diagnostics."""
    if not isinstance(result, CorrelationResult):
        raise TypeError("expected CorrelationResult")
    return {
        "best_lag_s": _json_safe(result.best_lag_s),
        "best_score": _json_safe(result.best_peak),
        "second_peak_score": _json_safe(result.second_peak),
        "peak_ratio": _json_safe(result.peak_ratio),
        "overlap_s": _json_safe(result.overlap_duration_s),
        "status": result.status,
        "rejection_reason": result.rejection_reason,
        "lags_s": _json_safe(result.lags_s),
        "scores": _json_safe(result.scores),
        "overlap_by_lag_s": _json_safe(result.overlap_s),
    }


def temporal_params_to_json(params: Mapping[str, Any]) -> dict[str, Any]:
    """Convert temporal-fit output while retaining scalar compatibility fields."""
    persisted: dict[str, Any] = {}
    for key, value in params.items():
        if key in {"valid_lags", "valid_corr"}:
            continue
        if key == "correlation_result":
            persisted[key] = correlation_result_to_json(value)
        else:
            persisted[str(key)] = _json_safe(value)
    return persisted


def _csv_text(value: Any) -> str:
    """Return deterministic CSV text with contract null semantics."""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return ""
    if isinstance(value, np.bool_):
        return str(bool(value))
    return str(value)


def persist_fitted_parameters(rows: Any, output_path: str | Path) -> Path:
    """Persist the already-computed Stage 05 clock log without refitting."""
    records = list(rows)
    if not records:
        raise ValueError("clock export schema requires at least one computed row")
    missing = [field for field in CLOCK_EXPORT_FIELDS if field not in records[0]]
    if missing:
        raise ValueError(f"clock export schema missing fields: {missing}")

    normalized = []
    seen_trials: set[str] = set()
    for record in records:
        missing = [field for field in CLOCK_EXPORT_FIELDS if field not in record]
        if missing:
            raise ValueError(f"clock export schema missing fields: {missing}")
        trial_id = _csv_text(record["trial_id"])
        if not trial_id:
            raise ValueError("clock export schema requires a non-empty trial_id")
        for field in CLOCK_NON_NULL_FIELDS:
            if not _csv_text(record[field]):
                raise ValueError(f"clock export schema requires non-null {field}")
        if trial_id in seen_trials:
            raise ValueError(f"clock export schema duplicate trial_id: {trial_id}")
        seen_trials.add(trial_id)
        normalized.append({field: _csv_text(record[field]) for field in CLOCK_EXPORT_FIELDS})

    normalized.sort(key=lambda record: record["trial_id"])
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CLOCK_EXPORT_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(normalized)
    return target


def _normalise_provenance_value(value: str, run_dir: Path) -> str:
    text = value or ""
    run_text = str(run_dir)
    candidates = {run_text, run_text.replace("\\", "/"), run_text.replace("/", "\\")}
    for candidate in sorted(candidates, key=len, reverse=True):
        text = text.replace(candidate, "<RUN_ROOT>")
    return text


def _source_cell_text(value: Any, column: str, run_dir: Path) -> str:
    text = "" if value is None else str(value)
    if "provenance" in column.lower():
        return _normalise_provenance_value(text, run_dir)
    return text


def _value_type(value_text: str) -> str:
    if value_text == "":
        return "null"
    if value_text.lower() in {"true", "false"}:
        return "boolean"
    try:
        float(value_text)
    except ValueError:
        return "string"
    return "numeric"


def _canonical_extract_payload(fieldnames: list[str], rows: list[dict[str, str]], run_dir: Path) -> tuple[str, list[dict[str, str]]]:
    normalized_rows = [
        {field: _source_cell_text(row.get(field), field, run_dir) for field in fieldnames}
        for row in rows
    ]
    payload = {"columns": fieldnames, "rows": normalized_rows}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), normalized_rows


def persist_figure_source_table(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    """Persist a lossless, normalized index of the generated figure extracts."""
    run_root = Path(run_dir).resolve()
    paper_dir = run_root / "paper_figures"
    manifest_path = paper_dir / "figure_manifest.csv"
    extracts_dir = paper_dir / "figure_data_extracts"
    if not manifest_path.exists():
        raise FileNotFoundError(f"figure manifest missing: {manifest_path}")
    if not extracts_dir.is_dir():
        raise FileNotFoundError(f"figure extracts directory missing: {extracts_dir}")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_reader = csv.DictReader(handle)
        required_manifest = {"figure_id", "source_files", "analysis_set", "valid_for_paper"}
        if not manifest_reader.fieldnames or not required_manifest.issubset(manifest_reader.fieldnames):
            raise ValueError("figure manifest schema is missing required fields")
        manifest_rows = list(manifest_reader)
    if not manifest_rows:
        raise ValueError("figure manifest schema requires at least one figure")

    output_rows: list[dict[str, str]] = []
    seen_figures: set[str] = set()
    seen_identities: set[str] = set()
    for manifest in manifest_rows:
        figure_id = manifest["figure_id"]
        if not figure_id or figure_id in seen_figures:
            raise ValueError(f"figure manifest duplicate or empty figure_id: {figure_id}")
        seen_figures.add(figure_id)
        source_files = [item for item in manifest["source_files"].split(";") if item]
        if not source_files:
            raise ValueError(f"figure manifest has no source artifact: {figure_id}")
        for source_artifact in source_files:
            source_path = Path(source_artifact)
            if source_path.is_absolute() or ".." in source_path.parts:
                raise ValueError(f"figure source path is not relative: {source_artifact}")
            extract_path = extracts_dir / f"{figure_id}_{source_path.name}"
            if not extract_path.exists():
                raise FileNotFoundError(f"figure extract missing: {extract_path}")
            with extract_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or any(field is None for field in reader.fieldnames):
                    raise ValueError(f"figure extract schema malformed: {extract_path}")
                fieldnames = list(reader.fieldnames)
                if len(set(fieldnames)) != len(fieldnames):
                    raise ValueError(f"figure extract schema has duplicate columns: {extract_path}")
                source_rows = list(reader)
            if not source_rows:
                raise ValueError(f"figure extract is empty: {extract_path}")
            if any(any(value is None for value in row.values()) for row in source_rows):
                raise ValueError(f"figure extract schema has malformed rows: {extract_path}")
            payload_hash, normalized_rows = _canonical_extract_payload(fieldnames, source_rows, run_root)
            extract_id = extract_path.stem
            for row_index, normalized_row in enumerate(normalized_rows):
                row_identity = hashlib.sha256(
                    json.dumps(
                        {"extract_id": extract_id, "row_index": row_index, "row": normalized_row},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                if row_identity in seen_identities:
                    raise ValueError(f"duplicate figure source row identity: {row_identity}")
                seen_identities.add(row_identity)
                support_hash = normalized_row.get("support_hash", "")
                provenance = {
                    "figure_id": figure_id,
                    "extract_id": extract_id,
                    "source_artifact": source_artifact,
                    "source_payload_sha256": payload_hash,
                    "source_row_index": row_index,
                    "manifest": {
                        key: _source_cell_text(value, key, run_root)
                        for key, value in manifest.items()
                    },
                }
                for column in fieldnames:
                    output_rows.append({
                        "figure_id": figure_id,
                        "panel_id": "",
                        "extract_id": extract_id,
                        "source_artifact": source_artifact,
                        "source_payload_sha256": payload_hash,
                        "source_row_index": str(row_index),
                        "source_row_identity": row_identity,
                        "source_column": column,
                        "value_text": normalized_row[column],
                        "value_type": _value_type(normalized_row[column]),
                        "unit": manifest.get("units", ""),
                        "population": manifest["analysis_set"],
                        "eligibility": manifest.get("valid_for_paper", ""),
                        "support_hash": support_hash,
                        "provenance_json": json.dumps(provenance, sort_keys=True, separators=(",", ":")),
                    })

    target = Path(output_path) if output_path is not None else paper_dir / "figure_source_table.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIGURE_SOURCE_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    return target


__all__ = [
    "CLOCK_EXPORT_FIELDS",
    "FIGURE_SOURCE_FIELDS",
    "correlation_result_to_json",
    "temporal_params_to_json",
    "persist_fitted_parameters",
    "persist_figure_source_table",
]
