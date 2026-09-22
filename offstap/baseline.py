"""Current-run engineering-baseline adapter.

The numerical contract is delegated to the curated comparator primitives.  This
module supplies current-run artifact validation, the approved composite arm,
reason-coded trial exclusions, summaries, and release-neutral provenance.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from offstap.core.alignment import generate_aligned_trajectory
from offstap.core.baseline_comparator import (
    ArmProvenance,
    PairedScore,
    _finite_evaluation,
    canonical_offstap_metric_matches,
    score_paired_frames,
    validate_arm_provenance,
)
from offstap.core.config import load_config
from offstap.core.spatial import _estimate_rigid_calibration_transform
from offstap.core.support import validate_arm_support
from offstap.core.trajectory_reference import ContinuousTrajectory


BASELINE_PROVENANCE = ArmProvenance(
    "baseline", "full_trajectory_rigid_se2", "onset_only", "constant_offset", "zoh"
)
OFFSTAP_PROVENANCE = ArmProvenance(
    "offstap",
    "calibration_evaluation_split",
    "crosscorr_refined",
    "affine_or_constant_fallback",
    "natural_cubic",
)

REQUIRED_TABLES = {
    "01_manifest/pair_manifest.csv": ("trial_id", "dataset_tier"),
    "09_temporal/crosscorr_report.csv": ("trial_id", "b_onset", "b_final"),
    "10_clock_model/fitted_parameters.csv": ("trial_id", "selected_model", "scale_a"),
    "07_spatial/spatial_calibration_parameters.csv": ("trial_id", "theta_rad", "tx", "ty"),
    "14_metrics/evaluation_metrics.csv": ("trial_id", "ate_rmse", "num_samples"),
}

TRIAL_COLUMNS = (
    "trial_id",
    "dataset_tier",
    "baseline_rmse_mm",
    "offstap_rmse_mm",
    "delta_rmse_mm",
    "common_support_count",
    "common_support_hash",
    "baseline_temporal_offset_s",
    "offstap_temporal_offset_s",
    "baseline_clock_model",
    "baseline_clock_scale_a",
    "offstap_clock_model",
    "offstap_clock_scale_a",
    "baseline_spatial_policy",
    "baseline_spatial_theta_rad",
    "baseline_spatial_tx",
    "baseline_spatial_ty",
    "offstap_spatial_policy",
    "offstap_spatial_theta_rad",
    "offstap_spatial_tx",
    "offstap_spatial_ty",
    "baseline_representation",
    "offstap_representation",
    "canonical_offstap_rmse_mm",
)

REJECTION_COLUMNS = (
    "trial_id",
    "dataset_tier",
    "reason",
    "baseline_common_support_count",
    "canonical_offstap_support_count",
)


def sha256(path: Path) -> str:
    """Hash one persisted input or output artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value):
    """Return a JSON-safe finite float or ``None``."""
    return float(value) if value is not None and np.isfinite(value) else None


def source_value(frame: pd.DataFrame, trial_id: str, column: str):
    """Select exactly one persisted source row for a trial."""
    rows = frame.loc[frame["trial_id"].astype(str).eq(str(trial_id))]
    if len(rows) != 1:
        raise ValueError(f"{trial_id}: expected exactly one persisted {column} row")
    return rows.iloc[0]


def _canonical_frame_path(source_run: Path, trial_id: str) -> Path:
    matches = sorted(
        (source_run / "13_integrated_logs" / "ct_reference").glob(f"{trial_id}_*.parquet")
    )
    if len(matches) != 1:
        raise ValueError(
            f"{trial_id}: expected exactly one canonical OFF-STAP frame, found {len(matches)}"
        )
    return matches[0]


def canonical_frame(source_run: Path, trial_id: str) -> pd.DataFrame:
    """Load the unique current-run canonical OFF-STAP frame."""
    return pd.read_parquet(_canonical_frame_path(Path(source_run), str(trial_id)))


def _trial_input_paths(source_run: Path, trial_id: str) -> tuple[Path, ...]:
    paths = (
        source_run / "03_cleaned" / "ev3" / f"{trial_id}_ev3_cleaned.parquet",
        source_run / "03_cleaned" / "vrmt" / f"{trial_id}_vrmt_cleaned.parquet",
        source_run / "11_continuous_time_reference" / "models" / f"{trial_id}_spline.pkl",
        _canonical_frame_path(source_run, trial_id),
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        relative = ", ".join(path.relative_to(source_run).as_posix() for path in missing)
        raise FileNotFoundError(f"{trial_id}: required current-run artifact missing: {relative}")
    return paths


def baseline_frame(
    source_run: Path,
    trial_id: str,
    offstap: pd.DataFrame,
    b_onset: float,
    cfg,
) -> tuple[pd.DataFrame, dict]:
    """Construct the approved engineering-comparator arm on current-run data."""
    ev3 = pd.read_parquet(
        source_run / "03_cleaned" / "ev3" / f"{trial_id}_ev3_cleaned.parquet"
    )
    vive = pd.read_parquet(
        source_run / "03_cleaned" / "vrmt" / f"{trial_id}_vrmt_cleaned.parquet"
    )
    spline_path = (
        source_run / "11_continuous_time_reference" / "models" / f"{trial_id}_spline.pkl"
    )
    with spline_path.open("rb") as stream:
        spline_data = pickle.load(stream)
    if not spline_data.get("valid", False):
        raise ValueError(f"{trial_id}: spline reference is invalid")
    spline = ContinuousTrajectory(spline_data["t"], spline_data["x"], spline_data["y"])

    cfg_baseline = deepcopy(cfg)
    cfg_baseline.config_id = f"{cfg.config_id}_baseline"
    cfg_baseline.preprocessing = {"resampling_method": "zoh"}
    cfg_baseline.geometry["spatial_fit_method"] = "calibration_partition_rigid_se2"
    cfg_baseline.geometry["planar_projection_mode"] = "direct"
    temporal = {
        "b_onset": float(b_onset),
        "b_final": float(b_onset),
        "method_used": "onset_only",
    }
    clock = {
        "selected_model": "constant_offset",
        "offset_b": float(b_onset),
        "scale_a": 1.0,
    }
    spatial = dict(
        _estimate_rigid_calibration_transform(ev3, vive, cfg_baseline, temporal, clock)
    )
    if not np.isfinite(spatial.get("trajectory_residual", np.nan)):
        raise ValueError(f"{trial_id}: full-trajectory baseline spatial fit is invalid")
    spatial["method_used"] = "full_trajectory_rigid_se2"

    source_rows = offstap.loc[
        :,
        [
            column
            for column in ("source_row_index", "source_timestamp_ns", "partition")
            if column in offstap.columns
        ],
    ].copy()
    aligned = generate_aligned_trajectory(
        ev3,
        vive,
        spline,
        temporal,
        clock,
        spatial,
        cfg_baseline,
        source_rows=source_rows,
    )
    if aligned.empty:
        raise ValueError(f"{trial_id}: baseline alignment produced no finite overlap")
    return aligned, spatial


def canonical_support_complete(score: PairedScore, expected_count: int) -> bool:
    """Require the paired support to cover every canonical OFF-STAP sample."""
    return score.common_support_count == int(expected_count)


def _canonical_evaluation_support_count(offstap: pd.DataFrame) -> int:
    support = validate_arm_support(
        _finite_evaluation(offstap),
        ("source_row_index",),
    )
    if not support.valid:
        raise ValueError(f"invalid canonical OFF-STAP support: {support.status}")
    return support.unique_count


def classify_paired_score(
    trial_id: str,
    score: PairedScore,
    expected_count: int,
) -> tuple[dict | None, dict | None]:
    """Classify a score without ever substituting incomplete support."""
    if not canonical_support_complete(score, expected_count):
        return None, {
            "trial_id": trial_id,
            "reason": "baseline_common_support_does_not_cover_canonical_offstap_support",
            "baseline_common_support_count": score.common_support_count,
            "canonical_offstap_support_count": int(expected_count),
        }
    score_metrics = (
        score.baseline_rmse_mm,
        score.offstap_rmse_mm,
        score.delta_rmse_mm,
    )
    if not np.isfinite(np.asarray(score_metrics, dtype=float)).all():
        return None, {
            "trial_id": trial_id,
            "reason": "paired_score_metrics_non_finite",
            "baseline_common_support_count": score.common_support_count,
            "canonical_offstap_support_count": int(expected_count),
        }
    return {
        "trial_id": trial_id,
        "baseline_rmse_mm": score.baseline_rmse_mm,
        "offstap_rmse_mm": score.offstap_rmse_mm,
        "delta_rmse_mm": score.delta_rmse_mm,
        "common_support_count": score.common_support_count,
        "common_support_hash": score.common_support_hash,
    }, None


def describe(values: pd.Series) -> dict:
    """Describe paired deltas while representing an empty population explicitly."""
    data = values.to_numpy(float)
    absolute = np.abs(data)
    thresholds = (0.0, 1e-12, 1e-9, 1e-6, 0.001, 0.01, 0.1)
    if len(data) == 0:
        return {
            "paired_n": 0,
            "exact_equality_count": 0,
            "nonzero_count": 0,
            "threshold_counts": {str(threshold): 0 for threshold in thresholds},
            "mean_delta_mm": None,
            "median_delta_mm": None,
            "sd_delta_mm": None,
            "iqr_delta_mm": None,
            "p01_delta_mm": None,
            "p05_delta_mm": None,
            "p25_delta_mm": None,
            "p75_delta_mm": None,
            "p95_delta_mm": None,
            "p99_delta_mm": None,
            "min_delta_mm": None,
            "max_delta_mm": None,
            "max_abs_delta_mm": None,
        }
    return {
        "paired_n": int(len(data)),
        "exact_equality_count": int((data == 0.0).sum()),
        "nonzero_count": int((data != 0.0).sum()),
        "threshold_counts": {
            str(threshold): int((absolute > threshold).sum()) for threshold in thresholds
        },
        "mean_delta_mm": finite(np.mean(data)),
        "median_delta_mm": finite(np.median(data)),
        "sd_delta_mm": finite(np.std(data, ddof=1)) if len(data) > 1 else None,
        "iqr_delta_mm": finite(np.percentile(data, 75) - np.percentile(data, 25)),
        "p01_delta_mm": finite(np.percentile(data, 1)),
        "p05_delta_mm": finite(np.percentile(data, 5)),
        "p25_delta_mm": finite(np.percentile(data, 25)),
        "p75_delta_mm": finite(np.percentile(data, 75)),
        "p95_delta_mm": finite(np.percentile(data, 95)),
        "p99_delta_mm": finite(np.percentile(data, 99)),
        "min_delta_mm": finite(np.min(data)),
        "max_delta_mm": finite(np.max(data)),
        "max_abs_delta_mm": finite(np.max(absolute)),
    }


def comparator_statistic(group: pd.DataFrame) -> dict:
    """Apply the approved paired-test decision rule to one population."""
    baseline = group["baseline_rmse_mm"].to_numpy(float)
    offstap = group["offstap_rmse_mm"].to_numpy(float)
    delta = group["delta_rmse_mm"].to_numpy(float)
    normality_stat, normality_p = (
        stats.shapiro(delta) if len(delta) >= 3 else (np.nan, np.nan)
    )
    if len(delta) < 2:
        name, statistic, p_value = "not_computable", np.nan, np.nan
    elif np.isfinite(normality_p) and normality_p < 0.05:
        name = "wilcoxon"
        statistic, p_value = stats.wilcoxon(baseline, offstap)
    else:
        name = "ttest_rel"
        statistic, p_value = stats.ttest_rel(baseline, offstap)
    return {
        "normality_test_name": "shapiro",
        "normality_statistic": finite(normality_stat),
        "normality_p_value": finite(normality_p),
        "paired_test_name": name,
        "paired_test_statistic": finite(statistic),
        "paired_test_p_value": finite(p_value),
    }


def _read_required_table(source_run: Path, relative: str, required: tuple[str, ...]) -> pd.DataFrame:
    path = source_run / relative
    if not path.is_file():
        raise FileNotFoundError(f"required current-run artifact missing: {relative}")
    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{relative}: missing required columns: {missing}")
    return frame


def _validate_unique_trials(frame: pd.DataFrame, label: str) -> None:
    duplicates = frame.loc[frame["trial_id"].astype(str).duplicated(keep=False), "trial_id"]
    if not duplicates.empty:
        raise ValueError(f"{label}: duplicate trial_id rows: {sorted(set(duplicates.astype(str)))}")


def _population_summaries(trial_table: pd.DataFrame) -> list[dict]:
    summaries = []
    populations = (
        ("planned_population", {"planned_405"}),
        ("extended_only_population", {"extended_previous_paper"}),
        (
            "combined_compatible_population",
            {"planned_405", "extended_previous_paper"},
        ),
    )
    for label, tiers in populations:
        group = trial_table.loc[trial_table["dataset_tier"].isin(tiers)].copy()
        row = {"population": label, **describe(group["delta_rmse_mm"])}
        row.update({
            "mean_baseline_rmse_mm": finite(group["baseline_rmse_mm"].mean()),
            "median_baseline_rmse_mm": finite(group["baseline_rmse_mm"].median()),
            "mean_offstap_rmse_mm": finite(group["offstap_rmse_mm"].mean()),
            "median_offstap_rmse_mm": finite(group["offstap_rmse_mm"].median()),
        })
        if label != "extended_only_population":
            row.update(comparator_statistic(group))
        summaries.append(row)
    return summaries


def _refuse_nonempty(output_dir: Path) -> None:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"refusing nonempty baseline destination: {output_dir}")


def run_baseline(source_run: Path, output_dir: Path) -> dict[str, Path]:
    """Compute the corrected composite baseline from one newly generated run."""
    source_run = Path(source_run).resolve()
    output_dir = Path(output_dir).resolve()
    _refuse_nonempty(output_dir)
    if not source_run.is_dir():
        raise FileNotFoundError(f"current run does not exist: {source_run}")

    config_path = source_run / "config_used.yaml"
    if not config_path.is_file():
        raise FileNotFoundError("required current-run artifact missing: config_used.yaml")
    cfg = load_config(str(config_path))
    tables = {
        relative: _read_required_table(source_run, relative, required)
        for relative, required in REQUIRED_TABLES.items()
    }
    pairs = tables["01_manifest/pair_manifest.csv"]
    temporal = tables["09_temporal/crosscorr_report.csv"]
    clock = tables["10_clock_model/fitted_parameters.csv"]
    spatial = tables["07_spatial/spatial_calibration_parameters.csv"]
    metrics = tables["14_metrics/evaluation_metrics.csv"]
    if metrics.empty:
        raise ValueError("metric artifact contains no canonical trials")
    _validate_unique_trials(pairs, "manifest")
    _validate_unique_trials(metrics, "metrics")
    validate_arm_provenance(BASELINE_PROVENANCE, OFFSTAP_PROVENANCE)

    input_paths = [config_path] + [source_run / relative for relative in REQUIRED_TABLES]
    rows: list[dict] = []
    rejections: list[dict] = []
    for metric in metrics.itertuples(index=False):
        trial_id = str(metric.trial_id)
        dataset_tier = "unknown"
        try:
            pair_row = source_value(pairs, trial_id, "manifest")
            dataset_tier = str(pair_row.dataset_tier)
            trial_paths = _trial_input_paths(source_run, trial_id)
            input_paths.extend(trial_paths)
            offstap = pd.read_parquet(trial_paths[-1])
            canonical_mm = float(metric.ate_rmse) * 1000.0
            if not canonical_offstap_metric_matches(offstap, canonical_mm):
                raise ValueError("canonical_offstap_metric_mismatch")
            declared_count = int(metric.num_samples)
            actual_count = _canonical_evaluation_support_count(offstap)
            if declared_count != actual_count:
                rejections.append({
                    "trial_id": trial_id,
                    "dataset_tier": dataset_tier,
                    "reason": "canonical_offstap_support_count_mismatch",
                    "canonical_offstap_support_count": actual_count,
                })
                continue
            if actual_count <= 0:
                raise ValueError("canonical_offstap_support_is_empty")
            time_row = source_value(temporal, trial_id, "temporal")
            clock_row = source_value(clock, trial_id, "clock")
            spatial_row = source_value(spatial, trial_id, "spatial")
            baseline, full_spatial = baseline_frame(
                source_run, trial_id, offstap, float(time_row.b_onset), cfg
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Mean of empty slice", category=RuntimeWarning
                )
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in scalar divide",
                    category=RuntimeWarning,
                )
                score = score_paired_frames(baseline, offstap)
            result, rejection = classify_paired_score(trial_id, score, actual_count)
            if rejection is not None:
                rejections.append({"dataset_tier": dataset_tier, **rejection})
                continue
            rows.append({
                **result,
                "dataset_tier": dataset_tier,
                "baseline_temporal_offset_s": float(time_row.b_onset),
                "offstap_temporal_offset_s": float(time_row.b_final),
                "baseline_clock_model": "constant_offset",
                "baseline_clock_scale_a": 1.0,
                "offstap_clock_model": str(clock_row.selected_model),
                "offstap_clock_scale_a": float(clock_row.scale_a),
                "baseline_spatial_policy": BASELINE_PROVENANCE.spatial_policy,
                "baseline_spatial_theta_rad": finite(full_spatial.get("theta_rad")),
                "baseline_spatial_tx": finite(full_spatial.get("tx")),
                "baseline_spatial_ty": finite(full_spatial.get("ty")),
                "offstap_spatial_policy": OFFSTAP_PROVENANCE.spatial_policy,
                "offstap_spatial_theta_rad": finite(spatial_row.theta_rad),
                "offstap_spatial_tx": finite(spatial_row.tx),
                "offstap_spatial_ty": finite(spatial_row.ty),
                "baseline_representation": "zoh",
                "offstap_representation": "natural_cubic",
                "canonical_offstap_rmse_mm": canonical_mm,
            })
        except Exception as exc:
            rejections.append({
                "trial_id": trial_id,
                "dataset_tier": dataset_tier,
                "reason": str(exc),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    trial_table = pd.DataFrame(rows, columns=TRIAL_COLUMNS).sort_values("trial_id")
    trial_path = output_dir / "baseline_trial_metrics.csv"
    trial_table.to_csv(trial_path, index=False)
    rejection_table = pd.DataFrame(rejections, columns=REJECTION_COLUMNS).sort_values("trial_id")
    rejection_path = output_dir / "baseline_rejections.csv"
    rejection_table.to_csv(rejection_path, index=False)
    summary_path = output_dir / "baseline_summary.json"
    summary_path.write_text(
        json.dumps(_population_summaries(trial_table), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    output_paths = (trial_path, rejection_path, summary_path)
    unique_inputs = sorted(set(input_paths), key=lambda path: path.relative_to(source_run).as_posix())
    manifest = {
        "identity": "baseline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": ".",
        "contract": {
            "baseline": BASELINE_PROVENANCE.__dict__,
            "offstap": OFFSTAP_PROVENANCE.__dict__,
            "support": "exact finite evaluation source_row_index intersection",
        },
        "input_sha256": {
            path.relative_to(source_run).as_posix(): sha256(path) for path in unique_inputs
        },
        "output_sha256": {path.name: sha256(path) for path in output_paths},
        "validation": {
            "canonical_offstap_metric_mismatches": int(
                sum(row.get("reason") == "canonical_offstap_metric_mismatch" for row in rejections)
            ),
            "common_support_mismatch_exclusions": int(
                sum(
                    row.get("reason")
                    == "baseline_common_support_does_not_cover_canonical_offstap_support"
                    for row in rejections
                )
            ),
            "other_reason_coded_exclusions": int(
                sum(
                    row.get("reason")
                    not in {
                        "canonical_offstap_metric_mismatch",
                        "baseline_common_support_does_not_cover_canonical_offstap_support",
                    }
                    for row in rejections
                )
            ),
        },
    }
    manifest_path = output_dir / "baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "trial_metrics": trial_path,
        "rejections": rejection_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


__all__ = [
    "baseline_frame",
    "canonical_frame",
    "canonical_support_complete",
    "classify_paired_score",
    "run_baseline",
]
