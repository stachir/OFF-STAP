"""Machine-readable contracts for Stage 10 figure inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


class PlotSchemaError(ValueError):
    """Raised when a required plotting input cannot support its figure."""


H2_PLOT_APPLICABILITY_CONTRACT = {
    "h2_method_boxplot": {
        "required_inputs": ["method", "rmse_mm"],
        "mode_behavior": "required_full_dev",
    },
    "h2_method_violin_or_histogram": {
        "required_inputs": ["method", "rmse_mm"],
        "mode_behavior": "required_full_dev",
    },
    "h2_ct_vs_discrete_scatter": {
        "required_inputs": ["method", "trial_id", "rmse_mm"],
        "mode_behavior": "record_skip_no_png",
    },
    "h2_error_vs_timestamp_irregularity": {
        "required_inputs": ["method", "rmse_mm", "timestamp_irregularity"],
        "mode_behavior": "record_skip_no_png",
    },
    "h2_phase_specific_method_comparison": {
        "required_inputs": ["method", "rmse_mm", "phase"],
        "mode_behavior": "record_skip_no_png",
    },
}


PLOT_SCHEMA = [
    {
        "figure_id": "h1_onset_vs_crosscorr_scatter",
        "source_table": "15_hypotheses/H1/h1_trial_pairs.csv",
        "required_columns": ["trial_id", "valid_for_paper", "rmse_onset_mm", "rmse_refined_mm", "delta_rmse_mm", "common_support_count", "common_support_hash"],
        "optional_columns": ["status", "reason", "tier", "common_support_hash_sha256", "onset_support_count", "refined_support_count"],
        "population": "accepted trials",
        "comparison": "H1 onset versus refined timing",
        "unit": "mm",
        "empty_data_behavior": "skip_if_no_eligible_rows_dev",
    },
    {
        "figure_id": "h2_method_rmse",
        "source_table": "15_hypotheses/H2/h2_trial_comparison.csv",
        "required_columns": ["method", "rmse_mm"],
        "optional_columns": ["trial_id", "status", "common_support_hash_sha256"],
        "population": "accepted trials with common support",
        "comparison": "Vive representation/query operator",
        "unit": "mm",
        "empty_data_behavior": "skip_if_no_eligible_rows_dev",
    },
    {
        "figure_id": "h3_calibration_vs_evaluation_residuals",
        "source_table": "15_hypotheses/H3/h3_calibration_mode_comparison.csv",
        "required_columns": ["calibration_mode", "heldout_rmse_mm"],
        "optional_columns": ["trial_id", "status", "valid_for_paper", "common_support_count"],
        "population": "eligible H3 evaluation comparisons",
        "comparison": "split-fit versus intentionally leaky full-fit",
        "unit": "mm",
        "empty_data_behavior": "skip_if_no_eligible_rows_dev",
    },
    {
        "figure_id": "primary_heldout_rmse_distribution",
        "source_table": "14_metrics/per_trial_metrics.csv",
        "required_columns": ["ate_rmse_mm"],
        "optional_columns": ["trial_id", "metric_status", "support_count"],
        "population": "held-out evaluation rows",
        "comparison": "primary alignment residual",
        "unit": "mm",
        "empty_data_behavior": "skip_if_no_eligible_rows_dev",
    },
]

# The manifest covers every Stage 10 output, including figures whose inputs
# are optional diagnostics or trajectory examples.  Their plotting code keeps
# the existing display-only empty behavior; validation of hypothesis source
# tables above is stricter because those figures encode scientific comparisons.
for _figure_id, _source, _columns, _population, _comparison, _unit in [
    ("sampling_rate_distribution_ev3", "04_quality/stream_diagnostics.csv", ["ev3_eff_rate"], "quality-gated streams", "EV3 sampling rate", "Hz"),
    ("sampling_rate_distribution_vive", "04_quality/stream_diagnostics.csv", ["vrmt_eff_rate"], "quality-gated streams", "Vive sampling rate", "Hz"),
    ("timestamp_irregularity_distribution_ev3", "04_quality/stream_diagnostics.csv", ["ev3_irregularity"], "quality-gated streams", "EV3 timestamp irregularity", "s"),
    ("timestamp_irregularity_distribution_vive", "04_quality/stream_diagnostics.csv", ["vrmt_irregularity"], "quality-gated streams", "Vive timestamp irregularity", "s"),
    ("rejection_reasons", "04_quality/rejection_log.csv", ["reason"], "rejected streams", "quality rejection reasons", "count"),
    ("trajectory_after_registration", "13_integrated_logs/ct_reference/*.parquet", ["odometry_x", "odometry_y", "vrmt_aligned_x", "vrmt_aligned_y", "t_common_s", "residual_norm"], "representative aligned trial", "trajectory registration", "m"),
    ("trajectory_before_registration", "13_integrated_logs/ct_reference/*.parquet", ["odometry_x", "odometry_y"], "representative aligned trial", "pre-registration trajectory", "m"),
    ("ct_reference_fit", "13_integrated_logs/ct_reference/*.parquet", ["t_common_s", "vrmt_aligned_x"], "representative aligned trial", "continuous reference fit", "m"),
    ("h1_delta_residual_boxplot", "15_hypotheses/H1/h1_trial_pairs.csv", ["delta_rmse_mm"], "eligible H1 trials", "H1 paired residual difference", "mm"),
    ("h1_peak_confidence_histogram", "15_hypotheses/H1/h1_trial_pairs.csv", ["crosscorr_peak_confidence_ratio"], "eligible H1 trials", "H1 peak confidence", "ratio"),
    ("h1_ambiguous_cases_barplot", "15_hypotheses/H1/h1_trial_pairs.csv", ["crosscorr_accepted"], "H1 trials", "H1 acceptance status", "count"),
    ("h2_method_boxplot", "15_hypotheses/H2/h2_trial_comparison.csv", ["method", "rmse_mm"], "eligible H2 trials", "Vive representation/query operator", "mm"),
    ("h2_method_violin_or_histogram", "15_hypotheses/H2/h2_trial_comparison.csv", ["method", "rmse_mm"], "eligible H2 trials", "Vive representation/query operator", "mm"),
    ("h2_ct_vs_discrete_scatter", "15_hypotheses/H2/h2_trial_comparison.csv", ["method", "rmse_mm", "trial_id"], "eligible H2 trials", "H2 paired method comparison", "mm"),
    ("h2_error_vs_timestamp_irregularity", "15_hypotheses/H2/h2_trial_comparison.csv", ["method", "rmse_mm", "timestamp_irregularity"], "eligible H2 trials", "H2 error versus timestamp irregularity", "mm"),
    ("h2_phase_specific_method_comparison", "15_hypotheses/H2/h2_trial_comparison.csv", ["method", "rmse_mm", "phase"], "eligible H2 trials", "H2 phase comparison", "mm"),
    ("h3_split_vs_full_scatter", "15_hypotheses/H3/h3_calibration_mode_comparison.csv", ["trial_id", "calibration_mode", "heldout_rmse_mm"], "eligible H3 trials", "H3 split versus full fit", "mm"),
    ("h3_residual_difference_histogram", "15_hypotheses/H3/h3_calibration_mode_comparison.csv", ["trial_id", "calibration_mode", "heldout_rmse_mm"], "eligible H3 trials", "H3 paired residual difference", "mm"),
    ("h3_example_trial_full_fit_vs_split_fit", "15_hypotheses/H3/h3_calibration_mode_comparison.csv", ["calibration_mode", "heldout_rmse_mm"], "eligible H3 trials", "H3 example comparison", "mm"),
    ("residual_by_configuration", "14_metrics/per_trial_metrics.csv", ["rmse_mm"], "held-out evaluation rows", "configuration residual", "mm"),
    ("ate_by_configuration", "14_metrics/per_trial_metrics.csv", ["ate_rmse_mm"], "held-out evaluation rows", "ATE residual", "mm"),
    ("rpe_by_configuration", "14_metrics/per_trial_metrics.csv", ["rpe_rmse_mm"], "held-out evaluation rows", "RPE residual", "mm"),
    ("ate_rmse_boxplot", "14_metrics/evaluation_metrics.csv", ["ate_rmse"], "held-out evaluation rows", "ATE residual", "m"),
    ("rpe_rmse_boxplot", "14_metrics/evaluation_metrics.csv", ["rpe_rmse"], "held-out evaluation rows", "RPE residual", "m"),
]:
    PLOT_SCHEMA.append({
        "figure_id": _figure_id,
        "source_table": _source,
        "required_columns": _columns,
        "optional_columns": ["trial_id", "status"],
        "population": _population,
        "comparison": _comparison,
        "unit": _unit,
        "empty_data_behavior": "skip_if_no_eligible_rows_dev",
    })

for _specification in PLOT_SCHEMA:
    contract = H2_PLOT_APPLICABILITY_CONTRACT.get(_specification["figure_id"])
    if contract:
        _specification["applicability_states"] = ["APPLICABLE", "NOT_APPLICABLE", "INVALID_SOURCE"]
        _specification["applicability_required_inputs"] = contract["required_inputs"]
        _specification["applicability_mode_behavior"] = contract["mode_behavior"]

# H1 figures share the producer's persisted eligibility and support
# provenance.  Their plotted quantities differ, but none may silently accept
# an un-auditable row or reconstruct a historical arm-residual alias.
_h1_required = {
    "h1_delta_residual_boxplot": [
        "trial_id", "valid_for_paper", "delta_rmse_mm",
        "common_support_count", "common_support_hash",
    ],
    "h1_peak_confidence_histogram": [
        "trial_id", "valid_for_paper", "crosscorr_peak_confidence_ratio",
        "common_support_count", "common_support_hash",
    ],
    "h1_ambiguous_cases_barplot": [
        "trial_id", "valid_for_paper", "crosscorr_accepted",
        "common_support_count", "common_support_hash",
    ],
}
for _specification in PLOT_SCHEMA:
    if _specification["figure_id"] in _h1_required:
        _specification["required_columns"] = _h1_required[_specification["figure_id"]]
        _specification["optional_columns"] = [
            "status", "reason", "tier", "common_support_hash_sha256",
        ]


def _spec(figure_id: str) -> Mapping[str, Any]:
    for spec in PLOT_SCHEMA:
        if spec["figure_id"] == figure_id:
            return spec
    raise KeyError(f"Unknown Stage 10 figure schema: {figure_id}")


def _eligible_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame
    if "valid_for_paper" in result.columns:
        values = result["valid_for_paper"].astype(str).str.lower().isin({"true", "1", "yes", "valid"})
        result = result[values]
    elif "status" in result.columns:
        result = result[result["status"].astype(str).str.lower().isin({"ok", "valid", "success", "accepted"})]
    return result


def validate_plot_input(figure_id: str, frame: pd.DataFrame, *, dev_mode: bool = False) -> dict[str, Any]:
    """Validate one figure's frozen source table before plotting.

    DEV smoke runs may explicitly skip a figure when no eligible rows exist;
    complete runs fail early with a precise schema error.
    """

    spec = _spec(figure_id)
    missing = [column for column in spec["required_columns"] if column not in frame.columns]
    eligible = _eligible_rows(frame)
    if (not eligible.empty) and missing:
        raise PlotSchemaError(f"{figure_id}: missing required columns {missing} in {spec['source_table']}")
    if missing or eligible.empty:
        if dev_mode and spec["empty_data_behavior"] == "skip_if_no_eligible_rows_dev":
            return {"status": "SKIPPED", "reason": "no_eligible_rows", "figure_id": figure_id}
        raise PlotSchemaError(f"{figure_id}: no eligible rows for required schema {spec['required_columns']}")
    return {"status": "READY", "reason": None, "figure_id": figure_id}


def write_plot_schema_manifest(path: str | Path) -> None:
    Path(path).write_text(json.dumps(PLOT_SCHEMA, indent=2, sort_keys=True) + "\n", encoding="utf-8")
