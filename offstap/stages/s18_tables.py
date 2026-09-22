import os
import sys
import glob
import json
import yaml
import logging
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from offstap.core.config import load_config

def setup_logging(cfg):
    log_file = os.path.join(cfg.paths.get("output_base", "outputs"), "pipeline_stage_log.csv")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return log_file


def _require_authoritative_gate_ledger(base_out):
    path = os.path.join(base_out, "19_tables", "authoritative_population_gate_ledger.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Authoritative population gate ledger is required: {path}")
    ledger = pd.read_csv(path)
    required = {"trial_id", "gate", "gate_status", "evidence_path"}
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise ValueError(f"Authoritative population gate ledger missing columns: {missing}")
    return path

def main():
    cfg = load_config("config/alignment_config.yaml")
    out_dir = os.path.join(cfg.paths.get("output_base", "outputs"), "19_tables")
    os.makedirs(out_dir, exist_ok=True)

    base_out = cfg.paths.get("output_base", "outputs")
    _require_authoritative_gate_ledger(base_out)
    stats_dir = os.path.join(base_out, "17_statistics")
    h1_dir = os.path.join(base_out, "15_hypotheses", "H1")
    h2_dir = os.path.join(base_out, "15_hypotheses", "H2")
    h3_dir = os.path.join(base_out, "15_hypotheses", "H3")

    # Safely load data or provide defaults
    def load_df(path):
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    df_acc = load_df(os.path.join(stats_dir, "accepted_trials_summary.csv"))
    df_rej = load_df(os.path.join(stats_dir, "rejected_trials_summary.csv"))

    acc_count = int(df_acc["n"].iloc[0]) if not df_acc.empty and "n" in df_acc.columns else 0
    rej_count = int(df_rej["count"].sum()) if not df_rej.empty and "count" in df_rej.columns else 0

    df_metrics = load_df(os.path.join(base_out, "14_metrics", "per_trial_metrics.csv"))
    if not df_metrics.empty:
        mean_start_res = df_metrics["spatial_fit_rmse_mm"].mean() if "spatial_fit_rmse_mm" in df_metrics.columns else np.nan
        mean_temp_off = df_metrics["temporal_offset_s"].mean() if "temporal_offset_s" in df_metrics.columns else np.nan
        mean_peak_conf = df_metrics["peak_confidence_ratio"].mean() if "peak_confidence_ratio" in df_metrics.columns else np.nan
    else:
        mean_start_res, mean_temp_off, mean_peak_conf = np.nan, np.nan, np.nan

    tables = [
        ("table_1_dataset_summary.csv", pd.DataFrame([{"metric": "paired trials", "count": acc_count + rej_count}, {"metric": "accepted trials", "count": acc_count}])),
        ("table_2_quality_filtering.csv", df_rej if not df_rej.empty else pd.DataFrame([{"rejection_reason": "none", "count": 0}])),
        ("table_3_spatial_diagnostics.csv", pd.DataFrame([{"metric": "mean_spatial_fit_rmse_mm", "value": mean_start_res}])),
        ("table_4_temporal_alignment.csv", pd.DataFrame([{"metric": "mean_temporal_offset_s", "value": mean_temp_off}, {"metric": "mean_peak_confidence", "value": mean_peak_conf}])),
    ]

    # H2 results
    df_h2 = load_df(os.path.join(h2_dir, "h2_summary_by_method.csv"))
    if not df_h2.empty:
        tables.append(("table_5_representation_methods.csv", df_h2))
    else:
        tables.append(("table_5_representation_methods.csv", pd.DataFrame(columns=["method", "rmse_mm"])))

    # H3 results
    df_h3 = load_df(os.path.join(h3_dir, "h3_calibration_mode_comparison.csv"))
    if not df_h3.empty:
        # compute summary
        h3_summ = df_h3.groupby("calibration_mode")["heldout_rmse_mm"].mean().reset_index()
        tables.append(("table_6_calibration_modes.csv", h3_summ))
    else:
        tables.append(("table_6_calibration_modes.csv", pd.DataFrame(columns=["calibration_mode", "heldout_rmse_mm"])))

    df_h1_pairs = load_df(os.path.join(h1_dir, "h1_trial_pairs.csv"))
    if df_h1_pairs.empty or df_h2.empty or df_h3.empty:
        raise RuntimeError("Current-run H1/H2/H3 artifacts are required for tables")
    decision_path = os.path.join(stats_dir, "hypothesis_decision_summary.csv")
    df_decisions = load_df(decision_path)
    if df_decisions.empty:
        raise RuntimeError("Current-run hypothesis decisions are required for tables")
    tables.append(("table_7_hypothesis_results.csv", df_decisions))
    tables.append(("table_8_selected_pipeline_configuration.csv", pd.DataFrame([{"config_id": cfg.config_id, "description": "Proposed pipeline"}])))

    # Baseline vs OFF-STAP: package the corrected current-run adapter outputs
    # without recomputing or reinterpreting any metric.
    baseline_dir = os.path.join(base_out, "baseline")
    baseline_trials_path = os.path.join(baseline_dir, "baseline_trial_metrics.csv")
    baseline_summary_path = os.path.join(baseline_dir, "baseline_summary.json")
    if not os.path.exists(baseline_trials_path) or not os.path.exists(baseline_summary_path):
        raise FileNotFoundError("Corrected current-run baseline artifacts are required for tables")
    df_pipeline = pd.read_csv(baseline_trials_path)
    if df_pipeline.empty:
        raise RuntimeError("Corrected current-run baseline trial table is empty")
    with open(baseline_summary_path, encoding="utf-8") as stream:
        df_effect = pd.DataFrame(json.load(stream))
    if df_effect.empty:
        raise RuntimeError("Corrected current-run baseline summary is empty")
    tables.append(("table_9_baseline_vs_proposed_pipeline.csv", df_pipeline))
    tables.append(("table_10_paired_pipeline_difference_summary.csv", df_effect))

    for filename, data in tables:
        data.to_csv(os.path.join(out_dir, filename), index=False)

    logging.info("Paper tables generated.")

if __name__ == "__main__":
    main()
