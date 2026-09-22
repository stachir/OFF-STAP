import os
import sys
import glob
import json
import yaml
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from offstap.core.config import load_config

def setup_logging(cfg):
    log_file = os.path.join(cfg.paths.get("output_base", "outputs"), "pipeline_stage_log.csv")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return log_file

def main():
    cfg = load_config("config/alignment_config.yaml")
    out_dir = os.path.join(cfg.paths.get("output_base", "outputs"), "17_statistics")
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(cfg.paths.get("output_base", "outputs"), "18_plots", "scenario_grid")
    os.makedirs(plot_dir, exist_ok=True)

    base_out = cfg.paths.get("output_base", "outputs")

    metrics_path = os.path.join(base_out, "14_metrics", "per_trial_metrics.csv")
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
    else:
        df_metrics = pd.DataFrame(columns=["trial_id", "rmse_mm", "ate_rmse_mm", "rpe_rmse_mm"])

    df_metrics["drive_id"] = df_metrics["trial_id"].str.split('_').str[0]
    df_metrics["route_id"] = df_metrics["trial_id"].str.split('_').str[1]
    df_metrics["scenario_id"] = df_metrics["drive_id"] + "_" + df_metrics["route_id"]

    drives = [f"D{i}" for i in range(1, 10)]
    routes = [f"T{i}" for i in range(1, 10)]

    # 1. scenario_level_metrics.csv
    scen_group = df_metrics.groupby("scenario_id")
    scen_rows = []
    diff_rows = []

    # Load additional diagnostics
    spatial_path = os.path.join(base_out, "07_spatial", "spatial_diagnostics.csv")
    df_spatial = pd.read_csv(spatial_path) if os.path.exists(spatial_path) else pd.DataFrame()
    temporal_path = os.path.join(base_out, "09_temporal", "crosscorr_report.csv")
    df_temporal = pd.read_csv(temporal_path) if os.path.exists(temporal_path) else pd.DataFrame()
    diag_path = os.path.join(base_out, "04_quality", "stream_diagnostics.csv")
    df_diag = pd.read_csv(diag_path) if os.path.exists(diag_path) else pd.DataFrame()

    for scen_id, grp in scen_group:
        trials = grp["trial_id"].tolist()

        if not df_spatial.empty and "trial_id" in df_spatial.columns:
            sp_grp = df_spatial[df_spatial["trial_id"].isin(trials)]
            heading_rmse = sp_grp.get("heading_residual_deg", pd.Series(dtype=float)).mean()
        else:
            heading_rmse = np.nan

        if not df_temporal.empty and "trial_id" in df_temporal.columns:
            temp_grp = df_temporal[df_temporal["trial_id"].isin(trials)]
            peak_conf = temp_grp.get("peak_confidence_ratio", pd.Series(dtype=float)).mean()
            ambig_rate = (temp_grp["status"] != "ok").mean() if not temp_grp.empty and "status" in temp_grp.columns else np.nan
        else:
            peak_conf = np.nan
            ambig_rate = np.nan

        if not df_diag.empty and "trial_id" in df_diag.columns:
            diag_grp = df_diag[df_diag["trial_id"].isin(trials)]
            ts_irreg = diag_grp.get("ev3_irregularity", pd.Series(dtype=float)).mean()
        else:
            ts_irreg = np.nan

        scen_rows.append({
            "scenario_id": scen_id,
            "drive_id": grp["drive_id"].iloc[0],
            "route_id": grp["route_id"].iloc[0],
            "num_valid_repetitions": len(grp),
            "mean_rmse_mm": grp["rmse_mm"].mean() if not grp.empty else 0,
            "median_rmse_mm": grp["rmse_mm"].median() if not grp.empty else 0,
            "p95_rmse_mm": grp["rmse_mm"].quantile(0.95) if not grp.empty else 0,
            "mean_ate_mm": grp.get("ate_rmse_mm", pd.Series(dtype=float)).mean(),
            "mean_rpe_mm": grp.get("rpe_rmse_mm", pd.Series(dtype=float)).mean(),
            "mean_heading_rmse_deg": heading_rmse,
            "mean_timestamp_irregularity": ts_irreg,
            "mean_peak_confidence_ratio": peak_conf,
            "mean_tracking_loss_duration_s": np.nan,
            "mean_planarity_rmse_mm": np.nan,
            "scenario_status": "robust" if grp["rmse_mm"].mean() < 50 else "brittle"
        })

        diff_rows.append({
            "scenario_id": scen_id,
            "drive_id": grp["drive_id"].iloc[0],
            "route_id": grp["route_id"].iloc[0],
            "quality_rejection_rate": np.nan,
            "temporal_ambiguity_rate": ambig_rate,
            "spatial_failure_rate": np.nan,
            "ct_model_failure_rate": np.nan,
            "mean_crosscorr_peak_confidence": peak_conf,
            "mean_calibration_degeneracy_score": np.nan,
            "mean_residual_rmse_mm": grp["rmse_mm"].mean() if not grp.empty else 0,
            "alignment_difficulty_score": (grp["rmse_mm"].mean() / 100.0) if not grp.empty else 0.0,
            "difficulty_class": "hard" if (grp["rmse_mm"].mean() > 50) else "easy",
            "main_difficulty_reason": "none"
        })

    df_scen = pd.DataFrame(scen_rows)
    df_scen.to_csv(os.path.join(out_dir, "scenario_level_metrics.csv"), index=False)

    df_diff = pd.DataFrame(diff_rows)
    df_diff.to_csv(os.path.join(out_dir, "scenario_level_alignment_difficulty.csv"), index=False)

    pd.DataFrame(scen_rows).to_csv(os.path.join(out_dir, "scenario_level_acceptance.csv"), index=False)
    with open(os.path.join(out_dir, "method_sensitivity_status.json"), "w", encoding="utf-8") as stream:
        json.dump({
            "product": "method_sensitivity",
            "status": "NOT_AVAILABLE",
            "reason": (
                "No valid current-run between-method sensitivity computation is defined "
                "in the approved processing contract."
            ),
            "suppressed_artifacts": [
                "17_statistics/scenario_level_method_sensitivity.csv",
                "18_plots/scenario_grid/method_sensitivity_9x9_heatmap.png",
            ],
        }, stream, indent=2)
        stream.write("\n")

    if not df_scen.empty:
        df_scen.sort_values(by="mean_rmse_mm", inplace=True)
        df_scen.to_csv(os.path.join(out_dir, "scenario_ranking.csv"), index=False)

        # Plots
        grid_rmse = np.zeros((9, 9))
        grid_ate = np.zeros((9, 9))
        grid_rpe = np.zeros((9, 9))

        for d_idx, d in enumerate(drives):
            for r_idx, r in enumerate(routes):
                scen = f"{d}_{r}"
                row = df_scen[df_scen["scenario_id"] == scen]
                if not row.empty:
                    grid_rmse[d_idx, r_idx] = row["mean_rmse_mm"].iloc[0]
                    grid_ate[d_idx, r_idx] = row["mean_ate_mm"].iloc[0]
                    grid_rpe[d_idx, r_idx] = row["mean_rpe_mm"].iloc[0]

        plt.figure(figsize=(8,6))
        sns.heatmap(grid_rmse, annot=True, xticklabels=routes, yticklabels=drives, cmap="viridis")
        plt.title("Residual Mean 9x9 Heatmap")
        plt.savefig(os.path.join(plot_dir, "residual_mean_9x9_heatmap.png"))

        plt.figure(figsize=(8,6))
        sns.heatmap(grid_ate, annot=True, xticklabels=routes, yticklabels=drives, cmap="viridis")
        plt.title("ATE Mean 9x9 Heatmap")
        plt.savefig(os.path.join(plot_dir, "ate_mean_9x9_heatmap.png"))

        plt.figure(figsize=(8,6))
        sns.heatmap(grid_rpe, annot=True, xticklabels=routes, yticklabels=drives, cmap="viridis")
        plt.title("RPE Mean 9x9 Heatmap")
        plt.savefig(os.path.join(plot_dir, "rpe_mean_9x9_heatmap.png"))

        # Create difficulty grid
        grid_diff = np.zeros((9, 9))
        for d_idx, d in enumerate(drives):
            for r_idx, r in enumerate(routes):
                scen = f"{d}_{r}"
                row = df_diff[df_diff["scenario_id"] == scen]
                if not row.empty:
                    grid_diff[d_idx, r_idx] = row["alignment_difficulty_score"].iloc[0]

        plt.figure(figsize=(8,6))
        sns.heatmap(grid_diff, annot=True, xticklabels=routes, yticklabels=drives, cmap="magma")
        plt.title("Alignment Difficulty Heatmap")
        plt.savefig(os.path.join(plot_dir, "alignment_difficulty_9x9_heatmap.png"))

        # Create temporal ambiguity grid
        grid_ambig = np.zeros((9, 9))
        for d_idx, d in enumerate(drives):
            for r_idx, r in enumerate(routes):
                scen = f"{d}_{r}"
                row = df_diff[df_diff["scenario_id"] == scen]
                if not row.empty and pd.notna(row["temporal_ambiguity_rate"].iloc[0]):
                    grid_ambig[d_idx, r_idx] = row["temporal_ambiguity_rate"].iloc[0]

        plt.figure(figsize=(8,6))
        sns.heatmap(grid_ambig, annot=True, xticklabels=routes, yticklabels=drives, cmap="plasma")
        plt.title("Temporal Ambiguity Rate")
        plt.savefig(os.path.join(plot_dir, "temporal_ambiguity_9x9_heatmap.png"))

        plt.figure(figsize=(10,5))
        sns.barplot(x="scenario_id", y="mean_rmse_mm", data=df_scen)
        plt.xticks(rotation=90)
        plt.title("Scenario Ranking by Residual")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "scenario_ranking_by_residual.png"))

        plt.figure(figsize=(10,5))
        sns.barplot(x="scenario_id", y="alignment_difficulty_score", data=df_diff.sort_values(by="alignment_difficulty_score"))
        plt.xticks(rotation=90)
        plt.title("Scenario Ranking by Difficulty")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "scenario_ranking_by_difficulty.png"))

        plt.close('all')

    logging.info("Scenario robustness outputs generated.")

if __name__ == "__main__":
    main()
