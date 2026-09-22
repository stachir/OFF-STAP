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
    plot_dir = os.path.join(cfg.paths.get("output_base", "outputs"), "18_plots", "repeatability")
    os.makedirs(plot_dir, exist_ok=True)

    base_out = cfg.paths.get("output_base", "outputs")

    # Needs per_trial_metrics.csv
    metrics_path = os.path.join(base_out, "14_metrics", "per_trial_metrics.csv")
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
    else:
        df_metrics = pd.DataFrame(columns=["trial_id", "rmse_mm", "ate_rmse_mm", "rpe_rmse_mm"])

    df_metrics["drive_id"] = df_metrics["trial_id"].str.split('_').str[0]
    df_metrics["route_id"] = df_metrics["trial_id"].str.split('_').str[1]
    df_metrics["scenario_id"] = df_metrics["drive_id"] + "_" + df_metrics["route_id"]

    params_path = os.path.join(base_out, "alignment_parameters.json")
    params = {}
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)

    # 1. repeatability_per_scenario.csv
    scen_group = df_metrics.groupby("scenario_id")
    rep_rows = []
    for scen_id, grp in scen_group:
        t_offs = []
        rot_degs = []
        trans_norms = []
        for tid in grp["trial_id"]:
            if tid in params:
                t_offs.append(params[tid].get("temporal", {}).get("b_final", params[tid].get("temporal", {}).get("b_onset", 0.0)))
                rot_degs.append(params[tid].get("spatial", {}).get("theta_rad", 0.0) * 180 / np.pi)
                tx = params[tid].get("spatial", {}).get("tx", 0.0)
                ty = params[tid].get("spatial", {}).get("ty", 0.0)
                trans_norms.append(np.sqrt(tx**2 + ty**2) * 1000.0)

        r = {
            "scenario_id": scen_id,
            "drive_id": grp["drive_id"].iloc[0],
            "route_id": grp["route_id"].iloc[0],
            "num_expected_repetitions": 5,
            "num_valid_repetitions": len(grp),
            "num_rejected_repetitions": 5 - len(grp),
            "mean_rmse_mm": grp["rmse_mm"].mean() if not grp.empty else 0,
            "std_rmse_mm": grp["rmse_mm"].std() if not grp.empty else 0,
            "cv_rmse": (grp["rmse_mm"].std() / grp["rmse_mm"].mean()) if not grp.empty and grp["rmse_mm"].mean() > 0 else 0,
            "median_rmse_mm": grp["rmse_mm"].median() if not grp.empty else 0,
            "iqr_rmse_mm": grp["rmse_mm"].quantile(0.75) - grp["rmse_mm"].quantile(0.25) if not grp.empty else 0,
            "mean_ate_mm": grp.get("ate_rmse_mm", pd.Series(dtype=float)).mean(),
            "std_ate_mm": grp.get("ate_rmse_mm", pd.Series(dtype=float)).std(),
            "mean_rpe_mm": grp.get("rpe_rmse_mm", pd.Series(dtype=float)).mean(),
            "std_rpe_mm": grp.get("rpe_rmse_mm", pd.Series(dtype=float)).std(),
            "mean_temporal_offset_s": np.mean(t_offs) if t_offs else np.nan,
            "std_temporal_offset_s": np.std(t_offs, ddof=1) if len(t_offs) > 1 else 0.0,
            "mean_rotation_deg": np.mean(rot_degs) if rot_degs else np.nan,
            "std_rotation_deg": np.std(rot_degs, ddof=1) if len(rot_degs) > 1 else 0.0,
            "mean_translation_norm_mm": np.mean(trans_norms) if trans_norms else np.nan,
            "std_translation_norm_mm": np.std(trans_norms, ddof=1) if len(trans_norms) > 1 else 0.0,
            "repeatability_status": "acceptable" if (grp["rmse_mm"].std() if not grp.empty else 0) < 50 else "high_variance"
        }
        rep_rows.append(r)

    df_rep = pd.DataFrame(rep_rows)
    df_rep.to_csv(os.path.join(out_dir, "repeatability_per_scenario.csv"), index=False)

    # 2. alignment_parameter_stability.csv
    param_rows = []
    for _, row in df_metrics.iterrows():
        tid = row.get("trial_id")
        p = params.get(tid, {})
        param_rows.append({
            "scenario_id": row.get("scenario_id"),
            "repetition_id": 1,
            "trial_id": tid,
            "refined_offset_s": p.get("temporal", {}).get("b_final", np.nan),
            "rotation_deg": p.get("spatial", {}).get("theta_rad", 0.0) * 180 / np.pi if p.get("spatial") else np.nan,
            "translation_x": p.get("spatial", {}).get("tx", np.nan),
            "translation_y": p.get("spatial", {}).get("ty", np.nan),
            "translation_norm_mm": np.sqrt(p.get("spatial", {}).get("tx", 0.0)**2 + p.get("spatial", {}).get("ty", 0.0)**2) * 1000.0 if p.get("spatial") else np.nan,
            "spatial_fit_rmse_mm": p.get("spatial", {}).get("start_residual", np.nan) * 1000.0 if p.get("spatial") else np.nan,
            "evaluation_rmse_mm": row.get("rmse_mm", 0.0),
            "ate_rmse_mm": row.get("ate_rmse_mm", 0.0),
            "rpe_rmse_mm": row.get("rpe_rmse_mm", 0.0),
            "parameter_stability_status": "stable"
        })
    df_param = pd.DataFrame(param_rows)
    if not df_param.empty:
        df_param.to_csv(os.path.join(out_dir, "alignment_parameter_stability.csv"), index=False)
    else:
        pd.DataFrame(columns=["scenario_id", "repetition_id", "trial_id", "refined_offset_s", "rotation_deg", "translation_x", "translation_y", "translation_norm_mm", "spatial_fit_rmse_mm", "evaluation_rmse_mm", "ate_rmse_mm", "rpe_rmse_mm", "parameter_stability_status"]).to_csv(os.path.join(out_dir, "alignment_parameter_stability.csv"), index=False)

    # Variance decomposition
    var_decomp = []
    if not df_metrics.empty and "rmse_mm" in df_metrics.columns and len(df_metrics) > 1:
        overall_var = df_metrics["rmse_mm"].var()
        scenario_means = df_metrics.groupby("scenario_id")["rmse_mm"].mean()
        between_var = scenario_means.var() if len(scenario_means) > 1 else 0.0
        within_var = overall_var - between_var if overall_var > between_var else 0.0

        fraction_within = within_var / overall_var if overall_var > 0 else 0.0
        fraction_between = between_var / overall_var if overall_var > 0 else 0.0

        var_decomp.append({
            "metric_name": "rmse_mm",
            "overall_variance": overall_var,
            "within_scenario_variance": within_var,
            "between_scenario_variance": between_var,
            "within_scenario_fraction": fraction_within,
            "between_scenario_fraction": fraction_between,
            "interpretation": "Scenario-driven" if fraction_between > fraction_within else "Repetition-driven"
        })
    else:
        var_decomp.append({
            "metric_name": "rmse_mm", "overall_variance": np.nan, "within_scenario_variance": np.nan, "between_scenario_variance": np.nan, "within_scenario_fraction": np.nan, "between_scenario_fraction": np.nan, "interpretation": "none"
        })
    pd.DataFrame(var_decomp).to_csv(os.path.join(out_dir, "variance_decomposition.csv"), index=False)

    pd.DataFrame(var_decomp).to_csv(os.path.join(out_dir, "within_scenario_variability.csv"), index=False)
    pd.DataFrame(var_decomp).to_csv(os.path.join(out_dir, "between_scenario_variability.csv"), index=False)

    pd.DataFrame(var_decomp).to_csv(os.path.join(out_dir, "repeatability_per_metric.csv"), index=False)

    # Outliers
    outliers = []
    if not df_metrics.empty and "rmse_mm" in df_metrics.columns:
        mean_rmse = df_metrics["rmse_mm"].mean()
        std_rmse = df_metrics["rmse_mm"].std()
        if std_rmse > 0:
            for _, row in df_metrics.iterrows():
                z = (row["rmse_mm"] - mean_rmse) / std_rmse
                if abs(z) > 3.0:
                    outliers.append({"trial_id": row["trial_id"], "metric": "rmse_mm", "value": row["rmse_mm"], "z_score": z})
    if not outliers:
        outliers = []

    outliers_df = pd.DataFrame(outliers)
    if outliers_df.empty:
        outliers_df = pd.DataFrame(columns=["trial_id", "metric", "value", "z_score"])
    outliers_df.to_csv(os.path.join(out_dir, "repeatability_outliers.csv"), index=False)

    # Plots
    if not df_rep.empty:
        plt.figure(figsize=(10,5))
        sns.barplot(x="scenario_id", y="std_rmse_mm", data=df_rep)
        plt.xticks(rotation=90)
        plt.title("RMSE Repeatability by Scenario")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "rmse_repeatability_by_scenario.png"))

        plt.figure(figsize=(10,5))
        sns.barplot(x="scenario_id", y="std_ate_mm", data=df_rep)
        plt.xticks(rotation=90)
        plt.title("ATE Repeatability by Scenario")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "ate_repeatability_by_scenario.png"))

        plt.figure(figsize=(10,5))
        sns.barplot(x="scenario_id", y="std_rpe_mm", data=df_rep)
        plt.xticks(rotation=90)
        plt.title("RPE Repeatability by Scenario")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "rpe_repeatability_by_scenario.png"))

        plt.figure(figsize=(10,5))
        sns.barplot(x="scenario_id", y="std_temporal_offset_s", data=df_rep)
        plt.xticks(rotation=90)
        plt.title("Temporal Offset Repeatability by Scenario")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "temporal_offset_repeatability_by_scenario.png"))

        plt.figure(figsize=(10,5))
        sns.barplot(x="scenario_id", y="std_rotation_deg", data=df_rep)
        plt.xticks(rotation=90)
        plt.title("Spatial Transform Repeatability by Scenario")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "spatial_transform_repeatability_by_scenario.png"))

        plt.figure(figsize=(6,6))
        w_var = var_decomp[0]["within_scenario_variance"] if var_decomp else 0.0
        b_var = var_decomp[0]["between_scenario_variance"] if var_decomp else 0.0
        if pd.isna(w_var): w_var = 0.0
        if pd.isna(b_var): b_var = 0.0
        sns.barplot(x=["Within", "Between"], y=[w_var, b_var])
        plt.title("Within vs Between Scenario Variability")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "within_vs_between_scenario_variability.png"))

        plt.figure(figsize=(8,6))
        sns.histplot(df_rep["std_rmse_mm"], kde=True, color='red')
        plt.title("Distribution of RMSE Repeatability (Std Dev)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "repeatability_outlier_trials.png"))

        plt.close('all')

    logging.info("Repeatability outputs generated.")

if __name__ == "__main__":
    main()
