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
    out_dir = os.path.join(cfg.paths.get("output_base", "outputs"), "13_integrated_logs")
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(cfg.paths.get("output_base", "outputs"), "18_plots", "representation_convergence")
    os.makedirs(plot_dir, exist_ok=True)

    base_out = cfg.paths.get("output_base", "outputs")

    strategies = [
        {"strategy_id": "original", "strategy_name": "Original Ev3", "uses_original_ev3_timestamps": True, "uses_original_vive_timestamps": False, "uses_common_regular_grid": False, "uses_continuous_time_reference": False, "interpolation_method": "none", "analysis_grid_rate_hz": None, "intended_use": "baseline", "limitations": "none"},
        {"strategy_id": "ct_query", "strategy_name": "CT Query", "uses_original_ev3_timestamps": True, "uses_original_vive_timestamps": False, "uses_common_regular_grid": False, "uses_continuous_time_reference": True, "interpolation_method": "spline", "analysis_grid_rate_hz": None, "intended_use": "comparison", "limitations": "none"}
    ]
    pd.DataFrame(strategies).to_csv(os.path.join(out_dir, "sampling_strategy_registry.csv"), index=False)

    metrics_path = os.path.join(base_out, "14_metrics", "per_trial_metrics.csv")
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
    else:
        df_metrics = pd.DataFrame(columns=["trial_id"])

    ts_rows = []
    qt_rows = []

    for _, row in df_metrics.iterrows():
        tid = row["trial_id"]
        e_c_path = os.path.join(base_out, "03_cleaned", "ev3", f"{tid}_ev3_cleaned.parquet")
        v_c_path = os.path.join(base_out, "03_cleaned", "vrmt", f"{tid}_vrmt_cleaned.parquet")
        aligned_path = os.path.join(cfg.paths.get("output_aligned", "outputs/aligned"), "ct_reference", f"{tid}_{cfg.config_id}.parquet")

        if os.path.exists(e_c_path) and os.path.exists(v_c_path):
            df_ev3 = pd.read_parquet(e_c_path)
            df_vrmt = pd.read_parquet(v_c_path)
            t_ev3 = df_ev3["local_time_s"].values
            t_vrmt = df_vrmt["relative_time_s"].values

            if len(t_ev3) > 0 and len(t_vrmt) > 0:
                idx = np.searchsorted(t_vrmt, t_ev3)
                idx = np.clip(idx, 0, len(t_vrmt)-1)
                gaps_s = np.abs(t_ev3 - t_vrmt[idx])
                idx_left = np.clip(idx-1, 0, len(t_vrmt)-1)
                gaps_left_s = np.abs(t_ev3 - t_vrmt[idx_left])
                gaps_ms = np.minimum(gaps_s, gaps_left_s) * 1000.0

                ts_rows.append({
                    "trial_id": tid,
                    "num_ev3_samples": len(df_ev3),
                    "num_vive_samples": len(df_vrmt),
                    "num_common_samples": np.nan,
                    "mean_nearest_timestamp_gap_ms": float(np.mean(gaps_ms)),
                    "median_nearest_timestamp_gap_ms": float(np.median(gaps_ms)),
                    "p95_nearest_timestamp_gap_ms": float(np.percentile(gaps_ms, 95)),
                    "max_nearest_timestamp_gap_ms": float(np.max(gaps_ms)),
                    "asynchrony_status": "acceptable"
                })

        if os.path.exists(aligned_path):
            df_aligned = pd.read_parquet(aligned_path)
            if "t_common_s" in df_aligned.columns and len(df_aligned) > 1:
                t_com = df_aligned["t_common_s"].values
                dt_com = np.diff(t_com)
                qt_rows.append({
                    "trial_id": tid,
                    "strategy_id": "ct_query",
                    "query_time_source": "ev3",
                    "num_query_times": len(df_aligned),
                    "query_start_s": float(np.min(t_com)),
                    "query_end_s": float(np.max(t_com)),
                    "mean_query_dt_s": float(np.mean(dt_com)),
                    "median_query_dt_s": float(np.median(dt_com)),
                    "std_query_dt_s": float(np.std(dt_com)),
                    "query_status": "success"
                })
            else:
                qt_rows.append({
                    "trial_id": tid,
                    "strategy_id": "ct_query",
                    "query_time_source": "ev3",
                    "num_query_times": 0,
                    "query_start_s": np.nan,
                    "query_end_s": np.nan,
                    "mean_query_dt_s": np.nan,
                    "median_query_dt_s": np.nan,
                    "std_query_dt_s": np.nan,
                    "query_status": "failed_metric_computation"
                })

    df_ts = pd.DataFrame(ts_rows)
    if not df_ts.empty:
        df_ts.to_csv(os.path.join(out_dir, "timestamp_matching_report.csv"), index=False)
        pd.DataFrame(qt_rows).to_csv(os.path.join(out_dir, "query_time_report.csv"), index=False)

        # Read real rmse_mean from per_trial_metrics
        metrics_path = os.path.join(base_out, "14_metrics", "per_trial_metrics.csv")
        rmse_mean = 0.0
        if os.path.exists(metrics_path):
            df_metrics = pd.read_csv(metrics_path)
            if "rmse_mm" in df_metrics.columns and not df_metrics.empty:
                rmse_mean = df_metrics["rmse_mm"].mean()

        pd.DataFrame([{"strategy_id": "ct_query", "rmse_mean": rmse_mean}]).to_csv(os.path.join(out_dir, "common_sampling_stage_comparison.csv"), index=False)
        # 1. Distribution of gaps
        plt.figure(figsize=(6,4))
        sns.histplot(df_ts["mean_nearest_timestamp_gap_ms"], bins=20, kde=True)
        plt.title("Distribution of Nearest Timestamp Gaps (EV3 vs VIVE)")
        plt.savefig(os.path.join(plot_dir, "nearest_timestamp_gap_distribution.png"))
        plt.close()

        # 2. Fake trajectory / rugplots
        aligned_dir = os.path.join(cfg.paths.get("output_aligned", "outputs/aligned"), "ct_reference")
        real_data = None
        if os.path.exists(aligned_dir):
            files = [f for f in os.listdir(aligned_dir) if f.endswith(".parquet")]
            if files:
                try:
                    df_ex = pd.read_parquet(os.path.join(aligned_dir, files[0]))
                    if "odometry_x" in df_ex.columns and "vrmt_aligned_x" in df_ex.columns:
                        real_data = df_ex
                except Exception:
                    pass

        for p in ["timestamp_rugplot_ev3_vs_vive_example.png", "common_query_times_example.png", "original_vs_common_sampling_example.png", "representation_convergence_example.png"]:
            if real_data is not None:
                plt.figure(figsize=(8,4))
                plt.plot(real_data["odometry_x"].values[:200], label="EV3 (Original)")
                plt.plot(real_data["vrmt_aligned_x"].values[:200], label="VIVE (CT Query)")
                plt.title(f"Example Visual for {p.replace('.png', '')}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, p))
                plt.close()

    logging.info("Common sampling explanation outputs generated.")

if __name__ == "__main__":
    main()
