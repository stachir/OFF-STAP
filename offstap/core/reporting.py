"""
src/reporting.py

Generates the final markdown evaluation report.
"""
import os
import json
import pandas as pd
from datetime import datetime


def _read_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _metric_line(name: str, value, unit: str = "") -> str:
    if pd.isna(value):
        return f"- {name}: unavailable\n"
    if isinstance(value, float):
        return f"- {name}: {value:.3f}{unit}\n"
    return f"- {name}: {value}{unit}\n"

def generate_markdown_report(metrics_df: pd.DataFrame, agg_metrics: dict, output_path: str):
    """
    Generates a markdown report summarizing the pipeline run.
    """
    base_out = os.path.dirname(output_path)
    pair_df = _read_csv(os.path.join(base_out, "01_manifest", "pair_manifest.csv"))
    rejection_df = _read_csv(os.path.join(base_out, "04_quality", "rejection_log.csv"))
    per_trial_df = _read_csv(os.path.join(base_out, "14_metrics", "per_trial_metrics.csv"))
    h1_df = _read_csv(os.path.join(base_out, "15_hypotheses", "H1", "h1_trial_pairs.csv"))
    h2_df = _read_csv(os.path.join(base_out, "15_hypotheses", "H2", "h2_trial_comparison.csv"))
    h3_df = _read_csv(os.path.join(base_out, "15_hypotheses", "H3", "h3_calibration_mode_comparison.csv"))
    decisions_df = _read_csv(os.path.join(base_out, "17_statistics", "hypothesis_decision_summary.csv"))
    pipeline_df = _read_csv(
        os.path.join(base_out, "19_tables", "table_9_baseline_vs_proposed_pipeline.csv")
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Spatial-Temporal Alignment Report\n\n")
        f.write(f"Generated at: {datetime.utcnow().isoformat()}Z\n\n")
        f.write("Residuals are pipeline-conditional differences between EV3 odometry and aligned Vive trajectories. They are not absolute physical drift.\n\n")

        f.write("## Dataset Inventory\n\n")
        f.write(_metric_line("Paired or classified trial rows", len(pair_df)))
        if "dataset_tier" in pair_df.columns and not pair_df.empty:
            for tier, count in pair_df["dataset_tier"].value_counts().items():
                f.write(_metric_line(f"Dataset tier {tier}", int(count)))
        f.write(_metric_line("Rejected trial rows", rejection_df["trial_id"].nunique() if "trial_id" in rejection_df.columns else 0))

        f.write("\n## Residual Metrics\n\n")
        if not per_trial_df.empty:
            f.write(_metric_line("Trials with per-trial metrics", len(per_trial_df)))
            for col in ["rmse_mm", "ate_rmse_mm", "rpe_rmse_mm"]:
                if col in per_trial_df.columns:
                    f.write(_metric_line(f"Median {col}", per_trial_df[col].median(), " mm"))
                    f.write(_metric_line(f"P95 {col}", per_trial_df[col].quantile(0.95), " mm"))
        else:
            f.write("No per-trial metrics were generated.\n")

        f.write("\n## Hypothesis Outputs\n\n")
        f.write(_metric_line("H1 rows", len(h1_df)))
        f.write(_metric_line("H2 rows", len(h2_df)))
        f.write(_metric_line("H3 rows", len(h3_df)))
        if not decisions_df.empty:
            f.write("\n| Analysis set | Hypothesis | Decision | n | p-value |\n")
            f.write("|---|---|---:|---:|---:|\n")
            for _, row in decisions_df.iterrows():
                p_val = row.get("p_value")
                p_text = "" if pd.isna(p_val) else f"{float(p_val):.4g}"
                f.write(f"| {row.get('analysis_set', '')} | {row.get('hypothesis_id', '')} | {row.get('decision', '')} | {row.get('n', '')} | {p_text} |\n")

        f.write("\n## Baseline Versus Proposed Pipeline\n\n")
        if not pipeline_df.empty:
            keep_cols = [
                c
                for c in [
                    "trial_id",
                    "dataset_tier",
                    "baseline_rmse_mm",
                    "offstap_rmse_mm",
                    "delta_rmse_mm",
                    "common_support_count",
                ]
                if c in pipeline_df.columns
            ]
            f.write("| " + " | ".join(keep_cols) + " |\n")
            f.write("|" + "|".join(["---"] * len(keep_cols)) + "|\n")
            for _, row in pipeline_df[keep_cols].iterrows():
                f.write("| " + " | ".join(str(row.get(c, "")) for c in keep_cols) + " |\n")
        else:
            f.write("No baseline/proposed interpretation summary was generated.\n")

        f.write("\n## Files Generated\n\n")
        for root, _, files in os.walk(base_out):
            rel_root = os.path.relpath(root, base_out)
            if rel_root.startswith("03_cleaned") or rel_root.startswith("13_integrated_logs"):
                continue
            for name in files[:20]:
                f.write(f"- `{os.path.join(rel_root, name).replace(os.sep, '/')}`\n")
