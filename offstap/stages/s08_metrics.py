"""
scripts/08_compute_metrics.py

Executable script to compute ATE and RPE on aligned trials.
"""
import sys
import os
import json
import pandas as pd

# Add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offstap.core.config import load_config
from offstap.core.metrics import evaluate_trajectory_metrics
from offstap.core.robust_statistics import compute_robust_statistics
from offstap.core.provenance import ProvenanceTracker
import numpy as np
from offstap.core.support import intersect_support


def _canonical_support_hash(values):
    """Persist the exact hash emitted by the canonical support contract."""
    frame = pd.DataFrame({"source_row_index": list(values or [])})
    evidence = intersect_support({"canonical": frame}, ["source_row_index"])
    return evidence.hash_sha256 if evidence.valid else ""

def main():
    cfg = load_config()
    prov = ProvenanceTracker(cfg.paths.get("output_provenance", "outputs/provenance"))

    base_out = cfg.paths.get("output_base", "outputs")
    aligned_dir = os.path.join(cfg.paths.get("output_aligned", "outputs/aligned"), "ct_reference")

    if not os.path.exists(aligned_dir):
        raise FileNotFoundError(f"Error: {aligned_dir} not found.")

    aligned_files = [f for f in os.listdir(aligned_dir) if f.endswith(".parquet")]

    print(f"Stage 08: Computing metrics for {len(aligned_files)} aligned trials...")

    metrics_log = []

    for f in aligned_files:
        tid = f.replace(f"_{cfg.config_id}.parquet", "")
        file_path = os.path.join(aligned_dir, f)

        df_aligned = pd.read_parquet(file_path)

        # Primary OFF-STAP metrics are held-out by construction.  The metric
        # function fails closed when persisted partition membership is absent.
        metrics = evaluate_trajectory_metrics(df_aligned, cfg, evaluation_only=True)
        metrics["trial_id"] = tid
        metrics_log.append(metrics)

        prov.record_stage(
            stage_name="08_compute_metrics",
            config_id=cfg.config_id,
            input_files=[file_path],
            output_files=[], # recorded below
            additional_meta={"ate_rmse": metrics.get("ate_rmse", np.nan), "status": metrics["status"]}
        )

    if metrics_log:
        metrics_out_dir = os.path.join(base_out, "14_metrics")
        os.makedirs(metrics_out_dir, exist_ok=True)

        metrics_path = os.path.join(metrics_out_dir, "evaluation_metrics.csv")
        df_metrics = pd.DataFrame(metrics_log)
        df_metrics.to_csv(metrics_path, index=False)

        # Granular exports

        # per_trial_metrics.csv
        # Requires: trial_id, config_id, alignment_method, representation_method, calibration_mode, clock_model,
        # rmse_mm, median_error_mm, mean_error_mm, p75_error_mm, p95_error_mm, max_error_mm,
        # ate_rmse_mm, ate_median_mm, rpe_rmse_mm, rpe_median_mm, heading_rmse_deg, num_outliers, outlier_fraction, accepted_for_hypothesis_tests
        per_trial = []
        for m in metrics_log:
            per_trial.append({
                "trial_id": m.get("trial_id"),
                "config_id": cfg.config_id,
                "alignment_method": cfg.temporal.get("alignment_method", "onset_plus_constrained_crosscorr"),
                "representation_method": cfg.representation.get("continuous_type", "spline"),
                "calibration_mode": "calibration_evaluation_split",
                "clock_model": "selected_per_trial",
                "rmse_mm": m.get("ate_rmse", np.nan) * 1000.0 if not np.isnan(m.get("ate_rmse", np.nan)) else np.nan,
                "median_error_mm": m.get("ate_mae", np.nan) * 1000.0 if not np.isnan(m.get("ate_mae", np.nan)) else np.nan,
                "mean_error_mm": m.get("ate_mean", np.nan) * 1000.0 if not np.isnan(m.get("ate_mean", np.nan)) else np.nan,
                "p75_error_mm": m.get("ate_p75", np.nan) * 1000.0 if not np.isnan(m.get("ate_p75", np.nan)) else np.nan,
                "p95_error_mm": m.get("ate_p95", np.nan) * 1000.0 if not np.isnan(m.get("ate_p95", np.nan)) else np.nan,
                "max_error_mm": m.get("ate_max", np.nan) * 1000.0 if not np.isnan(m.get("ate_max", np.nan)) else np.nan,
                "ate_rmse_mm": m.get("ate_rmse", np.nan) * 1000.0 if not np.isnan(m.get("ate_rmse", np.nan)) else np.nan,
                "ate_median_mm": m.get("ate_mae", np.nan) * 1000.0 if not np.isnan(m.get("ate_mae", np.nan)) else np.nan,
                "rpe_rmse_mm": m.get("rpe_rmse", np.nan) * 1000.0 if not np.isnan(m.get("rpe_rmse", np.nan)) else np.nan,
                "rpe_median_mm": m.get("rpe_median", np.nan) * 1000.0 if not np.isnan(m.get("rpe_median", np.nan)) else np.nan,
                "heading_rmse_deg": np.nan,
                "num_outliers": m.get("ate_outlier_count", 0),
                "outlier_fraction": m.get("ate_outlier_count", 0) / max(1, m.get("num_samples", 1)),
                "num_samples": m.get("num_samples", 0),
                "accepted_for_hypothesis_tests": 1 if m.get("status") == "success" and m.get("num_samples", 0) > 0 else 0,
                "metric_status": m.get("status"),
                "reason_if_invalid": m.get("failure_reason", ""),
                "support_count": len(m.get("support_keys", []) or []),
                "support_keys": json.dumps(m.get("support_keys", []) or [], separators=(",", ":")),
                "support_hash": _canonical_support_hash(m.get("support_keys", []) or []),
            })
        pd.DataFrame(per_trial).to_csv(os.path.join(metrics_out_dir, "per_trial_metrics.csv"), index=False)

        # per_phase_metrics.csv
        per_phase = []
        for m in metrics_log:
            phase_names = sorted({
                key.replace("num_samples_", "")
                for key in m.keys()
                if key.startswith("num_samples_") and m.get(key, 0) > 0
            })
            for phase in phase_names:
                phase_dict = {
                    "trial_id": m.get("trial_id"),
                    "config_id": cfg.config_id,
                    "motion_phase": phase,
                    "rmse_mm": m.get(f"ate_rmse_{phase}", np.nan) * 1000.0 if not np.isnan(m.get(f"ate_rmse_{phase}", np.nan)) else np.nan,
                    "median_error_mm": m.get(f"ate_mae_{phase}", np.nan) * 1000.0 if not np.isnan(m.get(f"ate_mae_{phase}", np.nan)) else np.nan,
                    "p95_error_mm": m.get(f"ate_p95_{phase}", np.nan) * 1000.0 if not np.isnan(m.get(f"ate_p95_{phase}", np.nan)) else np.nan,
                    "ate_rmse_mm": m.get(f"ate_rmse_{phase}", np.nan) * 1000.0 if not np.isnan(m.get(f"ate_rmse_{phase}", np.nan)) else np.nan,
                    "rpe_rmse_mm": m.get(f"rpe_rmse_{phase}", np.nan) * 1000.0 if not np.isnan(m.get(f"rpe_rmse_{phase}", np.nan)) else np.nan,
                    "num_samples": m.get(f"num_samples_{phase}", m.get("num_samples", 0))
                }

                rmse_val = phase_dict["rmse_mm"]
                n_samples = phase_dict["num_samples"]
                phase_dict["valid_for_paper"] = bool(np.isfinite(rmse_val) and n_samples > 0)

                per_phase.append(phase_dict)
        pd.DataFrame(per_phase).to_csv(os.path.join(metrics_out_dir, "per_phase_metrics.csv"), index=False)

        # ate_metrics.csv and rpe_metrics.csv
        df_metrics[["trial_id", "ate_rmse", "ate_mae", "ate_trimmed_rmse", "ate_huber_loss"]].to_csv(os.path.join(metrics_out_dir, "ate_metrics.csv"), index=False)
        df_metrics[["trial_id", "rpe_rmse"]].to_csv(os.path.join(metrics_out_dir, "rpe_metrics.csv"), index=False)

        # outlier_report.csv
        outlier_data = [{"trial_id": m.get("trial_id"), "num_outliers": m.get("ate_outlier_count", 0), "outlier_fraction": m.get("ate_outlier_count", 0)/max(1, m.get("num_samples", 1))} for m in metrics_log]
        pd.DataFrame(outlier_data).to_csv(os.path.join(metrics_out_dir, "outlier_report.csv"), index=False)

        # Robust Aggregation
        agg = {
            "ate_rmse": compute_robust_statistics(metrics_log, "ate_rmse"),
            "rpe_rmse": compute_robust_statistics(metrics_log, "rpe_rmse"),
            "ate_mae": compute_robust_statistics(metrics_log, "ate_mae"),
            "ate_trimmed_rmse": compute_robust_statistics(metrics_log, "ate_trimmed_rmse"),
            "ate_huber_loss": compute_robust_statistics(metrics_log, "ate_huber_loss")
        }

        # Add phase specific aggregates if present
        for phase in ["straight", "turning", "stopped"]:
            key = f"ate_rmse_{phase}"
            if key in pd.DataFrame(metrics_log).columns:
                agg[key] = compute_robust_statistics(metrics_log, key)

        agg_path = os.path.join(metrics_out_dir, "aggregated_metrics.json")
        with open(agg_path, 'w') as fh:
            json.dump(agg, fh, indent=2)

        # robust_metrics.csv
        robust_rows = []
        for metric_name, stats in agg.items():
            row = {"metric": metric_name}
            row.update(stats)
            robust_rows.append(row)
        pd.DataFrame(robust_rows).to_csv(os.path.join(metrics_out_dir, "robust_metrics.csv"), index=False)

        print("Metrics computed and robust statistics aggregated.")
    else:
        print("No metrics computed.")

if __name__ == "__main__":
    main()
