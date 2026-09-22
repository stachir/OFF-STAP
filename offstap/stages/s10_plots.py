"""
scripts/10_generate_plots.py

Executable script to generate evaluation plots.
"""
import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offstap.core.config import load_config
from offstap.core.plots import plot_spatial_alignment, plot_metrics_boxplot, plot_speed_comparison
from offstap.core.provenance import ProvenanceTracker
from offstap.core.plot_schema import (
    H2_PLOT_APPLICABILITY_CONTRACT,
    PlotSchemaError,
    validate_plot_input,
    write_plot_schema_manifest,
)

CANONICAL_H2_METHODS = frozenset({"zoh", "linear", "pchip", "spline"})


def _valid_h1_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the producer's already-approved H1 rows for display only."""
    if frame.empty or "valid_for_paper" not in frame.columns:
        return frame.iloc[0:0]
    accepted = frame["valid_for_paper"].astype(str).str.lower().isin(
        {"true", "1", "yes", "valid"}
    )
    return frame.loc[accepted].copy()


def _valid_hypothesis_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select persisted-valid H2/H3 rows for scientific comparison plots."""
    if frame.empty:
        return frame.iloc[0:0]
    if "valid_for_paper" in frame.columns:
        accepted = frame["valid_for_paper"].astype(str).str.lower().isin(
            {"true", "1", "yes", "valid"}
        )
        return frame.loc[accepted].copy()
    if "status" in frame.columns:
        accepted = frame["status"].astype(str).str.lower().isin(
            {"ok", "valid", "success", "accepted"}
        )
        return frame.loc[accepted].copy()
    return frame.iloc[0:0]


def _h2_decision(figure_id: str, status: str, reason: str | None) -> dict:
    contract = H2_PLOT_APPLICABILITY_CONTRACT[figure_id]
    return {
        "figure_id": figure_id,
        "applicability_status": status,
        "applicability_reason": reason,
        "required_inputs": contract["required_inputs"],
        "mode_behavior": contract["mode_behavior"],
    }


def _h2_canonical(frame: pd.DataFrame) -> bool:
    methods = set(frame["method"].astype(str).str.strip().str.lower()) if "method" in frame.columns else set()
    phases = set(frame["phase"].astype(str).str.strip().str.lower()) if "phase" in frame.columns else set()
    return methods == CANONICAL_H2_METHODS and phases == {"all"}


def h2_plot_applicability(frame: pd.DataFrame, figure_id: str) -> dict:
    """Return the frozen applicability decision for one H2 Stage 10 output."""
    if figure_id not in H2_PLOT_APPLICABILITY_CONTRACT:
        raise KeyError(f"Unknown H2 plot applicability contract: {figure_id}")
    required = H2_PLOT_APPLICABILITY_CONTRACT[figure_id]["required_inputs"]
    canonical = _h2_canonical(frame)
    missing = [column for column in required if column not in frame.columns]
    if (
        figure_id == "h2_error_vs_timestamp_irregularity"
        and canonical
        and missing == ["timestamp_irregularity"]
    ):
        return _h2_decision(
            figure_id,
            "NOT_APPLICABLE",
            "canonical_h2_artifact_has_no_persisted_timestamp_irregularity",
        )
    if missing:
        return _h2_decision(figure_id, "INVALID_SOURCE", f"missing_required_inputs:{','.join(missing)}")

    eligible = _valid_hypothesis_rows(frame)
    if eligible.empty:
        return _h2_decision(figure_id, "INVALID_SOURCE", "no_eligible_rows")

    canonical = _h2_canonical(eligible)
    if figure_id in {"h2_method_boxplot", "h2_method_violin_or_histogram"}:
        if not canonical:
            return _h2_decision(
                figure_id,
                "INVALID_SOURCE",
                "canonical_h2_methods_or_phase_contract_mismatch",
            )
        rmse = pd.to_numeric(eligible["rmse_mm"], errors="coerce")
        if not np.isfinite(rmse).all():
            return _h2_decision(figure_id, "INVALID_SOURCE", "rmse_mm_nonfinite")
        return _h2_decision(figure_id, "APPLICABLE", None)
    if figure_id == "h2_ct_vs_discrete_scatter":
        methods = set(eligible["method"].astype(str).str.strip().str.lower())
        if canonical:
            return _h2_decision(
                figure_id,
                "NOT_APPLICABLE",
                "canonical_h2_methods_exclude_continuous_time_query",
            )
        if {"continuous_time_query", "linear"}.issubset(methods):
            ct_ids = set(eligible.loc[eligible["method"].astype(str).str.lower().eq("continuous_time_query"), "trial_id"])
            linear_ids = set(eligible.loc[eligible["method"].astype(str).str.lower().eq("linear"), "trial_id"])
            if ct_ids & linear_ids:
                return _h2_decision(figure_id, "APPLICABLE", None)
            reason = "continuous_time_query_and_linear_have_no_common_trials"
        else:
            reason = "h2_methods_do_not_match_canonical_or_ct_comparison_contract"
        return _h2_decision(figure_id, "INVALID_SOURCE", reason)
    if figure_id == "h2_error_vs_timestamp_irregularity":
        if canonical and "timestamp_irregularity" not in eligible.columns:
            return _h2_decision(
                figure_id,
                "NOT_APPLICABLE",
                "canonical_h2_artifact_has_no_persisted_timestamp_irregularity",
            )
        if not pd.to_numeric(eligible["timestamp_irregularity"], errors="coerce").dropna().empty:
            return _h2_decision(figure_id, "APPLICABLE", None)
        return _h2_decision(figure_id, "INVALID_SOURCE", "timestamp_irregularity_present_but_nonfinite")
    if figure_id == "h2_phase_specific_method_comparison":
        if canonical:
            return _h2_decision(
                figure_id,
                "NOT_APPLICABLE",
                "canonical_h2_artifact_contains_only_all_phase",
            )
        phases = set(eligible["phase"].astype(str).str.strip().str.lower())
        if phases - {"all"}:
            return _h2_decision(figure_id, "APPLICABLE", None)
        return _h2_decision(figure_id, "INVALID_SOURCE", "phase_specific_rows_unavailable")
    raise AssertionError(f"Unhandled H2 plot applicability contract: {figure_id}")


def _h2_status_row(decision: dict, source_table: str, status: str) -> dict:
    return {
        "figure_id": decision["figure_id"],
        "status": status,
        "reason": decision["applicability_reason"],
        "applicability_status": decision["applicability_status"],
        "applicability_reason": decision["applicability_reason"],
        "required_inputs": "|".join(decision["required_inputs"]),
        "mode_behavior": decision["mode_behavior"],
        "source_table": source_table,
    }


def _enforce_h2_decision(decision: dict) -> None:
    if decision["applicability_status"] == "INVALID_SOURCE":
        raise PlotSchemaError(
            f"{decision['figure_id']}: INVALID_SOURCE: {decision['applicability_reason']}"
        )

def main():
    cfg = load_config()
    prov = ProvenanceTracker(cfg.paths.get("output_provenance", "outputs/provenance"))

    base_out = cfg.paths.get("output_base", "outputs")
    metrics_path = os.path.join(base_out, "14_metrics", "evaluation_metrics.csv")
    aligned_dir = os.path.join(cfg.paths.get("output_aligned", "outputs/aligned"), "ct_reference")
    plots_dir = cfg.paths.get("output_plots", os.path.join(base_out, "18_plots"))

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Create required subdirectories
    audit_dir = os.path.join(plots_dir, "audit")
    results_dir = os.path.join(plots_dir, "results")
    spatial_dir = os.path.join(plots_dir, "spatial")
    temporal_dir = os.path.join(plots_dir, "temporal")
    rep_dir = os.path.join(plots_dir, "representation")
    for d in [audit_dir, results_dir, spatial_dir, temporal_dir, rep_dir]:
        os.makedirs(d, exist_ok=True)
    write_plot_schema_manifest(os.path.join(plots_dir, "plot_schema_manifest.json"))
    plot_status = []

    # Helpers for actual plots
    def plot_dist(df, col, path, title):
        if df.empty or col not in df.columns or df[col].dropna().empty:
            return
        plt.figure(figsize=(6,4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(title)
        plt.xlabel(f"{col} (mm)" if "mm" in col else col)
        plt.ylabel("Count")
        plt.savefig(path)
        plt.close()

    # We will load relevant dataframes here
    per_trial_path = os.path.join(base_out, "14_metrics", "per_trial_metrics.csv")
    df_metrics = pd.read_csv(per_trial_path) if os.path.exists(per_trial_path) else pd.DataFrame()

    h1_path = os.path.join(base_out, "15_hypotheses", "H1", "h1_trial_pairs.csv")
    df_h1 = pd.read_csv(h1_path) if os.path.exists(h1_path) else pd.DataFrame()
    h1_validation = None
    h1_diagnostic = pd.DataFrame()
    h1_eligible = pd.DataFrame()
    if os.path.exists(h1_path):
        h1_validation = validate_plot_input("h1_onset_vs_crosscorr_scatter", df_h1, dev_mode=getattr(cfg, "dev_mode", False))
        if h1_validation["status"] == "SKIPPED":
            for figure_id in (
                "h1_onset_vs_crosscorr_scatter",
                "h1_delta_residual_boxplot",
                "h1_peak_confidence_histogram",
                "h1_ambiguous_cases_barplot",
            ):
                skipped = validate_plot_input(
                    figure_id, df_h1, dev_mode=getattr(cfg, "dev_mode", False)
                )
                plot_status.append({**skipped, "source_table": h1_path})
            print("Skipping H1 comparison plots: no eligible rows in DEV smoke input.")
        else:
            # Validate every H1 figure before producing any of them.  The
            # eligibility decision remains the persisted producer flag.
            for figure_id in (
                "h1_delta_residual_boxplot",
                "h1_peak_confidence_histogram",
                "h1_ambiguous_cases_barplot",
            ):
                validate_plot_input(figure_id, df_h1, dev_mode=getattr(cfg, "dev_mode", False))
            h1_diagnostic = df_h1.copy()
            h1_eligible = _valid_h1_rows(df_h1)

    # Load additional data sources for plotting
    diag_path = os.path.join(base_out, "04_quality", "stream_diagnostics.csv")
    df_diag = pd.read_csv(diag_path) if os.path.exists(diag_path) else pd.DataFrame()

    rej_path = os.path.join(base_out, "04_quality", "rejection_log.csv")
    df_rej = pd.read_csv(rej_path) if os.path.exists(rej_path) else pd.DataFrame()

    # 1. Audit plots
    if not df_diag.empty:
        plot_dist(df_diag, "ev3_eff_rate", os.path.join(audit_dir, "sampling_rate_distribution_ev3.png"), "Sampling Rate (EV3)")
        plot_dist(df_diag, "vrmt_eff_rate", os.path.join(audit_dir, "sampling_rate_distribution_vive.png"), "Sampling Rate (Vive)")
        plot_dist(df_diag, "ev3_irregularity", os.path.join(audit_dir, "timestamp_irregularity_distribution_ev3.png"), "Timestamp Irregularity (EV3)")
        plot_dist(df_diag, "vrmt_irregularity", os.path.join(audit_dir, "timestamp_irregularity_distribution_vive.png"), "Timestamp Irregularity (Vive)")

    if not df_rej.empty:
        plt.figure(figsize=(8,5))
        sns.countplot(data=df_rej, y="reason", order=df_rej["reason"].value_counts().index)
        plt.title("Rejection Reasons")
        plt.tight_layout()
        plt.savefig(os.path.join(audit_dir, "rejection_reasons.png"))
        plt.close()

        plot_dist(df_metrics, "ate_rmse_mm", os.path.join(audit_dir, "tracking_loss_distribution.png"), "Tracking Loss")
    plot_dist(df_metrics, "heading_rmse_deg", os.path.join(audit_dir, "calibration_degeneracy_distribution.png"), "Calibration Degeneracy")

    # 2. Temporal & 3. Spatial & 4. Representation Trajectory Examples
    # Pick the first aligned parquet to generate these examples
    aligned_files = [f for f in os.listdir(aligned_dir) if f.endswith(".parquet")] if os.path.exists(aligned_dir) else []
    if aligned_files:
        selected_tid = None
        if not df_metrics.empty and "rmse_mm" in df_metrics.columns:
            target = df_metrics["rmse_mm"].median()
            selected_row = df_metrics.iloc[(df_metrics["rmse_mm"] - target).abs().idxmin()]
            selected_tid = selected_row["trial_id"]
        if selected_tid:
            sample_file = next((f for f in aligned_files if selected_tid in f), None)
        else:
            sample_file = None
        sample_path = os.path.join(aligned_dir, sample_file) if sample_file else os.path.join(aligned_dir, aligned_files[0])
        df_sample = pd.read_parquet(sample_path)
        trial_label = selected_tid or os.path.splitext(os.path.basename(sample_path))[0]

        # Representative multi-panel trajectory registration figure
        if all(c in df_sample.columns for c in ["odometry_x", "odometry_y", "vrmt_aligned_x", "vrmt_aligned_y"]):
            residual = df_sample["residual_norm"] if "residual_norm" in df_sample.columns else None
            # Stage 10 consumes the producer's persisted residual series. It
            # does not recompute an aggregate scientific metric for display.
            rmse = None

            proj_files = []
            try:
                import glob as _glob
                proj_files = _glob.glob(os.path.join(base_out, "06_vive_reference", "projected_vive", f"*{trial_label}*.parquet"))
            except Exception:
                proj_files = []
            df_proj = pd.DataFrame()
            if proj_files:
                df_proj = pd.read_parquet(proj_files[0])
                has_projection = all(c in df_proj.columns for c in ["projected_x_m", "projected_y_m"])
            else:
                has_projection = False

            fig, axs = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle(f"Trajectory Registration Example: {trial_label}", fontsize=16)

            # Panel A: before registration
            ax = axs[0,0]
            ax.plot(df_sample["odometry_x"], df_sample["odometry_y"], label="EV3 Odometry", color="blue", alpha=0.8)
            if has_projection:
                ax.plot(df_proj["projected_x_m"], df_proj["projected_y_m"], label="Vive Projected Unaligned", color="red", alpha=0.7)
            ax.set_title("(A) Before Registration")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.legend()
            ax.axis("equal")

            # Panel B: after registration
            ax = axs[0,1]
            ax.plot(df_sample["odometry_x"], df_sample["odometry_y"], label="EV3 Odometry", color="blue", alpha=0.8)
            ax.plot(df_sample["vrmt_aligned_x"], df_sample["vrmt_aligned_y"], label="Vive Aligned", color="green", alpha=0.8)
            ax.set_title("(B) After Registration")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.legend()
            ax.axis("equal")

            # Panel C: residual over time
            ax = axs[1,0]
            if "t_common_s" in df_sample.columns and residual is not None:
                ax.plot(df_sample["t_common_s"], residual, label="Residual (Euclidean)", color="purple")
                ax.set_xlabel("Common Time (s)")
                ax.set_ylabel("Residual (m)")
                ax.set_title("(C) Residual over Time")
                ax.legend()
            else:
                ax.text(0.5, 0.5, "No common time available", ha='center', va='center')
                ax.set_title("(C) Residual over Time")
                ax.axis("off")

            # Panel D: zoomed segment
            ax = axs[1,1]
            if "t_common_s" in df_sample.columns and residual is not None:
                duration = df_sample["t_common_s"].max() - df_sample["t_common_s"].min()
                zoom_mask = (df_sample["t_common_s"] >= df_sample["t_common_s"].min() + duration * 0.25) & (df_sample["t_common_s"] <= df_sample["t_common_s"].min() + duration * 0.45)
                if zoom_mask.any():
                    ax.plot(df_sample.loc[zoom_mask, "odometry_x"], df_sample.loc[zoom_mask, "odometry_y"], label="EV3 Odometry", color="blue", alpha=0.8)
                    ax.plot(df_sample.loc[zoom_mask, "vrmt_aligned_x"], df_sample.loc[zoom_mask, "vrmt_aligned_y"], label="Vive Aligned", color="green", alpha=0.8)
                    ax.set_title("(D) Zoomed Segment")
                    ax.set_xlabel("X (m)")
                    ax.set_ylabel("Y (m)")
                    ax.legend()
                    ax.axis("equal")
                else:
                    ax.text(0.5, 0.5, "No zoom segment available", ha='center', va='center')
                    ax.axis("off")
            else:
                ax.text(0.5, 0.5, "No zoom segment available", ha='center', va='center')
                ax.axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(spatial_dir, "trajectory_after_registration_example.png"))
            plt.close()

        # If needed, also save a simpler before/after example file for diagnostics
        if all(c in df_sample.columns for c in ["odometry_x", "odometry_y"]):
            plt.figure(figsize=(6,6))
            plt.plot(df_sample["odometry_x"], df_sample["odometry_y"], label="EV3 (Odom)", alpha=0.8)
            if has_projection:
                plt.plot(df_proj["projected_x_m"], df_proj["projected_y_m"], label="Vive Projected Unaligned", alpha=0.8)
            plt.legend()
            plt.title("Trajectory Before Registration")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.axis("equal")
            plt.savefig(os.path.join(spatial_dir, "trajectory_before_registration_example.png"))
            plt.close()

            plt.figure(figsize=(8,4))
            if "t_common_s" in df_sample.columns:
                t = df_sample["t_common_s"]
                plt.plot(t, df_sample["vrmt_aligned_x"], label="CT Spline Fit X", color='orange')
                plt.scatter(t[::10], df_sample["vrmt_aligned_x"].iloc[::10], label="Discrete Samples", color='blue', s=10)
                plt.title("Continuous-Time Reference Fit")
                plt.xlabel("Time (s)")
                plt.ylabel("Aligned X (m)")
                plt.legend()
                plt.savefig(os.path.join(rep_dir, "ct_reference_fit_example.png"))
            plt.close()

    # Temporal distributions from df_metrics
    plot_dist(df_metrics, "ate_rmse_mm", os.path.join(spatial_dir, "spatial_residual_distribution.png"), "Spatial Residuals (ATE RMSE)")

    # 6. Results

    # Generate H1 specific plots inside 15_hypotheses/H1/h1_plots
    h1_plots_dir = os.path.join(base_out, "15_hypotheses", "H1", "h1_plots")
    os.makedirs(h1_plots_dir, exist_ok=True)

    if not h1_eligible.empty:
        # h1_onset_vs_crosscorr_scatter.png
        plt.figure(figsize=(6,4))
        sns.scatterplot(data=h1_eligible, x="rmse_onset_mm", y="rmse_refined_mm")
        max_val = max(h1_eligible["rmse_onset_mm"].max(), h1_eligible["rmse_refined_mm"].max())
        plt.plot([0, max_val], [0, max_val], 'r--')
        plt.title("H1 Onset vs Crosscorr")
        plt.savefig(os.path.join(h1_plots_dir, "h1_onset_vs_crosscorr_scatter.png"))
        plt.close()

        # h1_delta_residual_boxplot.png
        plt.figure(figsize=(6,4))
        sns.boxplot(data=h1_eligible, y="delta_rmse_mm")
        plt.title("H1 Delta Residual (RMSE)")
        plt.savefig(os.path.join(h1_plots_dir, "h1_delta_residual_boxplot.png"))
        plt.close()

        # h1_peak_confidence_histogram.png
        plt.figure(figsize=(6,4))
        confidence = pd.to_numeric(h1_eligible["crosscorr_peak_confidence_ratio"], errors="coerce")
        finite_confidence = confidence[np.isfinite(confidence)]
        if finite_confidence.empty:
            plot_status.append({
                "figure_id": "h1_peak_confidence_histogram",
                "status": "SKIPPED",
                "reason": "no_finite_confidence_values",
                "source_table": h1_path,
            })
            plt.close()
            print("Skipping H1 confidence histogram: no finite confidence values.")
        else:
            sns.histplot(finite_confidence, kde=True)
            plt.title("H1 Peak Confidence")
            plt.savefig(os.path.join(h1_plots_dir, "h1_peak_confidence_histogram.png"))
            plt.close()

        # h1_ambiguous_cases_barplot.png
        plt.figure(figsize=(6,4))
        sns.countplot(data=h1_diagnostic, x="crosscorr_accepted")
        plt.title("H1 Ambiguous Cases (Accepted vs Rejected)")
        plt.savefig(os.path.join(h1_plots_dir, "h1_ambiguous_cases_barplot.png"))
        plt.close()
    elif h1_validation is None:
        print("Skipping H1 plots because h1_trial_pairs.csv is missing or lacks required columns.")

    # Plot H2 if h2_trial_comparison.csv exists
    h2_path = os.path.join(base_out, "15_hypotheses", "H2", "h2_trial_comparison.csv")
    if os.path.exists(h2_path):
        h2_source = pd.read_csv(h2_path)
        h2_validation = validate_plot_input("h2_method_rmse", h2_source, dev_mode=getattr(cfg, "dev_mode", False))
        if h2_validation["status"] == "SKIPPED":
            plot_status.append({**h2_validation, "source_table": h2_path})
            print("Skipping H2 plots: no eligible rows in DEV smoke input.")
            df_h2 = pd.DataFrame()
        else:
            df_h2 = _valid_hypothesis_rows(h2_source)
            h2_plots_dir = os.path.join(base_out, "15_hypotheses", "H2", "h2_plots")
            os.makedirs(h2_plots_dir, exist_ok=True)
        if not df_h2.empty:
            h2_plot_ids = (
                "h2_method_boxplot",
                "h2_method_violin_or_histogram",
                "h2_ct_vs_discrete_scatter",
                "h2_error_vs_timestamp_irregularity",
                "h2_phase_specific_method_comparison",
            )
            h2_decisions = {
                figure_id: h2_plot_applicability(h2_source, figure_id)
                for figure_id in h2_plot_ids
            }
            for figure_id in h2_plot_ids:
                _enforce_h2_decision(h2_decisions[figure_id])
            for figure_id in ("h2_method_boxplot", "h2_method_violin_or_histogram"):
                if h2_decisions[figure_id]["applicability_status"] != "APPLICABLE":
                    raise PlotSchemaError(
                        f"{figure_id}: required canonical H2 plot is not applicable"
                    )

            # 1. Boxplot
            plt.figure(figsize=(8,6))
            sns.boxplot(data=df_h2, x="method", y="rmse_mm", color="lightgray", fliersize=0)
            sns.stripplot(data=df_h2, x="method", y="rmse_mm", alpha=0.5, jitter=True)
            plt.title("H2: Continuous-Time vs Discrete Grid Representations")
            plt.xlabel("Representation Method")
            plt.ylabel("RMSE (mm)")
            plt.savefig(os.path.join(h2_plots_dir, "h2_method_boxplot.png"))
            plt.close()
            plot_status.append(_h2_status_row(h2_decisions["h2_method_boxplot"], h2_path, "READY"))

            # 2. Violin or Histogram
            plt.figure(figsize=(8,6))
            sns.violinplot(data=df_h2, x="method", y="rmse_mm", inner="quartile")
            plt.title("H2: Method Violin Distribution")
            plt.savefig(os.path.join(h2_plots_dir, "h2_method_violin_or_histogram.png"))
            plt.close()
            plot_status.append(_h2_status_row(h2_decisions["h2_method_violin_or_histogram"], h2_path, "READY"))

            # 3. CT vs Discrete Scatter
            scatter_decision = h2_decisions["h2_ct_vs_discrete_scatter"]
            if scatter_decision["applicability_status"] == "NOT_APPLICABLE":
                plot_status.append(_h2_status_row(scatter_decision, h2_path, "SKIPPED"))
            else:
                df_ct = df_h2[df_h2["method"] == "continuous_time_query"].set_index("trial_id")
                df_lin = df_h2[df_h2["method"] == "linear"].set_index("trial_id")
                common_trials = df_ct.index.intersection(df_lin.index)
                plt.figure(figsize=(6,6))
                plt.scatter(df_lin.loc[common_trials, "rmse_mm"], df_ct.loc[common_trials, "rmse_mm"], alpha=0.6)
                max_val = max(df_lin.loc[common_trials, "rmse_mm"].max(), df_ct.loc[common_trials, "rmse_mm"].max())
                plt.plot([0, max_val], [0, max_val], 'r--')
                plt.xlabel("Linear Resampling RMSE (mm)")
                plt.ylabel("Continuous-Time Query RMSE (mm)")
                plt.title("H2: CT vs Discrete Representation")
                plt.savefig(os.path.join(h2_plots_dir, "h2_ct_vs_discrete_scatter.png"))
                plt.close()
                plot_status.append(_h2_status_row(scatter_decision, h2_path, "READY"))

            # 4. Error vs Timestamp Irregularity
            irregularity_decision = h2_decisions["h2_error_vs_timestamp_irregularity"]
            if irregularity_decision["applicability_status"] == "NOT_APPLICABLE":
                plot_status.append(_h2_status_row(irregularity_decision, h2_path, "SKIPPED"))
            else:
                plt.figure(figsize=(8,6))
                sns.scatterplot(data=df_h2, x="timestamp_irregularity", y="rmse_mm", hue="method")
                plt.title("H2: Error vs Timestamp Irregularity")
                plt.savefig(os.path.join(h2_plots_dir, "h2_error_vs_timestamp_irregularity.png"))
                plt.close()
                plot_status.append(_h2_status_row(irregularity_decision, h2_path, "READY"))

            # 5. Phase specific
            df_phase = df_h2[df_h2["phase"].astype(str).str.lower() != "all"]
            phase_decision = h2_decisions["h2_phase_specific_method_comparison"]
            if phase_decision["applicability_status"] == "NOT_APPLICABLE":
                plot_status.append(_h2_status_row(phase_decision, h2_path, "SKIPPED"))
            else:
                plt.figure(figsize=(10,6))
                sns.barplot(data=df_phase, x="method", y="rmse_mm", hue="phase")
                plt.title("H2: Phase-specific Method Performance")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(h2_plots_dir, "h2_phase_specific_method_comparison.png"))
                plt.close()
                plot_status.append(_h2_status_row(phase_decision, h2_path, "READY"))

    # Plot H3 if h3_calibration_mode_comparison.csv exists
    h3_path = os.path.join(base_out, "15_hypotheses", "H3", "h3_calibration_mode_comparison.csv")
    if os.path.exists(h3_path):
        df_h3 = pd.read_csv(h3_path)
        h3_validation = validate_plot_input(
            "h3_calibration_vs_evaluation_residuals",
            df_h3,
            dev_mode=getattr(cfg, "dev_mode", False),
        )
        if h3_validation["status"] == "SKIPPED":
            plot_status.append({**h3_validation, "source_table": h3_path})
            print("Skipping H3 plots: no eligible rows in DEV smoke input.")
            df_h3 = pd.DataFrame()
        else:
            df_h3 = _valid_hypothesis_rows(df_h3)
            h3_plots_dir = os.path.join(base_out, "15_hypotheses", "H3", "h3_plots")
            os.makedirs(h3_plots_dir, exist_ok=True)
        if not df_h3.empty:
                # 1. Boxplot (reusing generic for residuals)
                plt.figure(figsize=(8,6))
                sns.boxplot(data=df_h3, x="calibration_mode", y="heldout_rmse_mm", color="lightgray", fliersize=0)
                sns.stripplot(data=df_h3, x="calibration_mode", y="heldout_rmse_mm", alpha=0.5, jitter=True)
                plt.title("H3: Split vs Full Calibration")
                plt.xlabel("Calibration Mode")
                plt.ylabel("RMSE (mm)")
                plt.savefig(os.path.join(h3_plots_dir, "h3_calibration_vs_evaluation_residuals.png"))
                plt.close()

                df_split = df_h3[df_h3["calibration_mode"] == "calibration_evaluation_split"].set_index("trial_id")
                df_full = df_h3[df_h3["calibration_mode"] == "full_trajectory_calibration"].set_index("trial_id")
                common_trials_h3 = df_split.index.intersection(df_full.index)

                if len(common_trials_h3) > 0:
                    # 2. Scatter
                    plt.figure(figsize=(6,6))
                    plt.scatter(df_full.loc[common_trials_h3, "heldout_rmse_mm"], df_split.loc[common_trials_h3, "heldout_rmse_mm"], alpha=0.6)
                    max_val_h3 = max(df_full.loc[common_trials_h3, "heldout_rmse_mm"].max(), df_split.loc[common_trials_h3, "heldout_rmse_mm"].max())
                    plt.plot([0, max_val_h3], [0, max_val_h3], 'r--')
                    plt.xlabel("Full Calibration RMSE (mm)")
                    plt.ylabel("Split Calibration RMSE (mm)")
                    plt.title("H3: Split vs Full Calibration Residuals")
                    plt.savefig(os.path.join(h3_plots_dir, "h3_split_vs_full_scatter.png"))
                    plt.close()

                    # 3. Residual difference histogram
                    res_diff = df_split.loc[common_trials_h3, "heldout_rmse_mm"] - df_full.loc[common_trials_h3, "heldout_rmse_mm"]
                    plt.figure(figsize=(8,6))
                    sns.histplot(res_diff, kde=True, bins=20)
                    plt.title("H3: Residual Difference (Split - Full)")
                    plt.xlabel("Difference in RMSE (mm)")
                    plt.savefig(os.path.join(h3_plots_dir, "h3_residual_difference_histogram.png"))
                    plt.close()

                # 4. Example Trial
                plt.figure(figsize=(8,6))
                sns.barplot(data=df_h3.head(2), x="calibration_mode", y="heldout_rmse_mm")
                plt.title("H3: Example Trial Full vs Split Fit")
                plt.savefig(os.path.join(h3_plots_dir, "h3_example_trial_full_fit_vs_split_fit.png"))
                plt.close()

    plot_dist(df_metrics, "rmse_mm", os.path.join(results_dir, "residual_by_configuration.png"), "Residual by Config")
    plot_dist(df_metrics, "ate_rmse_mm", os.path.join(results_dir, "ate_by_configuration.png"), "ATE by Config")
    plot_dist(df_metrics, "rpe_rmse_mm", os.path.join(results_dir, "rpe_by_configuration.png"), "RPE by Config")

    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
        if not df_metrics.empty:
            plot_metrics_boxplot(df_metrics, "ate_rmse", os.path.join(results_dir, "ate_rmse_boxplot.png"))
            plot_metrics_boxplot(df_metrics, "rpe_rmse", os.path.join(results_dir, "rpe_rmse_boxplot.png"))

    generated_plots = []
    pd.DataFrame(plot_status, columns=[
        "figure_id", "status", "reason", "applicability_status", "applicability_reason",
        "required_inputs", "mode_behavior", "source_table",
    ]).to_csv(
        os.path.join(plots_dir, "plot_status.csv"), index=False
    )
    prov.record_stage(
        stage_name="10_generate_plots",
        config_id=cfg.config_id,
        input_files=[metrics_path] if os.path.exists(metrics_path) else [],
        output_files=generated_plots,
        additional_meta={"plots_dir": plots_dir}
    )
    print("Plots generated.")

if __name__ == "__main__":
    main()
