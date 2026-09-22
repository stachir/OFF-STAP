import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
try:
    from offstap.figures import plot_helpers
except Exception:
    plot_helpers = None

from offstap.core.config import load_config
from offstap.core.persistence import persist_figure_source_table


FIGURE_SPECS = [
    {
        "figure_id": "fig_dataset_design_9x9_grid",
        "file_name": "fig_dataset_design_9x9_grid.png",
        "title": "Dataset Design",
        "source_files": ["00_experiment_design/nominal_design_matrix.csv"],
        "metric_names": "scenario_id,drive_id,route_id,trial_id",
        "units": "none",
        "analysis_set": "design",
        "plot_type": "grid",
        "valid_for_paper": True,
        "diagnostic_only": False,
        "reason_if_not_valid": "",
        "caption_draft": "Nominal 9x9 grid showing the planned experimental design."
    },
    {
        "figure_id": "fig_accepted_trials_9x9_grid",
        "file_name": "fig_accepted_trials_9x9_grid.png",
        "title": "Accepted Trials",
        "source_files": ["00_experiment_design/accepted_design_matrix.csv"],
        "metric_names": "scenario_id,drive_id,route_id,trial_id,final_acceptance_status",
        "units": "none",
        "analysis_set": "design",
        "plot_type": "grid",
        "valid_for_paper": True,
        "diagnostic_only": False,
        "reason_if_not_valid": "",
        "caption_draft": "Accepted trials after quality filtering and alignment."
    },
    {
        "figure_id": "fig_repeatability_by_scenario",
        "file_name": "fig_repeatability_by_scenario.png",
        "title": "Repeatability",
        "source_files": ["17_statistics/repeatability_per_scenario.csv"],
        "metric_names": "scenario_id,mean_rmse_mm,cv_rmse,repeatability_status",
        "units": "mm",
        "analysis_set": "repeatability",
        "plot_type": "heatmap",
        "valid_for_paper": True,
        "diagnostic_only": False,
        "reason_if_not_valid": "",
        "caption_draft": "Repeatability performance per scenario."
    },
    {
        "figure_id": "fig_alignment_cascade_example",
        "file_name": "fig_alignment_cascade_example.png",
        "title": "Alignment Cascade Example",
        "source_files": ["14_metrics/cascade_per_trial_metrics.csv"],
        "metric_names": "trial_id,stage_id,rmse_mm,stage_name",
        "units": "mm",
        "analysis_set": "cascade",
        "plot_type": "trajectory_example",
        "valid_for_paper": True,
        "diagnostic_only": False,
        "reason_if_not_valid": "",
        "caption_draft": "Example trial illustrating the alignment cascade effect."
    },
    {
        "figure_id": "fig_alignment_cascade_metric_reduction",
        "file_name": "fig_alignment_cascade_metric_reduction.png",
        "title": "Cascade Metric Reduction",
        "source_files": ["14_metrics/cascade_per_stage_summary.csv"],
        "metric_names": "stage_id,mean_rmse_mm,median_rmse_mm,rmse_reduction_mm",
        "units": "mm",
        "analysis_set": "cascade",
        "plot_type": "barplot",
        "valid_for_paper": True,
        "diagnostic_only": False,
        "reason_if_not_valid": "",
        "caption_draft": "Mean RMSE reduction across cascade stages."
    },
    {
        "figure_id": "fig_scenario_level_residual_heatmap",
        "file_name": "fig_scenario_level_residual_heatmap.png",
        "title": "Residual Heatmap",
        "source_files": ["17_statistics/scenario_level_metrics.csv"],
        "metric_names": "scenario_id,residual_mean_mm,rpe_mean_mm",
        "units": "mm",
        "analysis_set": "scenario_robustness",
        "plot_type": "heatmap",
        "valid_for_paper": True,
        "diagnostic_only": False,
        "reason_if_not_valid": "",
        "caption_draft": "Scenario-level residual mean heatmap."
    },
    {
        "figure_id": "fig_h1_current_run",
        "file_name": "fig_h1_current_run.png",
        "title": "H1 Current-run Comparison",
        "source_files": ["15_hypotheses/H1/h1_trial_pairs.csv"],
        "metric_names": "trial_id,delta_rmse_mm",
        "units": "mm",
        "analysis_set": "h1",
        "plot_type": "barplot",
        "plot_metric": "delta_rmse_mm",
        "valid_for_paper": True,
        "diagnostic_only": False,
        "reason_if_not_valid": "",
        "caption_draft": "Current-run H1 paired comparison."
    },
    {
        "figure_id": "fig_h2_current_run",
        "file_name": "fig_h2_current_run.png",
        "title": "H2 Current-run Comparison",
        "source_files": ["15_hypotheses/H2/h2_trial_comparison.csv"],
        "metric_names": "trial_id,method,rmse_mm",
        "units": "mm",
        "analysis_set": "h2",
        "plot_type": "barplot",
        "plot_metric": "rmse_mm",
        "plot_group": "method",
        "valid_for_paper": True,
        "diagnostic_only": False,
        "reason_if_not_valid": "",
        "caption_draft": "Current-run H2 representation comparison."
    },
    {
        "figure_id": "fig_h3_current_run",
        "file_name": "fig_h3_current_run.png",
        "title": "H3 Current-run Comparison",
        "source_files": ["15_hypotheses/H3/h3_calibration_mode_comparison.csv"],
        "metric_names": "trial_id,calibration_mode,heldout_rmse_mm",
        "units": "mm",
        "analysis_set": "h3",
        "plot_type": "barplot",
        "plot_metric": "heldout_rmse_mm",
        "plot_group": "calibration_mode",
        "valid_for_paper": True,
        "diagnostic_only": False,
        "reason_if_not_valid": "",
        "caption_draft": "Current-run H3 calibration-mode comparison."
    },
    {
        "figure_id": "fig_baseline_current_run",
        "file_name": "fig_baseline_current_run.png",
        "title": "Corrected Current-run Baseline Comparison",
        "source_files": ["baseline/baseline_trial_metrics.csv"],
        "metric_names": "trial_id,baseline_rmse_mm,offstap_rmse_mm,delta_rmse_mm",
        "units": "mm",
        "analysis_set": "baseline",
        "plot_type": "barplot",
        "plot_metric": "delta_rmse_mm",
        "valid_for_paper": True,
        "diagnostic_only": False,
        "reason_if_not_valid": "",
        "caption_draft": "Corrected current-run engineering baseline versus OFF-STAP."
    }
]


def load_source_data(run_dir, relative_path):
    source_path = os.path.join(run_dir, relative_path)
    if not os.path.exists(source_path):
        return None, 0
    try:
        df = pd.read_csv(source_path)
        row_count = len(df)
        return df, row_count
    except Exception:
        return None, 0


def is_placeholder_df(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return True
    # check for common placeholder string patterns
    if any((df[col].astype(str).str.lower() == 'none').all() for col in ['scenario_id', 'trial_id'] if col in df.columns):
        return True
    # numeric placeholders: all numeric values <= 0 or == -1
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        try:
            if (df[num_cols] <= 0).all().all():
                return True
        except Exception:
            pass
    return False


def generate_paper_figures():
    cfg = load_config("config/alignment_config.yaml")
    run_dir = cfg.paths.get("output_base", "outputs")
    paper_dir = os.path.join(run_dir, "paper_figures")
    os.makedirs(paper_dir, exist_ok=True)
    extracts_dir = os.path.join(paper_dir, "figure_data_extracts")
    os.makedirs(extracts_dir, exist_ok=True)

    source_plot_dir = os.path.join(run_dir, "18_plots", "final_figures")
    source_plot_dir_exists = os.path.isdir(source_plot_dir)

    manifest_rows = []
    for spec in FIGURE_SPECS:
        source_fig = os.path.join(source_plot_dir, spec["file_name"])
        target_fig = os.path.join(paper_dir, spec["file_name"])

        source_row_count = 0
        available_sources = []
        any_non_placeholder = False
        for source_rel in spec["source_files"]:
            source_df, row_count = load_source_data(run_dir, source_rel)
            source_row_count += row_count
            available_sources.append(source_rel)
            if source_df is not None:
                extract_name = f"{spec['figure_id']}_{os.path.basename(source_rel)}"
                extract_path = os.path.join(extracts_dir, extract_name)
                # write extract for manual provenance and editing
                source_df.to_csv(extract_path, index=False)
                if not is_placeholder_df(source_df):
                    any_non_placeholder = True

        # If a final figure exists, copy it and keep valid flag
        if source_plot_dir_exists and os.path.exists(source_fig):
            shutil.copy2(source_fig, target_fig)
        else:
            # Try to auto-generate figure using available non-placeholder extracts and helpers
            generated = False
            if any_non_placeholder and plot_helpers is not None:
                # find first non-placeholder extract for this figure
                for f in os.listdir(extracts_dir):
                    if f.startswith(spec['figure_id'] + '_'):
                        p = os.path.join(extracts_dir, f)
                        try:
                            df_try = pd.read_csv(p)
                        except Exception:
                            df_try = None
                        if df_try is None or is_placeholder_df(df_try):
                            continue
                        try:
                            # dispatch by plot_type
                            pt = spec.get('plot_type', '').lower()
                            if pt == 'trajectory_example':
                                # require time-series columns for trajectory plots
                                time_cols = {'t_common_s', 'odometry_x', 'odometry_y', 'vrmt_aligned_x', 'vrmt_aligned_y'}
                                if time_cols.intersection(df_try.columns):
                                    plot_helpers.plot_trajectory_registration_example(df_try, target_fig)
                                    generated = os.path.exists(target_fig)
                                else:
                                    # data unsuitable for trajectory example
                                    generated = False
                                    spec['reason_if_not_valid'] = 'Insufficient time-series data for trajectory example; source contains aggregated metrics'
                            elif pt == 'grid':
                                # design grid / heatmap for experimental design tables
                                try:
                                    plot_helpers.plot_design_grid(df_try, target_fig)
                                    generated = os.path.exists(target_fig)
                                except Exception:
                                    generated = False
                            elif pt == 'heatmap':
                                # require at least one numeric column and more than one row/group for sensible heatmap
                                num_cols = df_try.select_dtypes(include=[np.number]).columns.tolist()
                                if not num_cols or len(df_try) < 2:
                                    generated = False
                                    spec['reason_if_not_valid'] = 'Insufficient numeric or group data for heatmap generation'
                                else:
                                    try:
                                        plot_helpers.plot_heatmap(df_try, target_fig)
                                        generated = os.path.exists(target_fig)
                                    except Exception:
                                        generated = False
                            elif pt == 'barplot':
                                # Bind current-run scientific figures to their declared metric.
                                # Older generic specifications retain the numeric fallback.
                                num_cols = df_try.select_dtypes(include=[np.number]).columns.tolist()
                                metric = spec.get('plot_metric') or (num_cols[0] if num_cols else None)
                                if metric:
                                    try:
                                        group = spec.get('plot_group')
                                        if group:
                                            plot_helpers.plot_metrics_boxplot(
                                                df_try, metric, target_fig, group=group
                                            )
                                        else:
                                            plot_helpers.plot_metrics_boxplot(df_try, metric, target_fig)
                                        generated = os.path.exists(target_fig)
                                    except Exception:
                                        generated = False
                            elif pt == 'speed_comparison':
                                plot_helpers.plot_speed_comparison(df_try, target_fig)
                                generated = os.path.exists(target_fig)
                            else:
                                # fallback: try simple distribution on first numeric column
                                num_cols = df_try.select_dtypes(include=[np.number]).columns.tolist()
                                metric = num_cols[0] if num_cols else None
                                if metric:
                                    try:
                                        plot_helpers.plot_metrics_boxplot(df_try, metric, target_fig)
                                        generated = os.path.exists(target_fig)
                                    except Exception:
                                        generated = False
                        except Exception:
                            generated = False
                        if generated:
                            break

            if generated:
                spec['valid_for_paper'] = True
                spec['reason_if_not_valid'] = ''
            else:
                # Do NOT create placeholder image files. Instead mark invalid with reason.
                if any_non_placeholder:
                    spec["valid_for_paper"] = False
                    spec["reason_if_not_valid"] = "No final figure generated; non-placeholder extracts available for manual plotting"
                else:
                    spec["valid_for_paper"] = False
                    spec["reason_if_not_valid"] = "Missing source final figure or placeholder source data"

        manifest_rows.append({
            "figure_id": spec["figure_id"],
            "file_name": spec["file_name"],
            "source_files": ";".join(available_sources),
            "source_row_count": source_row_count,
            "metric_names": spec["metric_names"],
            "units": spec["units"],
            "analysis_set": spec["analysis_set"],
            "plot_type": spec["plot_type"],
            "valid_for_paper": spec["valid_for_paper"],
            "diagnostic_only": spec["diagnostic_only"],
            "reason_if_not_valid": spec["reason_if_not_valid"],
            "caption_draft": spec["caption_draft"]
        })

    manifest_df = pd.DataFrame(manifest_rows)
    # Ensure semantic consistency: valid rows must not have a reason_if_not_valid
    if 'valid_for_paper' in manifest_df.columns and 'reason_if_not_valid' in manifest_df.columns:
        mask = manifest_df['valid_for_paper'] == True
        manifest_df.loc[mask, 'reason_if_not_valid'] = ''
    manifest_path = os.path.join(paper_dir, "figure_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    # Persist the generated manifest/extract cells without regenerating or
    # reinterpreting any scientific values.
    persist_figure_source_table(run_dir, os.path.join(paper_dir, "figure_source_table.csv"))
    print(f"Generated paper figure manifest and copies in {paper_dir}")


if __name__ == "__main__":
    generate_paper_figures()
