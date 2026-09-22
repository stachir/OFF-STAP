"""
paper_figures/plot_helpers.py

Standalone plotting helpers copied from `src/plots.py` for interactive styling.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

__all__ = [
    'plot_spatial_alignment',
    'plot_metrics_boxplot',
    'plot_speed_comparison'
]


def plot_dist(df: pd.DataFrame, col: str, path: str, title: str = None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if title is None:
        title = col
    if df is None or df.empty or col not in df.columns or df[col].dropna().empty:
        return
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_trajectory_registration_example(df_sample: pd.DataFrame, out_path: str, df_proj: pd.DataFrame = None, trial_label: str = None):
    import matplotlib.pyplot as plt
    import numpy as np

    if df_sample is None or df_sample.empty:
        return

    if trial_label is None:
        trial_label = df_sample['trial_id'].iloc[0] if 'trial_id' in df_sample.columns else 'example'

    residual = None
    if all(c in df_sample.columns for c in ['odometry_x', 'odometry_y', 'vrmt_aligned_x', 'vrmt_aligned_y']):
        residual = np.sqrt((df_sample['odometry_x'] - df_sample['vrmt_aligned_x'])**2 + (df_sample['odometry_y'] - df_sample['vrmt_aligned_y'])**2)
        rmse = np.sqrt((residual**2).mean())
    else:
        rmse = 0.0

    has_projection = df_proj is not None and not df_proj.empty and all(c in df_proj.columns for c in ['projected_x_m', 'projected_y_m'])

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Trajectory Registration Example: {trial_label}\nRMSE = {rmse:.2f} m", fontsize=16)

    # Panel A: before registration
    ax = axs[0,0]
    if 'odometry_x' in df_sample.columns and 'odometry_y' in df_sample.columns:
        ax.plot(df_sample['odometry_x'], df_sample['odometry_y'], label='EV3 Odometry', color='blue', alpha=0.8)
    if has_projection:
        ax.plot(df_proj['projected_x_m'], df_proj['projected_y_m'], label='Vive Projected Unaligned', color='red', alpha=0.7)
    ax.set_title('(A) Before Registration')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.axis('equal')

    # Panel B: after registration
    ax = axs[0,1]
    if 'odometry_x' in df_sample.columns and 'odometry_y' in df_sample.columns:
        ax.plot(df_sample['odometry_x'], df_sample['odometry_y'], label='EV3 Odometry', color='blue', alpha=0.8)
    if 'vrmt_aligned_x' in df_sample.columns and 'vrmt_aligned_y' in df_sample.columns:
        ax.plot(df_sample['vrmt_aligned_x'], df_sample['vrmt_aligned_y'], label='Vive Aligned', color='green', alpha=0.8)
    ax.set_title('(B) After Registration')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.axis('equal')

    # Panel C: residual over time
    ax = axs[1,0]
    if residual is not None and 't_common_s' in df_sample.columns:
        ax.plot(df_sample['t_common_s'], residual, label='Residual (Euclidean)', color='purple')
        ax.set_xlabel('Common Time (s)')
        ax.set_ylabel('Residual (m)')
        ax.set_title('(C) Residual over Time')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No common time available', ha='center', va='center')
        ax.set_title('(C) Residual over Time')
        ax.axis('off')

    # Panel D: zoomed segment
    ax = axs[1,1]
    if residual is not None and 't_common_s' in df_sample.columns:
        duration = df_sample['t_common_s'].max() - df_sample['t_common_s'].min()
        zoom_mask = (df_sample['t_common_s'] >= df_sample['t_common_s'].min() + duration * 0.25) & (df_sample['t_common_s'] <= df_sample['t_common_s'].min() + duration * 0.45)
        if zoom_mask.any():
            ax.plot(df_sample.loc[zoom_mask, 'odometry_x'], df_sample.loc[zoom_mask, 'odometry_y'], label='EV3 Odometry', color='blue', alpha=0.8)
            ax.plot(df_sample.loc[zoom_mask, 'vrmt_aligned_x'], df_sample.loc[zoom_mask, 'vrmt_aligned_y'], label='Vive Aligned', color='green', alpha=0.8)
            ax.set_title('(D) Zoomed Segment')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.legend()
            ax.axis('equal')
        else:
            ax.text(0.5, 0.5, 'No zoom segment available', ha='center', va='center')
            ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'No zoom segment available', ha='center', va='center')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_h1_plots(df_h1: pd.DataFrame, out_dir: str):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(out_dir, exist_ok=True)
    if df_h1 is None or df_h1.empty:
        return []

    statuses = []

    from offstap.core.plot_schema import validate_plot_input
    for figure_id in (
        "h1_onset_vs_crosscorr_scatter",
        "h1_delta_residual_boxplot",
        "h1_peak_confidence_histogram",
        "h1_ambiguous_cases_barplot",
    ):
        validate_plot_input(figure_id, df_h1, dev_mode=False)

    valid = df_h1["valid_for_paper"].astype(str).str.lower().isin({"true", "1", "yes", "valid"})
    h1_eligible = df_h1.loc[valid].copy()
    h1_diagnostic = df_h1.copy()

    if not h1_eligible.empty:
        plt.figure(figsize=(6,4))
        sns.scatterplot(data=h1_eligible, x="rmse_onset_mm", y="rmse_refined_mm")
        max_val = max(h1_eligible["rmse_onset_mm"].max(), h1_eligible["rmse_refined_mm"].max())
        plt.plot([0, max_val], [0, max_val], 'r--')
        plt.title("H1 Onset vs Crosscorr")
        plt.savefig(os.path.join(out_dir, "h1_onset_vs_crosscorr_scatter.png"))
        plt.close()
        statuses.append({"figure_id": "h1_onset_vs_crosscorr_scatter", "status": "READY", "reason": None})

    if not h1_eligible.empty:
        plt.figure(figsize=(6,4))
        sns.boxplot(data=h1_eligible, y="delta_rmse_mm")
        plt.title("H1 Delta Residual (RMSE)")
        plt.savefig(os.path.join(out_dir, "h1_delta_residual_boxplot.png"))
        plt.close()
        statuses.append({"figure_id": "h1_delta_residual_boxplot", "status": "READY", "reason": None})

    confidence = pd.to_numeric(h1_eligible["crosscorr_peak_confidence_ratio"], errors="coerce")
    finite_confidence = confidence[np.isfinite(confidence)]
    if not finite_confidence.empty:
        plt.figure(figsize=(6,4))
        sns.histplot(finite_confidence, kde=True)
        plt.title("H1 Peak Confidence")
        plt.savefig(os.path.join(out_dir, "h1_peak_confidence_histogram.png"))
        plt.close()
        statuses.append({"figure_id": "h1_peak_confidence_histogram", "status": "READY", "reason": None})
    else:
        statuses.append({
            "figure_id": "h1_peak_confidence_histogram",
            "status": "SKIPPED",
            "reason": "no_finite_confidence_values",
        })

    if not h1_diagnostic.empty:
        plt.figure(figsize=(6,4))
        sns.countplot(data=h1_diagnostic, x="crosscorr_accepted")
        plt.title("H1 Ambiguous Cases (Accepted vs Rejected)")
        plt.savefig(os.path.join(out_dir, "h1_ambiguous_cases_barplot.png"))
        plt.close()
        statuses.append({"figure_id": "h1_ambiguous_cases_barplot", "status": "READY", "reason": None})

    return statuses


def plot_h2_plots(df_h2: pd.DataFrame, out_dir: str):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df_h2 is None or df_h2.empty:
        return
    os.makedirs(out_dir, exist_ok=True)

    if "method" in df_h2.columns and "rmse_mm" in df_h2.columns:
        plt.figure(figsize=(8,6))
        sns.boxplot(data=df_h2, x="method", y="rmse_mm", color="lightgray", fliersize=0)
        sns.stripplot(data=df_h2, x="method", y="rmse_mm", alpha=0.5, jitter=True)
        plt.title("H2: Continuous-Time vs Discrete Grid Representations")
        plt.xlabel("Representation Method")
        plt.ylabel("RMSE (mm)")
        plt.savefig(os.path.join(out_dir, "h2_method_boxplot.png"))
        plt.close()

    # violin
    if "method" in df_h2.columns and "rmse_mm" in df_h2.columns:
        plt.figure(figsize=(8,6))
        sns.violinplot(data=df_h2, x="method", y="rmse_mm", inner="quartile")
        plt.title("H2: Method Violin Distribution")
        plt.savefig(os.path.join(out_dir, "h2_method_violin_or_histogram.png"))
        plt.close()


def plot_h3_plots(df_h3: pd.DataFrame, out_dir: str):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df_h3 is None or df_h3.empty:
        return
    os.makedirs(out_dir, exist_ok=True)

    if "calibration_mode" in df_h3.columns and "heldout_rmse_mm" in df_h3.columns:
        plt.figure(figsize=(8,6))
        sns.boxplot(data=df_h3, x="calibration_mode", y="heldout_rmse_mm", color="lightgray", fliersize=0)
        sns.stripplot(data=df_h3, x="calibration_mode", y="heldout_rmse_mm", alpha=0.5, jitter=True)
        plt.title("H3: Split vs Full Calibration")
        plt.xlabel("Calibration Mode")
        plt.ylabel("RMSE (mm)")
        plt.savefig(os.path.join(out_dir, "h3_calibration_vs_evaluation_residuals.png"))
        plt.close()


def plot_design_grid(df_design: pd.DataFrame, out_path: str, row='scenario_id', col='trial_id'):
    """Try to create a grid-style visualization (counts pivot) for experimental design tables."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    if df_design is None or df_design.empty:
        return

    if row not in df_design.columns or col not in df_design.columns:
        # fallback: plot counts per row category
        if row in df_design.columns:
            counts = df_design[row].value_counts()
            plt.figure(figsize=(8, 6))
            sns.barplot(x=counts.index.astype(str), y=counts.values)
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
        return

    pivot = pd.pivot_table(df_design, index=row, columns=col, values=df_design.columns[0], aggfunc='count', fill_value=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=False, cmap='viridis')
    plt.title('Design Grid')
    plt.xlabel(col)
    plt.ylabel(row)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_heatmap(df: pd.DataFrame, out_path: str, index=None, columns=None, values=None):
    """Generic heatmap renderer.

    Tries to pivot on two categorical columns and a numeric value. Falls back to grouped mean or a simple matrix view.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    if df is None or df.empty:
        return

    # identify categorical and numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

    # override with provided names if present
    if values and values in df.columns:
        num_cols = [values]
    if index and index in df.columns:
        cat_cols = [index] + [c for c in cat_cols if c != index]
    if columns and columns in df.columns and columns not in cat_cols:
        cat_cols = cat_cols + [columns]

    plt.figure(figsize=(8, 6))
    try:
        if len(cat_cols) >= 2 and num_cols:
            pivot = df.pivot_table(index=cat_cols[0], columns=cat_cols[1], values=num_cols[0], aggfunc='mean', fill_value=np.nan)
            sns.heatmap(pivot, cmap='coolwarm', annot=False)
            plt.title(f"Heatmap: {num_cols[0]} by {cat_cols[0]} x {cat_cols[1]}")
        elif len(cat_cols) >= 1 and num_cols:
            grp = df.groupby(cat_cols[0])[num_cols[0]].mean().sort_index()
            sns.heatmap(pd.DataFrame(grp).T, cmap='coolwarm', annot=False)
            plt.title(f"Heatmap (row means): {num_cols[0]} by {cat_cols[0]}")
        elif num_cols:
            # show numeric matrix (reshape if needed)
            arr = df[num_cols].to_numpy()
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            sns.heatmap(arr, cmap='coolwarm')
            plt.title(f"Heatmap: {num_cols}")
        else:
            # nothing numeric; create presence/absence matrix for first two categorical cols
            if len(df.columns) >= 2:
                a = pd.crosstab(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str))
                sns.heatmap(a, cmap='Greys', cbar=False)
                plt.title('Presence matrix')
            else:
                plt.close()
                return
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception:
        # last-resort fallback strategies
        try:
            # simple line plot of first numeric column
            if num_cols:
                plt.figure(figsize=(6, 4))
                plt.plot(df[num_cols[0]].dropna().values)
                plt.title(num_cols[0])
                plt.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close()
                return
        except Exception:
            pass

        try:
            # fallback to simple counts per first column
            first_col = df.columns[0]
            counts = df[first_col].value_counts()
            plt.figure(figsize=(8, 6))
            sns.barplot(x=counts.index.astype(str), y=counts.values)
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception:
            # give up silently
            pass


def plot_spatial_alignment(df_aligned: pd.DataFrame, output_path: str):
    if df_aligned is None or df_aligned.empty:
        return

    has_odom = all(c in df_aligned.columns for c in ["odometry_x", "odometry_y"])
    has_vrmt = all(c in df_aligned.columns for c in ["vrmt_aligned_x", "vrmt_aligned_y"])

    if not (has_odom and has_vrmt):
        return

    plt.figure(figsize=(8, 8))
    plt.plot(df_aligned["vrmt_aligned_x"], df_aligned["vrmt_aligned_y"], label="VRMT Reference", alpha=0.7)
    plt.plot(df_aligned["odometry_x"], df_aligned["odometry_y"], label="EV3 Odometry", alpha=0.7)

    plt.title(f"Spatial Alignment: {df_aligned['trial_id'].iloc[0]}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_boxplot(
    metrics_df: pd.DataFrame,
    metric: str,
    output_path: str,
    *,
    group: str | None = None,
):
    if (
        metrics_df is None
        or metrics_df.empty
        or metric not in metrics_df.columns
        or (group is not None and group not in metrics_df.columns)
    ):
        return

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=group, y=metric, data=metrics_df) if group else sns.boxplot(y=metric, data=metrics_df)
    if group:
        sns.stripplot(x=group, y=metric, data=metrics_df, color=".3", size=4)
    else:
        sns.stripplot(y=metric, data=metrics_df, color=".3", size=4)

    plt.title(f"{metric.upper()} Distribution")
    plt.ylabel(f"{metric.upper()}")

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_speed_comparison(df_aligned: pd.DataFrame, output_path: str):
    if df_aligned is None or df_aligned.empty:
        return

    has_time = "common_time_s" in df_aligned.columns
    has_ev3_speed = "left_speed" in df_aligned.columns and "right_speed" in df_aligned.columns
    has_vrmt_pos = all(c in df_aligned.columns for c in ["vrmt_aligned_x", "vrmt_aligned_y"])

    if not (has_time and has_ev3_speed and has_vrmt_pos):
        return

    WHEEL_RADIUS = 0.028

    omega_l = np.radians(df_aligned["left_speed"].fillna(0.0))
    omega_r = np.radians(df_aligned["right_speed"].fillna(0.0))
    v_l = omega_l * WHEEL_RADIUS
    v_r = omega_r * WHEEL_RADIUS
    ev3_speed = np.abs((v_r + v_l) / 2.0)

    dt = np.diff(df_aligned["common_time_s"], prepend=df_aligned["common_time_s"].iloc[0])
    dx = np.diff(df_aligned["vrmt_aligned_x"], prepend=df_aligned["vrmt_aligned_x"].iloc[0])
    dy = np.diff(df_aligned["vrmt_aligned_y"], prepend=df_aligned["vrmt_aligned_y"].iloc[0])

    with np.errstate(divide='ignore', invalid='ignore'):
        vrmt_speed = np.sqrt(dx**2 + dy**2) / dt
        vrmt_speed[dt == 0] = 0

    vrmt_speed = pd.Series(vrmt_speed).rolling(window=5, min_periods=1, center=True).mean().values

    plt.figure(figsize=(10, 4))
    plt.plot(df_aligned["common_time_s"], ev3_speed, label="EV3 Speed", alpha=0.8)
    plt.plot(df_aligned["common_time_s"], vrmt_speed, label="VRMT Speed", alpha=0.8)

    plt.title(f"Temporal Alignment (Speed Profile): {df_aligned['trial_id'].iloc[0]}")
    plt.xlabel("Common Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
