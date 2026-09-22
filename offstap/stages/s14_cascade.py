"""Stage 14 reporting consumer for the revision-preserving cascade.

Scientific fitting, alignment, reference querying, and metric computation are
performed upstream.  This stage reads the frozen cascade table and renders
diagnostic plots only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from offstap.core.config import load_config
from offstap.core.cascade import STAGES


ROLE_METADATA = {
    "reviewed_figure_3": {
        "scientific_role": "evidence_eligibility_population_accounting",
        "manuscript_continuity_status": "reviewed",
        "attribution_role": "none",
    },
    "S0_S4": {
        "scientific_role": "descriptive_system_level",
        "manuscript_continuity_status": "conditional",
        "attribution_role": "none",
    },
    "OFAT": {
        "scientific_role": "component_level_attribution",
        "manuscript_continuity_status": "conditional",
        "attribution_role": "authoritative",
    },
    "composite_baseline": {
        "scientific_role": "reviewed_multifactor_engineering_baseline",
        "manuscript_continuity_status": "reviewed",
        "attribution_role": "none",
    },
    "H1": {"scientific_role": "temporal_onset_vs_refinement", "attribution_role": "temporal"},
    "H2": {"scientific_role": "vive_representation_query", "attribution_role": "representation"},
    "H3": {"scientific_role": "split_vs_intentionally_leaky_full_fit", "attribution_role": "leakage"},
}

STAGE_TRANSITIONS = {"S0->S1": "SINGLE_FACTOR", "S1->S2": "SINGLE_FACTOR", "S2->S3": "SINGLE_FACTOR", "S3->S4": "MULTI_FACTOR"}


def _require_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Frozen canonical cascade table is required before Stage 14 plotting: {path}")
    table = pd.read_csv(path)
    required = {"trial_id", "stage_id", "stage_name", "rmse_mm", "metric_validity_status", "invalidity_reason", "support_count", "support_hash", "calibration_provenance", "evaluation_provenance", "temporal_provenance", "spatial_provenance", "representation_provenance", "support_provenance", "gate_status", "gate_reason", "gate_eligible", "gate_provenance_status"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Cascade table missing required columns: {missing}")
    return table


def build_stage_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Derive neutral stage summaries from valid frozen rows only."""
    valid = table[(table["metric_validity_status"].astype(str).str.lower() == "valid") & np.isfinite(pd.to_numeric(table["rmse_mm"], errors="coerce"))].copy()
    if valid.empty:
        return pd.DataFrame(columns=["stage_id", "stage_name", "mean_rmse_mm", "median_rmse_mm", "rmse_reduction_mm", "num_trials", "interpretation"])
    summary = valid.groupby(["stage_id", "stage_name"], sort=False).agg(
        mean_rmse_mm=("rmse_mm", "mean"),
        median_rmse_mm=("rmse_mm", "median"),
        num_trials=("trial_id", "nunique"),
    ).reset_index()
    # Reduction is a descriptive contrast only when adjacent valid stages are
    # both present; no direction is assumed and unavailable stages stay NaN.
    summary["rmse_reduction_mm"] = np.nan
    ordered = {row.stage_id: row.mean_rmse_mm for row in summary.itertuples()}
    for idx, stage in enumerate(STAGES[1:], start=1):
        previous = STAGES[idx - 1]["stage_id"]
        if stage["stage_id"] in ordered and previous in ordered:
            summary.loc[summary["stage_id"] == stage["stage_id"], "rmse_reduction_mm"] = ordered[previous] - ordered[stage["stage_id"]]
    summary["interpretation"] = "system-level staged comparison; no component attribution"
    return summary


def retained_boxplot_groups(table: pd.DataFrame) -> list[dict]:
    """Return finite groups; empty DEV groups are explicitly skipped."""
    groups: list[dict] = []
    for stage in STAGES:
        values = pd.to_numeric(table.loc[table["stage_id"] == stage["stage_id"], "rmse_mm"], errors="coerce")
        finite_values = values[np.isfinite(values)].astype(float).tolist()
        if not finite_values:
            continue
        groups.append({"stage_id": stage["stage_id"], "label": stage["stage_name"], "position": len(groups), "finite_values": finite_values, "n": len(finite_values), "status": "retained"})
    return groups


def _write_boxplot(table: pd.DataFrame, plot_dir: Path) -> None:
    groups = retained_boxplot_groups(table)
    if not groups:
        return
    stats = [{"label": g["label"], "whislo": float(np.min(g["finite_values"])), "q1": float(np.percentile(g["finite_values"], 25)), "med": float(np.median(g["finite_values"])), "q3": float(np.percentile(g["finite_values"], 75)), "whishi": float(np.max(g["finite_values"])), "fliers": []} for g in groups]
    positions = [g["position"] for g in groups]
    labels = [g["label"] for g in groups]
    if not (len(stats) == len(positions) == len(labels)):
        raise RuntimeError("Retained boxplot group contract violated")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bxp(stats, positions=positions, showfliers=False)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("RMSE (mm)")
    ax.set_title("System-level staged comparison")
    fig.tight_layout()
    fig.savefig(plot_dir / "cascade_stage_metric_reduction_boxplot.png")
    plt.close(fig)


def main() -> None:
    cfg = load_config("config/alignment_config.yaml")
    base_out = Path(cfg.paths.get("output_base", "outputs"))
    out_dir = base_out / "14_metrics"
    plot_dir = base_out / "18_plots" / "alignment_stages"
    table = _require_table(out_dir / "cascade_per_trial_metrics.csv")
    summary = build_stage_summary(table)
    summary.to_csv(out_dir / "cascade_per_stage_summary.csv", index=False)
    groups = retained_boxplot_groups(table)
    if not groups:
        if getattr(cfg, "dev_mode", False):
            return
        raise RuntimeError("Stage 14 has no finite retained alignment-stage groups")
    plot_dir.mkdir(parents=True, exist_ok=True)
    _write_boxplot(table, plot_dir)
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x="stage_id", y="mean_rmse_mm", data=summary, ax=ax)
        ax.set_title("System-level staged comparison")
        fig.tight_layout()
        fig.savefig(plot_dir / "cascade_stage_metric_reduction_barplot.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
