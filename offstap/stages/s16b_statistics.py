"""
scripts/16b_generate_statistics.py

Executable script to generate the final statistical summary outputs for the pipeline.
Reads per_trial_metrics and output files from earlier stages and populates 17_statistics.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import scipy
from scipy.stats import wilcoxon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from offstap.core.config import load_config
from offstap.core.statistics import signed_rank_audit, signed_rank_delta_audit, friedman_complete_blocks

def setup_logging(cfg):
    log_file = os.path.join(cfg.paths.get("output_base", "outputs"), "pipeline_stage_log.csv")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return log_file

def compute_summary_stats(df, group_by_col):
    """Computes summary statistics for a given dataframe grouped by a column."""
    if df.empty:
        return pd.DataFrame()

    metrics = ["rmse_mm", "ate_rmse_mm", "rpe_rmse_mm", "heading_rmse_deg"]
    summary_rows = []

    for group_val, grp in df.groupby(group_by_col):
        row = {group_by_col: group_val, "n": len(grp)}
        for m in metrics:
            if m in grp.columns:
                vals = grp[m].dropna()
                if not vals.empty:
                    row[f"{m}_mean"] = vals.mean()
                    row[f"{m}_std"] = vals.std()
                    row[f"{m}_median"] = vals.median()
                    row[f"{m}_iqr"] = vals.quantile(0.75) - vals.quantile(0.25)
                    row[f"{m}_min"] = vals.min()
                    row[f"{m}_max"] = vals.max()
                    # 95% CI roughly 1.96 * std / sqrt(n)
                    se = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0
                    row[f"{m}_ci_low"] = vals.mean() - 1.96 * se
                    row[f"{m}_ci_high"] = vals.mean() + 1.96 * se
                else:
                    row[f"{m}_mean"] = np.nan
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


_H1_BASE_COLUMNS = {"trial_id", "valid_for_paper", "delta_rmse_mm", "status"}
_H1_AUDIT_COLUMNS = {"common_support_count"}
_H1_SUPPORT_HASH_COLUMNS = ("common_support_hash", "common_support_hash_sha256")

_H3_REQUIRED_COLUMNS = {
    "trial_id", "calibration_mode", "status", "reason", "valid_for_paper",
    "partition_hash", "heldout_rmse_mm", "delta_rmse_mm",
    "common_support_count", "common_support_hash", "evaluation_support_count",
    "evaluation_support_hash", "fit_source_key_count", "fit_source_key_hash",
    "eligible_evaluation_fitting_key_count", "eligible_evaluation_fitting_key_hash",
    "split_excludes_evaluation_fitting", "full_includes_evaluation_fitting",
    "leakage_status",
}
_H3_MODES = ("calibration_evaluation_split", "full_trajectory_calibration")


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "ok", "valid"}


def _validate_h1_schema(df: pd.DataFrame) -> None:
    missing = sorted(_H1_BASE_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"H1 statistics schema contract missing required columns: {missing}")
    # Failed upstream rows may legitimately carry only the base status fields.
    # Any row advertised as valid must carry canonical common-support evidence;
    # arm RMSEs are optional descriptive provenance and are checked when both
    # are present.  No residual_* aliases are accepted.
    has_valid = df["valid_for_paper"].map(_as_bool).any()
    if has_valid:
        missing = sorted(_H1_AUDIT_COLUMNS - set(df.columns))
        if not any(c in df.columns for c in _H1_SUPPORT_HASH_COLUMNS):
            missing.append("common_support_hash (or common_support_hash_sha256)")
        if missing:
            raise ValueError(
                "H1 statistics schema contract missing canonical audit columns "
                f"for valid rows: {missing}"
            )


def _support_ok(row: pd.Series) -> bool:
    try:
        count = float(row.get("common_support_count", np.nan))
    except (TypeError, ValueError):
        return False
    support_hash = ""
    for column in _H1_SUPPORT_HASH_COLUMNS:
        if column in row.index and str(row.get(column, "")).strip().lower() not in {"", "nan", "none"}:
            support_hash = str(row.get(column)).strip()
            break
    return bool(np.isfinite(count) and count > 0 and support_hash and support_hash.lower() not in {"nan", "none"})


def _upstream_reason(value) -> bool:
    text = str(value or "").lower()
    markers = (
        "h1_arm_fit_failed", "missing_persisted", "partition", "source_row", "source_identity",
        "support_contract", "spatial", "temporal", "clock", "missing_cleaned",
        "missing_reference", "invalid_reference", "fit_failed", "invalid_fit", "h3_arm",
        "invalid_eligible_evaluation_fitting_keys", "invalid_evaluation_duration",
    )
    return any(marker in text for marker in markers)


def _h3_schema_valid(df: pd.DataFrame) -> None:
    missing = sorted(_H3_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"H3 statistics schema contract missing required columns: {missing}")


def _h3_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "ok", "valid"}


def _h3_support(row: pd.Series) -> tuple[int, str]:
    try:
        count = int(float(row.get("common_support_count", np.nan)))
    except (TypeError, ValueError, OverflowError):
        count = 0
    support = str(row.get("common_support_hash", "")).strip()
    return count, "" if support.lower() in {"", "nan", "none"} else support


def _h3_empty_summary(analysis_set: str, status: str, reason: str) -> dict:
    return {
        "analysis_set": analysis_set, "hypothesis_id": "H3", "eligible_n": 0,
        "n_total": 0, "n_positive": 0, "n_negative": 0, "n_zero": 0,
        "W_plus": np.nan, "W_minus": np.nan, "reported_W": np.nan,
        "p_value": np.nan, "p_raw": np.nan, "p_adjusted": np.nan,
        "p_family_adjusted": np.nan, "rank_biserial": np.nan,
        "mean_difference": np.nan, "median_difference": np.nan,
        "hodges_lehmann_pseudomedian": np.nan, "zero_method": "zsplit",
        "alternative": "two-sided", "method": "auto", "continuity_correction": False,
        "status": status, "reason": reason, "scipy_version": scipy.__version__,
    }


def _h3_blocked_row(analysis_set: str, reason: str) -> tuple[dict, dict]:
    summary = _h3_empty_summary(analysis_set, "upstream_h3_defect", reason)
    row = {
        "analysis_set": analysis_set, "test": "H3", "test_name": "wilcoxon_signed_rank",
        "test_statistic": np.nan, "p_value": np.nan, "p_raw": np.nan,
        "significant": False, "n": 0, "eligible_n": 0,
        "status": "upstream_h3_defect", "reason": reason,
        "p_family_adjusted": np.nan,
        **{field: summary.get(field, np.nan) for field in ("n_total", "n_positive", "n_negative", "n_zero", "W_plus", "W_minus", "reported_W", "rank_biserial", "mean_difference", "median_difference", "hodges_lehmann_pseudomedian", "zero_method", "alternative", "method", "continuity_correction")},
    }
    return row, summary


def compute_h3_statistics(df_h3: pd.DataFrame, tier_map: dict[str, str]):
    """Consume the canonical long H3 artifact without recomputing RMSEs."""
    _h3_schema_valid(df_h3)
    frame = df_h3.copy()
    frame["trial_id"] = frame["trial_id"].astype(str)
    frame["_valid"] = frame["valid_for_paper"].map(_as_bool)
    frame["_delta"] = pd.to_numeric(frame["delta_rmse_mm"], errors="coerce")
    frame["_heldout"] = pd.to_numeric(frame["heldout_rmse_mm"], errors="coerce")
    rows, summaries, audits = [], {}, {}
    for analysis_set in ("primary_planned_405", "all_compatible_logs"):
        tiers = {"planned_405"} if analysis_set == "primary_planned_405" else {"planned_405", "extended_previous_paper"}
        selected = frame[frame["trial_id"].map(lambda t: tier_map.get(t, "unknown") in tiers)].copy()
        eligible_ids = []
        for trial_id, group in selected.groupby("trial_id", sort=True):
            if set(group["calibration_mode"].astype(str)) != set(_H3_MODES) or len(group) != 2:
                continue
            by_mode = group.set_index("calibration_mode")
            if not by_mode["_valid"].all():
                continue
            if not by_mode["status"].astype(str).str.lower().isin({"valid", "ok", "success", "accepted"}).all():
                continue
            if not np.isfinite(by_mode["_delta"].to_numpy(float)).all() or not np.isfinite(by_mode["_heldout"].to_numpy(float)).all():
                continue
            split_count, split_hash = _h3_support(by_mode.loc["calibration_evaluation_split"])
            full_count, full_hash = _h3_support(by_mode.loc["full_trajectory_calibration"])
            if split_count <= 0 or split_count != full_count or not split_hash or split_hash != full_hash:
                continue
            partition_hashes = by_mode["partition_hash"].astype(str).str.strip()
            if partition_hashes.nunique() != 1 or partition_hashes.iloc[0].lower() in {"", "nan", "none"}:
                continue
            if not np.isclose(
                float(by_mode.loc["calibration_evaluation_split", "_heldout"])
                - float(by_mode.loc["full_trajectory_calibration", "_heldout"]),
                float(by_mode["_delta"].iloc[0]), rtol=1e-7, atol=1e-9,
            ) or not np.isclose(float(by_mode["_delta"].iloc[0]), float(by_mode["_delta"].iloc[1]), rtol=1e-7, atol=1e-9):
                continue
            if not _h3_bool(by_mode.loc["calibration_evaluation_split", "split_excludes_evaluation_fitting"]):
                continue
            if not _h3_bool(by_mode.loc["full_trajectory_calibration", "full_includes_evaluation_fitting"]):
                continue
            fit_counts = pd.to_numeric(by_mode["fit_source_key_count"], errors="coerce")
            fit_hashes = by_mode["fit_source_key_hash"].astype(str).str.strip()
            split_fit_count = float(fit_counts.loc["calibration_evaluation_split"])
            full_fit_count = float(fit_counts.loc["full_trajectory_calibration"])
            if not np.isfinite(fit_counts.to_numpy(float)).all() or split_fit_count <= 0 or full_fit_count <= split_fit_count or fit_hashes.isin(["", "nan", "none"]).any():
                continue
            leakage = by_mode["leakage_status"].astype(str)
            if leakage.loc["calibration_evaluation_split"] != "split_excludes_evaluation_fitting_keys" or leakage.loc["full_trajectory_calibration"] != "intentionally_leaky_evaluation_fitting_keys_included":
                continue
            eligible_counts = pd.to_numeric(by_mode["eligible_evaluation_fitting_key_count"], errors="coerce")
            eligible_hashes = by_mode["eligible_evaluation_fitting_key_hash"].astype(str).str.strip()
            if not np.isfinite(eligible_counts.to_numpy(float)).all() or not (eligible_counts == eligible_counts.iloc[0]).all() or eligible_counts.iloc[0] <= 0 or eligible_hashes.isin(["", "nan", "none"]).any():
                continue
            eval_counts = pd.to_numeric(by_mode["evaluation_support_count"], errors="coerce")
            eval_hashes = by_mode["evaluation_support_hash"].astype(str).str.strip()
            if not np.isfinite(eval_counts.to_numpy(float)).all() or not (eval_counts == split_count).all() or eval_hashes.nunique() != 1 or eval_hashes.iloc[0] in {"", "nan", "none"} or eval_hashes.iloc[0] != split_hash:
                continue
            eligible_ids.append(trial_id)
        eligible = selected[selected["trial_id"].isin(eligible_ids)].drop_duplicates("trial_id").sort_values("trial_id")
        deltas = eligible["_delta"].to_numpy(float)
        if len(eligible):
            audit_rows, audit_summary = signed_rank_delta_audit(deltas, eligible["trial_id"].to_numpy())
            audits[analysis_set] = audit_rows
            summary = {"analysis_set": analysis_set, "hypothesis_id": "H3", **audit_summary, "eligible_n": int(len(eligible)), "scipy_version": scipy.__version__}
            status, reason = audit_summary["status"], ""
            test_stat, p_val = audit_summary["reported_W"], audit_summary["p_value"]
        else:
            audits[analysis_set] = pd.DataFrame()
            reasons = [str(x) for x in selected.get("reason", pd.Series(dtype=object)).dropna() if _upstream_reason(x) or "invalid_temporal_status" in str(x)]
            if reasons:
                status, reason = "upstream_h3_defect", "; ".join(dict.fromkeys(reasons))
            else:
                status, reason = "insufficient_or_no_eligible_rows", "no_rows_passed_h3_pair_support_and_membership_contract"
            summary = _h3_empty_summary(analysis_set, status, reason)
            test_stat, p_val = np.nan, np.nan
        summaries[analysis_set] = summary
        rows.append({
            "analysis_set": analysis_set, "test": "H3", "test_name": "wilcoxon_signed_rank",
            "test_statistic": test_stat, "p_value": p_val, "p_raw": summary.get("p_raw", p_val),
            "significant": bool(np.isfinite(p_val) and p_val < 0.05), "n": int(len(eligible)),
            "eligible_n": int(len(eligible)), "status": status, "reason": reason,
            "p_family_adjusted": summary.get("p_family_adjusted", np.nan),
            **{field: summary.get(field, np.nan) for field in ("n_total", "n_positive", "n_negative", "n_zero", "W_plus", "W_minus", "reported_W", "rank_biserial", "mean_difference", "median_difference", "hodges_lehmann_pseudomedian", "zero_method", "alternative", "method", "continuity_correction")},
        })
    return rows, summaries, audits


def compute_h1_statistics(df_h1: pd.DataFrame, tier_map: dict[str, str]):
    """Consume canonical H1 paired rows and return population summaries.

    This function is deliberately a pure statistics consumer: it uses the
    persisted ``delta_rmse_mm`` and support/validity fields and never rebuilds
    trajectory metrics or aliases the historical residual_* columns.
    """
    _validate_h1_schema(df_h1)
    frame = df_h1.copy()
    frame["trial_id"] = frame["trial_id"].astype(str)
    frame["_valid"] = frame["valid_for_paper"].map(_as_bool)
    frame["_delta"] = pd.to_numeric(frame["delta_rmse_mm"], errors="coerce")
    rows, summaries, audits = [], {}, {}
    for analysis_set in ("primary_planned_405", "all_compatible_logs"):
        tiers = {"planned_405"} if analysis_set == "primary_planned_405" else {"planned_405", "extended_previous_paper"}
        selected = frame[frame["trial_id"].map(lambda t: tier_map.get(t, "unknown") in tiers)].copy()
        eligible_mask = selected["_valid"] & np.isfinite(selected["_delta"])
        # A valid row always needs explicit non-empty common support.  Arm
        # values are optional; when both are persisted, verify their delta but
        # never reconstruct missing arm values from the paired delta.
        eligible_mask &= selected.apply(_support_ok, axis=1)
        status_ok = selected["status"].astype(str).str.lower().isin({"valid", "ok", "success", "accepted"})
        if "crosscorr_status" in selected.columns:
            status_ok &= selected["crosscorr_status"].astype(str).str.lower().eq("ok")
        eligible_mask &= status_ok
        if {"rmse_onset_mm", "rmse_refined_mm"}.issubset(selected.columns):
            arm_a = pd.to_numeric(selected["rmse_onset_mm"], errors="coerce")
            arm_b = pd.to_numeric(selected["rmse_refined_mm"], errors="coerce")
            delta_consistent = np.isfinite(arm_a) & np.isfinite(arm_b) & np.isclose(arm_a - arm_b, selected["_delta"], rtol=1e-7, atol=1e-9)
            eligible_mask &= delta_consistent
        eligible = selected.loc[eligible_mask].copy()
        deltas = eligible["_delta"].to_numpy(float)
        summary = {"analysis_set": analysis_set, "hypothesis_id": "H1", "eligible_n": int(len(eligible))}
        if len(eligible):
            audit_rows, audit_summary = signed_rank_delta_audit(deltas, eligible["trial_id"].to_numpy())
            audits[analysis_set] = audit_rows
            summary.update(audit_summary)
            summary["scipy_version"] = scipy.__version__
            status = audit_summary["status"]
            reason = ""
            test_stat, p_val = audit_summary["reported_W"], audit_summary["p_value"]
        else:
            audits[analysis_set] = pd.DataFrame()
            defect_reasons = selected.loc[selected["_valid"].eq(False), "reason"] if "reason" in selected.columns else pd.Series(dtype=object)
            defect_reasons = [str(x) for x in defect_reasons.dropna() if _upstream_reason(x)]
            if defect_reasons:
                status, reason = "upstream_h1_defect", "; ".join(dict.fromkeys(defect_reasons))
            else:
                status, reason = "insufficient_or_no_eligible_rows", "no_rows_passed_validity_finite_delta_and_common_support"
            summary.update({"n_total": 0, "n_positive": 0, "n_negative": 0, "n_zero": 0,
                            "W_plus": np.nan, "W_minus": np.nan, "reported_W": np.nan,
                            "p_value": np.nan, "p_raw": np.nan, "p_adjusted": np.nan,
                            "p_family_adjusted": np.nan, "rank_biserial": np.nan,
                            "mean_difference": np.nan, "median_difference": np.nan,
                            "hodges_lehmann_pseudomedian": np.nan, "zero_method": "zsplit",
                            "alternative": "two-sided", "method": "auto", "continuity_correction": False,
                            "status": status, "reason": reason, "scipy_version": scipy.__version__})
            test_stat, p_val = np.nan, np.nan
        summary["eligible_n"] = int(len(eligible))
        summaries[analysis_set] = summary
        row = {"analysis_set": analysis_set, "test": "H1", "test_name": "wilcoxon_signed_rank",
               "test_statistic": test_stat, "p_value": p_val, "p_raw": summary.get("p_raw", p_val),
               "significant": bool(np.isfinite(p_val) and p_val < 0.05), "n": int(len(eligible)),
               "eligible_n": int(len(eligible)), "status": status, "reason": reason,
               "p_family_adjusted": summary.get("p_family_adjusted", np.nan)}
        for field in ("n_total", "n_positive", "n_negative", "n_zero", "W_plus", "W_minus", "reported_W", "rank_biserial", "mean_difference", "median_difference", "hodges_lehmann_pseudomedian", "zero_method", "alternative", "method", "continuity_correction"):
            row[field] = summary.get(field, np.nan)
        rows.append(row)
    return rows, summaries, audits


def generate_h1_statistics(base_out: str):
    """Run only the Stage16b H1 consumer against one explicit output run."""
    h1_path = os.path.join(base_out, "15_hypotheses", "H1", "h1_trial_pairs.csv")
    if not os.path.exists(h1_path):
        raise FileNotFoundError(f"Missing canonical H1 artifact: {h1_path}")
    df_h1 = pd.read_csv(h1_path)
    pair_path = os.path.join(base_out, "01_manifest", "pair_manifest.csv")
    manifest = pd.read_csv(pair_path) if os.path.exists(pair_path) else pd.DataFrame(columns=["trial_id", "dataset_tier"])
    tier_map = dict(zip(manifest.get("trial_id", pd.Series(dtype=str)).astype(str), manifest.get("dataset_tier", pd.Series(dtype=str))))
    ht_rows, summaries, audits = compute_h1_statistics(df_h1, tier_map)
    stats_dir = os.path.join(base_out, "17_statistics")
    audit_dir = os.path.join(stats_dir, "signed_rank_audits")
    os.makedirs(audit_dir, exist_ok=True)
    for name, audit in audits.items():
        audit.to_csv(os.path.join(audit_dir, f"{name}_H1_rows.csv"), index=False)
    out = pd.DataFrame(ht_rows)
    out["p_value_holm"] = np.nan
    out["p_family_adjusted"] = np.nan
    planned = out.index[(out["analysis_set"] == "primary_planned_405") & out["p_value"].notna()]
    if len(planned):
        vals = out.loc[planned, "p_value"].to_numpy(float)
        # Frozen Holm step-down (with one H1 row this equals the raw p-value,
        # but retaining the algorithm avoids silently changing the family when
        # additional planned hypotheses are present).
        order = np.argsort(vals, kind="mergesort")
        adjusted = np.empty(len(vals), dtype=float)
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (len(vals) - rank) * float(vals[idx]))
            adjusted[idx] = min(1.0, running)
        out.loc[planned, "p_value_holm"] = adjusted
        out.loc[planned, "p_family_adjusted"] = adjusted
    out.to_csv(os.path.join(stats_dir, "hypothesis_test_summary.csv"), index=False)
    pd.DataFrame(list(summaries.values())).to_csv(os.path.join(stats_dir, "signed_rank_audit_summary.csv"), index=False)
    return summaries

def main():
    cfg = load_config()
    setup_logging(cfg)

    base_out = cfg.paths.get("output_base", "outputs")
    stats_dir = os.path.join(base_out, "17_statistics")
    os.makedirs(stats_dir, exist_ok=True)

    metrics_path = os.path.join(base_out, "14_metrics", "per_trial_metrics.csv")
    if not os.path.exists(metrics_path):
        logging.warning("per_trial_metrics.csv not found, skipping statistics generation.")
        return

    df_metrics = pd.read_csv(metrics_path)

    if df_metrics.empty:
        logging.warning("per_trial_metrics.csv is empty.")
        return

    df_metrics["drive_id"] = df_metrics["trial_id"].str.split('_').str[0]
    df_metrics["route_id"] = df_metrics["trial_id"].str.split('_').str[1]
    df_metrics["scenario_id"] = df_metrics["drive_id"] + "_" + df_metrics["route_id"]

    # 1. accepted_trials_summary.csv
    # For this baseline implementation, assume all in per_trial_metrics are accepted
    df_metrics["overall"] = "overall"
    df_acc = compute_summary_stats(df_metrics, "overall")
    df_acc.to_csv(os.path.join(stats_dir, "accepted_trials_summary.csv"), index=False)

    # 2. rejected_trials_summary.csv
    rej_path = os.path.join(base_out, "04_quality", "rejection_log.csv")
    if os.path.exists(rej_path):
        df_rej = pd.read_csv(rej_path)
        df_rej_summary = df_rej.groupby("reason").size().reset_index(name="count")
        df_rej_summary.to_csv(os.path.join(stats_dir, "rejected_trials_summary.csv"), index=False)
    else:
        pd.DataFrame([{"reason": "none", "count": 0}]).to_csv(os.path.join(stats_dir, "rejected_trials_summary.csv"), index=False)

    # 3. summaries by drive, route, scenario
    df_drive = compute_summary_stats(df_metrics, "drive_id")
    df_drive.to_csv(os.path.join(stats_dir, "summary_by_drive.csv"), index=False)

    df_route = compute_summary_stats(df_metrics, "route_id")
    df_route.to_csv(os.path.join(stats_dir, "summary_by_route.csv"), index=False)

    df_scen = compute_summary_stats(df_metrics, "scenario_id")
    df_scen.to_csv(os.path.join(stats_dir, "summary_by_scenario.csv"), index=False)

    # 4. summary_by_phase.csv
    phase_path = os.path.join(base_out, "14_metrics", "per_phase_metrics.csv")
    if os.path.exists(phase_path):
        df_phase = pd.read_csv(phase_path)
        df_phase_summary = compute_summary_stats(df_phase, "motion_phase")
        df_phase_summary.to_csv(os.path.join(stats_dir, "summary_by_phase.csv"), index=False)
    else:
        pd.DataFrame(columns=["motion_phase", "n", "rmse_mm_mean"]).to_csv(os.path.join(stats_dir, "summary_by_phase.csv"), index=False)

    # 5. hypothesis_test_summary.csv
    # Merge tier from pair_manifest
    pair_path = os.path.join(base_out, "01_manifest", "pair_manifest.csv")
    df_pairs = pd.read_csv(pair_path) if os.path.exists(pair_path) else pd.DataFrame(columns=["trial_id", "dataset_tier"])
    tier_map = dict(zip(df_pairs.get("trial_id", pd.Series(dtype=str)).astype(str), df_pairs.get("dataset_tier", pd.Series(dtype=str))))

    ht_rows = []
    signed_rank_summaries = []
    signed_rank_dir = os.path.join(stats_dir, "signed_rank_audits")
    os.makedirs(signed_rank_dir, exist_ok=True)

    # H1 is consumed from its canonical paired artifact.  Keep this handoff
    # independent of the H2/H3 branches below so a stale H1 schema cannot be
    # accidentally reintroduced by a per-analysis-set loop.
    h1_path = os.path.join(base_out, "15_hypotheses", "H1", "h1_trial_pairs.csv")
    if os.path.exists(h1_path):
        h1_rows, h1_summaries, h1_audits = compute_h1_statistics(pd.read_csv(h1_path), tier_map)
        ht_rows.extend(h1_rows)
        signed_rank_summaries.extend(h1_summaries.values())
        for name, audit in h1_audits.items():
            audit.to_csv(os.path.join(signed_rank_dir, f"{name}_H1_rows.csv"), index=False)

    # H3 is mandatory for the Stage 16b handoff. Missing or empty artifacts
    # are upstream evidence failures, not a legitimate zero-eligible result.
    h3_upstream_defect = False
    h3_path = os.path.join(base_out, "15_hypotheses", "H3", "h3_calibration_mode_comparison.csv")
    if not os.path.exists(h3_path):
        h3_rows = []
        h3_summaries = {}
        h3_audits = {}
        for population in ("primary_planned_405", "all_compatible_logs"):
            row, summary = _h3_blocked_row(population, "missing_canonical_h3_artifact")
            h3_rows.append(row)
            h3_summaries[population] = summary
        h3_upstream_defect = True
    else:
        df_h3 = pd.read_csv(h3_path)
        if df_h3.empty:
            h3_rows = []
            h3_summaries = {}
            h3_audits = {}
            for population in ("primary_planned_405", "all_compatible_logs"):
                row, summary = _h3_blocked_row(population, "empty_canonical_h3_artifact")
                h3_rows.append(row)
                h3_summaries[population] = summary
            h3_upstream_defect = True
        else:
            h3_rows, h3_summaries, h3_audits = compute_h3_statistics(df_h3, tier_map)
            h3_upstream_defect = any(summary.get("status") == "upstream_h3_defect" for summary in h3_summaries.values())
    ht_rows.extend(h3_rows)
    signed_rank_summaries.extend(h3_summaries.values())
    for name, audit in h3_audits.items():
        audit.to_csv(os.path.join(signed_rank_dir, f"{name}_H3_rows.csv"), index=False)
    if h3_upstream_defect:
        logging.error("H3 statistics blocked by an upstream producer defect")

    for analysis_set in ["primary_planned_405", "all_compatible_logs"]:
        def is_in_set(tid):
            tier = tier_map.get(tid, "unknown")
            if analysis_set == "primary_planned_405":
                return tier == "planned_405"
            else:
                return tier in ["planned_405", "extended_previous_paper"]

        # H2 Test: complete repeated-measure blocks only.  The four methods
        # are Vive reference representations; continuous-time rows are
        # descriptive and excluded by construction.
        h2_path = os.path.join(base_out, "15_hypotheses", "H2", "h2_trial_comparison.csv")
        if os.path.exists(h2_path):
            df_h2 = pd.read_csv(h2_path)
            if not df_h2.empty:
                df_h2_set = df_h2[df_h2["trial_id"].apply(is_in_set)]
                if "phase" in df_h2_set.columns:
                    df_h2_set = df_h2_set[df_h2_set["phase"].astype(str).eq("all")]
                try:
                    h2_result = friedman_complete_blocks(
                        df_h2_set, ["zoh", "linear", "pchip", "spline"]
                    )
                    h2_result.pairwise.to_csv(
                        os.path.join(stats_dir, f"h2_pairwise_{analysis_set}.csv"), index=False
                    )
                    ht_rows.append({
                        "analysis_set": analysis_set,
                        "test": "H2_Friedman",
                        "test_name": "friedman_complete_blocks",
                        "test_statistic": h2_result.statistic,
                        "p_value": h2_result.p_value,
                        "significant": bool(np.isfinite(h2_result.p_value) and h2_result.p_value < 0.05),
                        "n": h2_result.n,
                        "kendall_w": h2_result.kendall_w,
                        "pairwise_contrast_count": len(h2_result.pairwise),
                        "status": "ok" if h2_result.n >= 2 else "insufficient_complete_blocks",
                    })
                except (ValueError, KeyError) as exc:
                    logging.warning("H2 statistics unavailable for %s: %s", analysis_set, exc)
                    ht_rows.append({"analysis_set": analysis_set, "test": "H2_Friedman", "test_name": "friedman_complete_blocks", "test_statistic": np.nan, "p_value": np.nan, "significant": False, "n": 0, "kendall_w": np.nan, "pairwise_contrast_count": 6, "status": "no_valid_complete_blocks"})

    if not ht_rows:
        ht_rows.append({"analysis_set": "none", "test": "no_valid_hypothesis_rows", "t_stat": np.nan, "p_value": np.nan, "significant": False, "n": 0})

    df_ht = pd.DataFrame(ht_rows)
    df_ht["p_value_holm"] = np.nan
    df_ht["p_family_adjusted"] = np.nan
    planned_mask = df_ht["analysis_set"] == "primary_planned_405"
    planned = df_ht.loc[planned_mask & df_ht["p_value"].notna()].copy()
    if not planned.empty:
        ordered = planned.sort_values("p_value")
        adjusted = []
        running = 0.0
        m = len(ordered)
        for index, raw_p in enumerate(ordered["p_value"].to_numpy()):
            running = max(running, (m - index) * float(raw_p))
            adjusted.append(min(1.0, running))
        df_ht.loc[ordered.index, "p_value_holm"] = adjusted
        df_ht.loc[planned_mask, "significant"] = (
            df_ht.loc[planned_mask, "p_value_holm"] < 0.05
        )
    # Stable public alias used by downstream audit tables.
    df_ht["p_family_adjusted"] = df_ht["p_value_holm"]
    df_ht.to_csv(os.path.join(stats_dir, "hypothesis_test_summary.csv"), index=False)
    for summary in signed_rank_summaries:
        matching = df_ht[
            (df_ht["analysis_set"] == summary.get("analysis_set"))
            & (df_ht["test"].astype(str) == str(summary.get("hypothesis_id")))
        ]
        if not matching.empty:
            adjusted = matching.iloc[0].get("p_family_adjusted", np.nan)
            summary["p_adjusted"] = adjusted
            summary["p_family_adjusted"] = adjusted
    pd.DataFrame(signed_rank_summaries).to_csv(
        os.path.join(stats_dir, "signed_rank_audit_summary.csv"), index=False
    )

    decision_rows = []
    for _, row in df_ht.iterrows():
        n = int(row.get("n", 0)) if pd.notna(row.get("n", np.nan)) else 0
        raw_p_value = row.get("p_value", np.nan)
        adjusted_p = row.get("p_value_holm", np.nan)
        p_value = adjusted_p if row.get("analysis_set") == "primary_planned_405" and pd.notna(adjusted_p) else raw_p_value
        if n < 2 or pd.isna(p_value):
            decision = "inconclusive"
            interpretation = "Insufficient paired valid rows for a statistical decision."
        elif bool(row.get("significant", False)):
            decision = "supported"
            interpretation = "Paired test is significant at alpha=0.05."
        else:
            decision = "not_supported"
            interpretation = "Paired test is not significant at alpha=0.05."

        if row.get("test") == "H3" and decision == "not_supported":
            interpretation = "No statistically detectable split-versus-full residual difference on the audited common support."

        decision_rows.append({
            "analysis_set": row.get("analysis_set"),
            "hypothesis_id": row.get("test"),
            "decision": decision,
            "n": n,
            "p_value": p_value,
            "p_value_raw": raw_p_value,
            "p_value_holm": adjusted_p,
            "interpretation": interpretation,
        })

    pd.DataFrame(decision_rows).to_csv(os.path.join(stats_dir, "hypothesis_decision_summary.csv"), index=False)

    if "h3_upstream_defect" in locals() and h3_upstream_defect:
        raise RuntimeError("Stage16b blocked: upstream_h3_defect in canonical H3 artifact")

    # 6. Sensitivity Comparison
    if not df_ht.empty and "analysis_set" in df_ht.columns:
        sens_rows = []
        tests = df_ht["test"].unique()
        for t in tests:
            if t == "no_valid_hypothesis_rows": continue
            r_planned = df_ht[(df_ht["test"] == t) & (df_ht["analysis_set"] == "primary_planned_405")]
            r_extended = df_ht[(df_ht["test"] == t) & (df_ht["analysis_set"] == "all_compatible_logs")]

            p_n = r_planned["n"].iloc[0] if not r_planned.empty else 0
            e_n = r_extended["n"].iloc[0] if not r_extended.empty else 0

            p_sig = r_planned["significant"].iloc[0] if not r_planned.empty else False
            e_sig = r_extended["significant"].iloc[0] if not r_extended.empty else False

            p_pval = r_planned["p_value"].iloc[0] if not r_planned.empty else np.nan
            e_pval = r_extended["p_value"].iloc[0] if not r_extended.empty else np.nan

            conclusion_stable = (p_sig == e_sig)

            sens_rows.append({
                "hypothesis_id": t,
                "metric": "p_value",
                "planned_n": p_n,
                "extended_n": e_n,
                "planned_effect": "sig" if p_sig else "not_sig",
                "extended_effect": "sig" if e_sig else "not_sig",
                "planned_p": p_pval,
                "extended_p": e_pval,
                "conclusion_stable": conclusion_stable,
                "interpretation": "Stable" if conclusion_stable else "Sensitivity issue: conclusion differs"
            })

        if sens_rows:
            pd.DataFrame(sens_rows).to_csv(os.path.join(stats_dir, "sensitivity_planned_vs_extended.csv"), index=False)

    logging.info("Generated summary statistics in 17_statistics.")

if __name__ == "__main__":
    main()
