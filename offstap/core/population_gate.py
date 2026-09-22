"""Canonical run-scoped population gate ledger producer.

The ledger is derived only from persisted run artifacts.  It is shared by the
early Stage 14 dependency boundary and later audit consumers;
there is intentionally one implementation of the gate semantics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TIERS = {
    "planned_405": "Planned",
    "extended_previous_paper": "Extended",
    "diagnostic_or_unplanned": "Diagnostic/unplanned",
}
H2_METHODS = ["zoh", "linear", "pchip", "spline"]

GATE_FILES = {
    "quality": "04_quality/rejection_log.csv",
    "degeneracy": "08_calibration_evaluation_split/calibration_motion_degeneracy.csv",
    "temporal": "09_temporal/crosscorr_report.csv",
    "clock": "10_clock_model/clock_model_comparison.csv",
    "metrics": "14_metrics/per_trial_metrics.csv",
    "h1": "15_hypotheses/H1/h1_trial_pairs.csv",
    "h2": "15_hypotheses/H2/h2_trial_comparison.csv",
    "h3": "15_hypotheses/H3/h3_calibration_mode_comparison.csv",
    "baseline": "baseline/population_gate_membership.csv",
}


def _persist_baseline_membership(run: Path) -> None:
    """Project accepted current-run baseline rows into gate-specific evidence."""
    source = run / "baseline" / "baseline_trial_metrics.csv"
    if not source.is_file():
        return
    baseline = pd.read_csv(source)
    required = ("trial_id", "baseline_rmse_mm", "offstap_rmse_mm", "delta_rmse_mm")
    missing = [column for column in required if column not in baseline.columns]
    if missing:
        raise ValueError(f"baseline trial metrics missing gate columns: {missing}")
    membership = baseline.loc[:, required].copy()
    destination = run / GATE_FILES["baseline"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    membership.to_csv(destination, index=False)


def _truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _optional_csv(run: Path, relative: str) -> pd.DataFrame:
    path = run / relative
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


def _ids(df: pd.DataFrame, mask: pd.Series | None = None) -> set[str]:
    if df.empty or "trial_id" not in df:
        return set()
    if mask is None:
        mask = pd.Series(True, index=df.index)
    return set(df.loc[mask, "trial_id"].astype(str))


def build_trial_gate_ledger(run: Path | str) -> pd.DataFrame:
    """Return one reason-coded row for every trial at every population gate."""
    run = Path(run)
    manifest = _optional_csv(run, "01_manifest/pair_manifest.csv")
    if manifest.empty or "trial_id" not in manifest:
        raise FileNotFoundError(
            f"manifest with trial_id is required: {run / '01_manifest/pair_manifest.csv'}"
        )
    manifest = manifest.copy()
    manifest["trial_id"] = manifest["trial_id"].astype(str)
    for column, default in (("dataset_tier", "unknown"), ("analysis_set", "unknown"), ("scenario_id", "unknown")):
        if column not in manifest:
            manifest[column] = default

    artifacts = {name: _optional_csv(run, rel) for name, rel in GATE_FILES.items()}
    all_ids = set(manifest.trial_id)
    paired = _ids(manifest, manifest.get("pair_status", pd.Series("paired", index=manifest.index)).astype(str).eq("paired"))

    quality_artifact = artifacts["quality"]
    quality_path_exists = (run / GATE_FILES["quality"]).exists()
    rejected_quality = _ids(quality_artifact, pd.Series(True, index=quality_artifact.index))
    quality_ok = (paired - rejected_quality) if quality_path_exists else set()

    degen = artifacts["degeneracy"]
    degen_ok = _ids(degen, degen.get("degeneracy_status", pd.Series(dtype=str)).astype(str).str.lower().eq("ok"))
    temporal = artifacts["temporal"]
    temporal_ok = _ids(temporal, temporal.get("status", pd.Series(dtype=str)).astype(str).str.lower().isin({"ok", "success"}))
    clock = artifacts["clock"]
    clock_ok = _ids(
        clock,
        np.isfinite(pd.to_numeric(clock.get("scale_a", pd.Series(dtype=float)), errors="coerce"))
        & np.isfinite(pd.to_numeric(clock.get("offset_b", pd.Series(dtype=float)), errors="coerce"))
        & clock.get("selected_model", pd.Series(dtype=str)).astype(str).isin({"constant_offset", "affine"}),
    )
    metrics = artifacts["metrics"]
    metric_ok = _ids(
        metrics,
        _truth(metrics.get("accepted_for_hypothesis_tests", pd.Series(False, index=metrics.index)))
        & np.isfinite(pd.to_numeric(metrics.get("ate_rmse", metrics.get("rmse_mm", pd.Series(dtype=float))), errors="coerce")),
    )
    baseline = artifacts["baseline"]
    baseline_ok = _ids(
        baseline,
        np.isfinite(
            pd.to_numeric(
                baseline.get("baseline_rmse_mm", pd.Series(dtype=float)), errors="coerce"
            )
        )
        & np.isfinite(
            pd.to_numeric(
                baseline.get("offstap_rmse_mm", pd.Series(dtype=float)), errors="coerce"
            )
        )
        & np.isfinite(
            pd.to_numeric(
                baseline.get("delta_rmse_mm", pd.Series(dtype=float)), errors="coerce"
            )
        ),
    )
    h1 = artifacts["h1"]
    h1_ok = _ids(h1, _truth(h1.get("valid_for_paper", pd.Series(False, index=h1.index))))
    h2 = artifacts["h2"]
    if h2.empty:
        h2_ok = set()
    else:
        h2_all = h2[
            h2.get("phase", pd.Series("all", index=h2.index)).astype(str).eq("all")
            & h2.get("method", pd.Series(dtype=str)).astype(str).isin(H2_METHODS)
        ].copy()
        h2_all["_ok"] = h2_all.get("fit_or_resampling_status", pd.Series("success", index=h2_all.index)).astype(str).eq("success") & np.isfinite(pd.to_numeric(h2_all.get("rmse_mm", pd.Series(dtype=float)), errors="coerce"))
        h2_ok = _ids(h2_all.groupby("trial_id").filter(lambda group: set(group.method.astype(str)) == set(H2_METHODS) and len(group) == len(H2_METHODS) and bool(group._ok.all())))
    h3 = artifacts["h3"]
    if h3.empty:
        h3_ok = set()
    else:
        h3 = h3.copy()
        h3["_ok"] = _truth(h3.get("valid_for_paper", pd.Series(False, index=h3.index))) & np.isfinite(pd.to_numeric(h3.get("heldout_rmse_mm", pd.Series(dtype=float)), errors="coerce"))
        h3_ok = _ids(h3.groupby("trial_id").filter(lambda group: len(group) == 2 and group.calibration_mode.astype(str).nunique() == 2 and bool(group._ok.all())))

    gate_sets = {
        "discovered": all_ids,
        "paired": paired,
        "missing_required_stream": all_ids - paired,
        "quality_accepted": quality_ok,
        "calibration_non_degenerate": degen_ok,
        "temporal_valid": temporal_ok,
        "clock_valid": clock_ok,
        "final_OFF_STAP_valid": metric_ok,
        "baseline_valid": baseline_ok,
        "H1_eligible": h1_ok,
        "H2_all_four_arm_eligible": h2_ok,
        "H3_both_arm_eligible": h3_ok,
    }
    evidence = {name: str((run / rel).as_posix()) for name, rel in GATE_FILES.items()}
    evidence.update({
        "quality_accepted": evidence["quality"],
        "calibration_non_degenerate": evidence["degeneracy"],
        "temporal_valid": evidence["temporal"],
        "clock_valid": evidence["clock"],
        "final_OFF_STAP_valid": evidence["metrics"],
        "baseline_valid": evidence["baseline"],
        "H1_eligible": evidence["h1"],
        "H2_all_four_arm_eligible": evidence["h2"],
        "H3_both_arm_eligible": evidence["h3"],
    })
    manifest_evidence = str((run / "01_manifest/pair_manifest.csv").as_posix())
    evidence.update({"discovered": manifest_evidence, "paired": manifest_evidence, "missing_required_stream": manifest_evidence})

    rows = []
    for trial in sorted(all_ids):
        meta = manifest.loc[manifest.trial_id.eq(trial)].iloc[0]
        for gate, passed_ids in gate_sets.items():
            passed = trial in passed_ids
            reason = "passed" if passed else (
                "evidence_unavailable"
                if gate not in {"discovered", "paired", "missing_required_stream"} and not Path(evidence[gate]).exists()
                else "gate_failed"
            )
            rows.append({
                "trial_id": trial,
                "dataset_tier": meta.dataset_tier,
                "analysis_set": meta.analysis_set,
                "scenario_id": meta.scenario_id,
                "gate": gate,
                "gate_status": "passed" if passed else "failed",
                "gate_result": "passed" if passed else "failed",
                "status": "passed" if passed else "failed",
                "reason": reason,
                "evidence_path": evidence.get(gate, ""),
            })
    return pd.DataFrame(rows)


def persist_trial_gate_ledger(run: Path | str) -> tuple[Path, Path]:
    """Persist the single authoritative ledger and its aggregate table."""
    run = Path(run)
    _persist_baseline_membership(run)
    out = run / "19_tables"
    out.mkdir(parents=True, exist_ok=True)
    ledger = build_trial_gate_ledger(run)
    ledger_path = out / "authoritative_population_gate_ledger.csv"
    table_path = out / "authoritative_population_gate_table.csv"
    ledger.to_csv(ledger_path, index=False)
    passed = ledger[ledger.gate_status.eq("passed")]
    table = passed.pivot_table(index="gate", columns="dataset_tier", values="trial_id", aggfunc="nunique", fill_value=0).reset_index()
    for tier, label in TIERS.items():
        if tier not in table.columns:
            table[tier] = 0
        table.rename(columns={tier: label}, inplace=True)
    table["Total"] = table[[label for label in TIERS.values()]].sum(axis=1)
    table.to_csv(table_path, index=False)
    return ledger_path, table_path
