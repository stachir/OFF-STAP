"""
scripts/03_detect_quality_issues.py

Executable script to compute stream diagnostics and apply quality gates.
"""

import sys
import os
import csv
import json
import pandas as pd

# Add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offstap.core.config import load_config
from offstap.core.quality import compute_stream_diagnostics, apply_quality_gates
from offstap.core.provenance import ProvenanceTracker

def main():
    cfg = load_config()
    prov = ProvenanceTracker(cfg.paths.get("output_provenance", "outputs/provenance"))

    pair_path = os.path.join(cfg.paths.get("output_base", "outputs"), "01_manifest", "pair_manifest.csv")
    if not os.path.exists(pair_path):
        raise FileNotFoundError(f"Error: {pair_path} not found.")

    pairs = []
    with open(pair_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(row)

    out_dir_ev3 = os.path.join(cfg.paths.get("output_cleaned", "outputs/cleaned"), "ev3")
    out_dir_vrmt = os.path.join(cfg.paths.get("output_cleaned", "outputs/cleaned"), "vrmt")

    stream_diagnostics = []
    rejection_log = []
    accepted_trials = []
    extended_compatibility_report = []

    print(f"Stage 03: Detecting quality issues on {len(pairs)} streams...")

    for p in pairs:
        tid = p["trial_id"]

        # Propagate upstream failures
        if p["status"] != "paired":
            rejection_log.append({
                "trial_id": tid, "stage": "pairing",
                "reason": p["status"], "value": 0, "threshold": 0, "config_id": cfg.config_id
            })
            continue

        ev3_file = os.path.join(out_dir_ev3, f"{tid}_ev3_cleaned.parquet")
        vrmt_file = os.path.join(out_dir_vrmt, f"{tid}_vrmt_cleaned.parquet")

        if not os.path.exists(ev3_file) or not os.path.exists(vrmt_file):
            rejection_log.append({
                "trial_id": tid, "stage": "missing_cleaned",
                "reason": "missing_parquet", "value": 0, "threshold": 0, "config_id": cfg.config_id
            })
            continue

        df_ev3 = pd.read_parquet(ev3_file)
        df_vrmt = pd.read_parquet(vrmt_file)

        ev3_diag = compute_stream_diagnostics(df_ev3, "local_time_s")
        vrmt_diag = compute_stream_diagnostics(df_vrmt, "relative_time_s")

        stream_diagnostics.append({
            "trial_id": tid,
            "ev3_valid_rows": ev3_diag.get("valid_rows"),
            "ev3_duration": ev3_diag.get("duration"),
            "ev3_eff_rate": ev3_diag.get("eff_rate"),
            "ev3_irregularity": ev3_diag.get("irregularity_index"),
            "vrmt_valid_rows": vrmt_diag.get("valid_rows"),
            "vrmt_duration": vrmt_diag.get("duration"),
            "vrmt_eff_rate": vrmt_diag.get("eff_rate"),
            "vrmt_irregularity": vrmt_diag.get("irregularity_index"),
        })

        rejections = apply_quality_gates(cfg, tid, ev3_diag, vrmt_diag)

        is_extended = str(p.get("is_extended_log", "False")).lower() == "true"
        tier = p.get("dataset_tier", "unknown")

        if is_extended:
            # Additional compatibility gates for extended logs
            compat_issues = []
            if p["status"] != "paired":
                compat_issues.append("not_paired")
            if not ev3_diag.get("valid_rows") or ev3_diag.get("valid_rows") < 100:
                compat_issues.append("insufficient_ev3_samples")
            if not vrmt_diag.get("valid_rows") or vrmt_diag.get("valid_rows") < 100:
                compat_issues.append("insufficient_vrmt_samples")
            if ev3_diag.get("duration", 0) < 5.0 or vrmt_diag.get("duration", 0) < 5.0:
                compat_issues.append("insufficient_duration")

            if compat_issues:
                extended_compatibility_report.append({
                    "trial_id": tid,
                    "dataset_tier": tier,
                    "compatibility_status": "incompatible",
                    "issues": ";".join(compat_issues)
                })
                # We can also add these to rejections so they don't proceed
                rejections.extend([{
                    "trial_id": tid, "stage": "compatibility_gate",
                    "reason": issue, "value": 0, "threshold": 0, "config_id": cfg.config_id
                } for issue in compat_issues])
            else:
                extended_compatibility_report.append({
                    "trial_id": tid,
                    "dataset_tier": tier,
                    "compatibility_status": "compatible",
                    "issues": "none"
                })

        if rejections:
            rejection_log.extend(rejections)
        else:
            accepted_trials.append(tid)

    # Save outputs
    base_out = os.path.join(cfg.paths.get("output_base", "outputs"), "04_quality")
    os.makedirs(base_out, exist_ok=True)

    stream_diag_path = os.path.join(base_out, "stream_diagnostics.csv")
    if stream_diagnostics:
        pd.DataFrame(stream_diagnostics).to_csv(stream_diag_path, index=False)

    rej_log_path = os.path.join(base_out, "rejection_log.csv")
    if rejection_log:
        pd.DataFrame(rejection_log).to_csv(rej_log_path, index=False)

    accepted_path = os.path.join(base_out, "accepted_trials.json")
    with open(accepted_path, 'w') as f:
        json.dump(accepted_trials, f, indent=2)

    compat_report_path = os.path.join(base_out, "extended_log_compatibility_report.csv")
    if extended_compatibility_report:
        pd.DataFrame(extended_compatibility_report).to_csv(compat_report_path, index=False)
    else:
        pd.DataFrame([{"trial_id": "none", "dataset_tier": "none", "compatibility_status": "none", "issues": "none"}]).to_csv(compat_report_path, index=False)

    prov.record_stage(
        stage_name="03_detect_quality_issues",
        config_id=cfg.config_id,
        input_files=[pair_path],
        output_files=[stream_diag_path, rej_log_path, accepted_path],
        additional_meta={"accepted": len(accepted_trials), "rejected": len(rejection_log)}
    )

    print(f"Quality gating complete. {len(accepted_trials)} trials accepted, {len(set([r['trial_id'] for r in rejection_log]))} trials rejected.")

if __name__ == "__main__":
    main()
