"""
scripts/04_prepare_calibration_intervals.py

Executable script to derive signals, split calibration intervals,
and check for motion degeneracy.
"""
import sys
import os
import json
import pandas as pd

# Add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offstap.core.config import load_config
from offstap.core.ev3_signals import derive_ev3_signals
from offstap.core.vrmt_signals import derive_vrmt_signals
from offstap.core.partition import compute_partition, partition_records
from offstap.core.degeneracy import check_degeneracy
from offstap.core.provenance import ProvenanceTracker

def main():
    cfg = load_config()
    prov = ProvenanceTracker(cfg.paths.get("output_provenance", "outputs/provenance"))

    base_out = cfg.paths.get("output_base", "outputs")
    accepted_path = os.path.join(base_out, "04_quality", "accepted_trials.json")

    if not os.path.exists(accepted_path):
        raise FileNotFoundError(f"Error: {accepted_path} not found.")

    with open(accepted_path, 'r') as f:
        accepted_trials = json.load(f)

    print(f"Stage 04: Preparing calibration intervals for {len(accepted_trials)} trials...")

    out_dir_ev3 = os.path.join(cfg.paths.get("output_cleaned", "outputs/cleaned"), "ev3")
    out_dir_vrmt = os.path.join(cfg.paths.get("output_cleaned", "outputs/cleaned"), "vrmt")

    split_dir = os.path.join(base_out, "08_calibration_evaluation_split")
    calib_dir = os.path.join(split_dir, "calibration_data")
    eval_dir = os.path.join(split_dir, "evaluation_data")
    os.makedirs(calib_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    degen_log = []
    split_intervals_data = []
    partition_rows = []

    for tid in accepted_trials:
        ev3_file = os.path.join(out_dir_ev3, f"{tid}_ev3_cleaned.parquet")
        vrmt_file = os.path.join(out_dir_vrmt, f"{tid}_vrmt_cleaned.parquet")

        df_ev3 = pd.read_parquet(ev3_file)
        df_vrmt = pd.read_parquet(vrmt_file)

        # 1. Derive signals
        df_ev3 = derive_ev3_signals(
            df_ev3,
            speed_col=cfg.schema.get("ev3", {}).get("speed_column", "motor_speed"),
            heading_col=cfg.schema.get("ev3", {}).get("heading_column", "gyro_heading")
        )
        df_vrmt = derive_vrmt_signals(df_vrmt)

        # 2. Compute one EV3-derived partition and share its boundary with Vive.
        partition = compute_partition(df_ev3, df_vrmt, cfg)
        ev3_calib, ev3_eval = partition.ev3_calibration, partition.ev3_evaluation
        vrmt_calib, vrmt_eval = partition.vive_calibration, partition.vive_evaluation
        ev3_eff_dur = partition.t_cal_s
        partition_rows.extend(partition_records(partition))

        # 3. Check degeneracy on calibration interval
        ev3_degen, ev3_metrics = check_degeneracy(ev3_calib, cfg, is_ev3=True)
        vrmt_degen, vrmt_metrics = check_degeneracy(vrmt_calib, cfg, is_ev3=False)

        # Determine overall degeneracy
        is_degen = ev3_degen or vrmt_degen
        reason = "short_distance_or_angle" if is_degen else "none"

        degen_log.append({
            "trial_id": tid,
            "partition_status": partition.status,
            "calibration_duration_s": ev3_eff_dur,
            "linear_excitation_score": ev3_metrics["distance"],
            "angular_excitation_score": ev3_metrics["angular_change"],
            "path_length_calibration": ev3_metrics["distance"],
            "rotation_range_deg": ev3_metrics["angular_change"],
            "degeneracy_status": "rejected" if partition.status != "success" else ("degenerate" if is_degen else "ok"),
            "reason_if_degenerate": partition.status if partition.status != "success" else reason,
            "ev3_degenerate": ev3_degen,
            "vrmt_degenerate": vrmt_degen
        })

        split_intervals_data.append({
            "trial_id": tid,
            "calibration_start_s": 0.0,
            "calibration_end_s": ev3_eff_dur,
            "evaluation_start_s": ev3_eff_dur,
            "evaluation_end_s": partition.t_total_s,
            "split_method": "ev3_derived_nominal_then_minimum_eval_gate",
            "split_status": partition.status,
            "partition_hash": partition.partition_hash,
            "n_calibration_rows_ev3": len(ev3_calib),
            "n_evaluation_rows_ev3": len(ev3_eval),
            "n_calibration_rows_vrmt": len(vrmt_calib),
            "n_evaluation_rows_vrmt": len(vrmt_eval),
            "vive_rows_excluded_outside_ev3_support": int(
                partition.metadata.get("vive_rows_excluded_outside_ev3_support", 0)
            ),
            "eligible_evaluation_fitting_keys_ev3": json.dumps(
                partition.metadata.get("eligible_evaluation_fitting_keys", {}).get("ev3", []),
                separators=(",", ":"),
            ),
            "eligible_evaluation_fitting_keys_vive": json.dumps(
                partition.metadata.get("eligible_evaluation_fitting_keys", {}).get("vive", []),
                separators=(",", ":"),
            ),
        })

        # Save partitioned files
        e_c_path = os.path.join(calib_dir, f"{tid}_ev3_calib.parquet")
        e_e_path = os.path.join(eval_dir, f"{tid}_ev3_eval.parquet")
        v_c_path = os.path.join(calib_dir, f"{tid}_vrmt_calib.parquet")
        v_e_path = os.path.join(eval_dir, f"{tid}_vrmt_eval.parquet")

        ev3_calib.to_parquet(e_c_path, index=False)
        ev3_eval.to_parquet(e_e_path, index=False)
        vrmt_calib.to_parquet(v_c_path, index=False)
        vrmt_eval.to_parquet(v_e_path, index=False)

        prov.record_stage(
            stage_name="04_prepare_calibration",
            config_id=cfg.config_id,
            input_files=[ev3_file, vrmt_file],
            output_files=[e_c_path, e_e_path, v_c_path, v_e_path],
            additional_meta={
                "ev3_degenerate": ev3_degen,
                "vrmt_degenerate": vrmt_degen,
                "partition_status": partition.status,
                "vive_rows_excluded_outside_ev3_support": int(
                    partition.metadata.get("vive_rows_excluded_outside_ev3_support", 0)
                ),
            }
        )

    degen_path = os.path.join(split_dir, "calibration_motion_degeneracy.csv")
    pd.DataFrame(degen_log).to_csv(degen_path, index=False)

    split_intervals_path = os.path.join(split_dir, "split_intervals.csv")
    pd.DataFrame(split_intervals_data).to_csv(split_intervals_path, index=False)

    # Canonical persisted membership consumed by downstream EV3/Vive stages.
    partition_path = os.path.join(split_dir, "partition.csv")
    pd.DataFrame(partition_rows).to_csv(partition_path, index=False)

    # Generate calibration_quality_report.csv
    quality_path = os.path.join(split_dir, "calibration_quality_report.csv")
    # Using degen_log to populate a basic quality report
    qual_log = [{
        "trial_id": d["trial_id"],
        "quality_status": "ok" if d["degeneracy_status"] == "ok" else "rejected",
        "partition_status": d.get("partition_status", "unknown"),
    } for d in degen_log]
    pd.DataFrame(qual_log).to_csv(quality_path, index=False)

    prov.record_stage(
        stage_name="04_partition_ledger",
        config_id=cfg.config_id,
        input_files=[accepted_path],
        output_files=[degen_path, split_intervals_path, partition_path, quality_path],
        additional_meta={
            "partition_hashes": sorted({row.get("partition_hash", "") for row in partition_rows}),
            "partition_row_count": len(partition_rows),
            "rejected_partition_count": sum(
                1 for row in split_intervals_data if row.get("split_status") != "success"
            ),
        },
    )

    print("Calibration intervals prepared and degeneracy checked.")

if __name__ == "__main__":
    main()
