"""
scripts/07_align_trials.py

Executable script to align EV3 and VRMT trials into a common timeframe.
"""
import sys
import os
import json
import pandas as pd
import pickle

# Add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offstap.core.config import load_config
from offstap.core.alignment import generate_aligned_trajectory
from offstap.core.trajectory_reference import ContinuousTrajectory
from offstap.core.provenance import ProvenanceTracker

def main():
    cfg = load_config()
    prov = ProvenanceTracker(cfg.paths.get("output_provenance", "outputs/provenance"))

    base_out = cfg.paths.get("output_base", "outputs")
    params_path = os.path.join(base_out, "alignment_parameters.json")

    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Error: {params_path} not found.")

    with open(params_path, 'r') as f:
        all_params = json.load(f)

    out_dir_ev3 = os.path.join(cfg.paths.get("output_cleaned", "outputs/cleaned"), "ev3")
    out_dir_vrmt = os.path.join(cfg.paths.get("output_cleaned", "outputs/cleaned"), "vrmt")
    spline_dir = os.path.join(base_out, "11_continuous_time_reference", "models")
    aligned_dir = os.path.join(cfg.paths.get("output_aligned", "outputs/aligned"), "ct_reference")
    partition_path = os.path.join(base_out, "08_calibration_evaluation_split", "partition.csv")
    if not os.path.exists(partition_path):
        raise FileNotFoundError(
            "Canonical calibration/evaluation partition ledger is required "
            f"before alignment: {partition_path}"
        )
    partition_ledger = pd.read_csv(partition_path)
    os.makedirs(aligned_dir, exist_ok=True)

    print(f"Stage 07: Aligning trials for {len(all_params)} valid calibration parameter sets...")

    for tid, params in all_params.items():
        ev3_file = os.path.join(out_dir_ev3, f"{tid}_ev3_cleaned.parquet")
        vrmt_file = os.path.join(out_dir_vrmt, f"{tid}_vrmt_cleaned.parquet")
        spline_file = os.path.join(spline_dir, f"{tid}_spline.pkl")

        if not os.path.exists(spline_file):
            continue

        df_ev3 = pd.read_parquet(ev3_file)
        df_vrmt = pd.read_parquet(vrmt_file)

        # Carry the canonical persisted partition into the aligned frame so
        # primary metrics can fail closed on calibration rows.  The ledger
        # stores source identities as JSON arrays, preserving source-row keys
        # across this stage boundary.
        if "source_row_index" in df_ev3.columns:
            rows = partition_ledger[
                (partition_ledger["trial_id"].astype(str) == str(tid))
                & (partition_ledger["stream"].astype(str).str.lower() == "ev3")
            ]
            membership = {}
            for _, row in rows.iterrows():
                try:
                    identities = json.loads(row.get("source_row_indices", "[]"))
                except (TypeError, json.JSONDecodeError):
                    identities = []
                for identity in identities:
                    part = str(row.get("partition", "")).strip().lower()
                    membership[identity] = part
                    # CSV/JSON round-trips may represent the same identity as
                    # an integer, float, or string; retain canonical aliases
                    # so no persisted membership is lost at this boundary.
                    membership[str(identity)] = part
                    try:
                        membership[int(identity)] = part
                    except (TypeError, ValueError):
                        pass
            source_ids = pd.to_numeric(df_ev3["source_row_index"], errors="coerce")
            df_ev3["partition"] = source_ids.map(membership)
            source_rows = df_ev3[[
                col for col in ("source_row_index", "source_timestamp_ns", "partition")
                if col in df_ev3.columns
            ]].copy()
        else:
            raise ValueError(f"{tid}: canonical source_row_index is required for partition propagation")

        with open(spline_file, 'rb') as f:
            spline_data = pickle.load(f)

        if spline_data.get('valid', False):
            spline_traj = ContinuousTrajectory(spline_data['t'], spline_data['x'], spline_data['y'])
        else:
            import numpy as np
            spline_traj = ContinuousTrajectory(np.array([]), np.array([]), np.array([]))

        temp_p = params["temporal"]
        clock_p = params["clock"]
        spatial_p = params["spatial"]

        df_aligned = generate_aligned_trajectory(
            df_ev3, df_vrmt, spline_traj, temp_p, clock_p, spatial_p, cfg,
            source_rows=source_rows,
        )

        if not df_aligned.empty:
            out_file = os.path.join(aligned_dir, f"{tid}_{cfg.config_id}.parquet")
            df_aligned.to_parquet(out_file, index=False)

            prov.record_stage(
                stage_name="07_align_trials",
                config_id=cfg.config_id,
                input_files=[ev3_file, vrmt_file, spline_file],
                output_files=[out_file],
                additional_meta={"aligned_rows": len(df_aligned)}
            )

    print("Trials aligned and saved to Parquet.")

if __name__ == "__main__":
    main()
