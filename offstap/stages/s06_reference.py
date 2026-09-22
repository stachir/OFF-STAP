"""
scripts/06_build_reference_trajectories.py

Executable script to build continuous-time reference trajectories for Vive data.
"""
import sys
import os
import json
import pandas as pd
import pickle
import numpy as np

# Add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offstap.core.config import load_config
from offstap.core.trajectory_reference import build_reference_trajectory
from offstap.core.spatial import get_planar_trajectory
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

    out_dir_vrmt = os.path.join(cfg.paths.get("output_cleaned", "outputs/cleaned"), "vrmt")
    spline_dir = os.path.join(base_out, "11_continuous_time_reference", "models")
    os.makedirs(spline_dir, exist_ok=True)

    print(f"Stage 06: Building reference trajectories for {len(all_params)} trials...")

    traj_diag = []

    for tid, params in all_params.items():
        vrmt_file = os.path.join(out_dir_vrmt, f"{tid}_vrmt_cleaned.parquet")
        df_vrmt = pd.read_parquet(vrmt_file)

        spatial_p = params["spatial"]

        # Save projected VIVE trajectory
        rx, ry, rtheta, _ = get_planar_trajectory(cfg, df_vrmt)
        df_proj = pd.DataFrame({
            "t_vive_s": df_vrmt["relative_time_s"].values,
            "projected_x_m": rx,
            "projected_y_m": ry,
            "projected_theta_rad": rtheta if rtheta is not None else 0.0
        })
        os.makedirs(os.path.join(base_out, "06_vive_reference", "projected_vive"), exist_ok=True)
        df_proj.to_parquet(os.path.join(base_out, "06_vive_reference", "projected_vive", f"{tid}_vive_projected.parquet"))

        # Build continuous-time trajectory
        spline_traj = build_reference_trajectory(df_vrmt, spatial_p, cfg)

        # Save CT fit diagnostics
        os.makedirs(os.path.join(base_out, "11_continuous_time_reference", "ct_fit_diagnostics"), exist_ok=True)
        pd.DataFrame([{
            "trial_id": tid,
            "model_type": "cubic_spline",
            "spline_degree": 3,
            "num_control_points": len(spline_traj.t) if spline_traj.valid else 0,
            "fit_interval_start_s": spline_traj.t[0] if spline_traj.valid and len(spline_traj.t)>0 else 0,
            "fit_interval_end_s": spline_traj.t[-1] if spline_traj.valid and len(spline_traj.t)>0 else 0,
            "ct_model_status": "valid" if spline_traj.valid else "invalid"
        }]).to_csv(os.path.join(base_out, "11_continuous_time_reference", "ct_fit_diagnostics", f"{tid}_ct_fit_diagnostics.csv"), index=False)

        # Query CT spline at EV3 timestamps (requires EV3 file)
        out_dir_ev3 = os.path.join(cfg.paths.get("output_cleaned", "outputs/cleaned"), "ev3")
        ev3_file = os.path.join(out_dir_ev3, f"{tid}_ev3_cleaned.parquet")
        if os.path.exists(ev3_file):
            df_ev3 = pd.read_parquet(ev3_file)
            t_ev3 = df_ev3["local_time_s"].values
            if spline_traj.valid:
                qx, qy, qh = spline_traj.query(t_ev3)
            else:
                qx, qy, qh = np.zeros_like(t_ev3), np.zeros_like(t_ev3), np.zeros_like(t_ev3)

            os.makedirs(os.path.join(base_out, "11_continuous_time_reference", "ct_query_results"), exist_ok=True)
            pd.DataFrame({
                "trial_id": tid,
                "t_common_s": t_ev3,
                "vive_position_ct_x": qx,
                "vive_position_ct_y": qy,
                "vive_orientation_ct": qh,
                "query_status": "success" if spline_traj.valid else "failed"
            }).to_parquet(os.path.join(base_out, "11_continuous_time_reference", "ct_query_results", f"{tid}_ct_queried_at_ev3_times.parquet"))


        # Save spline parameters instead of CubicSpline object
        spline_path = os.path.join(spline_dir, f"{tid}_spline.pkl")
        with open(spline_path, 'wb') as f:
            data_to_save = {
                't': spline_traj.t,
                'x': spline_traj.x,
                'y': spline_traj.y,
                'valid': spline_traj.valid
            } if hasattr(spline_traj, 't') else {'valid': False}
            pickle.dump(data_to_save, f)

        traj_diag.append({
            "trial_id": tid,
            "representation_type": cfg.representation.get("continuous_type", "spline"),
            "valid": spline_traj.valid
        })

        prov.record_stage(
            stage_name="06_build_references",
            config_id=cfg.config_id,
            input_files=[vrmt_file, params_path],
            output_files=[spline_path],
            additional_meta={"valid": spline_traj.valid}
        )

    ct_dir = os.path.join(base_out, "11_continuous_time_reference")
    pd.DataFrame(traj_diag).to_csv(os.path.join(ct_dir, "ct_reference_model_report.csv"), index=False)
    print("Continuous trajectories built.")

if __name__ == "__main__":
    main()
