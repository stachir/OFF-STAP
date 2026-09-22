"""
scripts/05_estimate_alignment_parameters.py

Executable script to estimate spatial, temporal, and clock model parameters
on the calibration interval.
"""
import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offstap.core.config import load_config
from offstap.core.spatial import estimate_spatial_transform
from offstap.core.temporal import estimate_temporal_alignment
from offstap.core.clock_model import fit_clock_model
from offstap.core.provenance import ProvenanceTracker
from offstap.core.persistence import temporal_params_to_json, _json_safe, persist_fitted_parameters

def main():
    cfg = load_config()
    prov = ProvenanceTracker(cfg.paths.get("output_provenance", "outputs/provenance"))

    base_out = cfg.paths.get("output_base", "outputs")
    split_dir = os.path.join(base_out, "08_calibration_evaluation_split")
    degen_path = os.path.join(split_dir, "calibration_motion_degeneracy.csv")

    if not os.path.exists(degen_path):
        raise FileNotFoundError(f"Error: {degen_path} not found.")

    df_degen = pd.read_csv(degen_path)
    if "partition_status" not in df_degen.columns:
        raise RuntimeError(
            "FAILED: calibration partition status is missing; refusing to fit parameters"
        )
    # Filter out degenerate trials
    partition_ok = df_degen["partition_status"].eq("success")
    valid_trials = df_degen[
        (~df_degen["ev3_degenerate"])
        & (~df_degen["vrmt_degenerate"])
        & partition_ok
    ]["trial_id"].tolist()

    print(f"Stage 05: Estimating parameters for {len(valid_trials)} non-degenerate calibration intervals...")

    calib_dir = os.path.join(split_dir, "calibration_data")

    spatial_log = []
    spatial_diag_log = []
    lever_arm_log = []
    temporal_onset_log = []
    temporal_cc_log = []
    clock_log = []

    params_out = os.path.join(base_out, "alignment_parameters.json")
    all_params = {}

    # Plotting tracking
    max_plots = cfg.plotting.get("max_per_scenario", 1)
    plots_generated = {}

    for tid in valid_trials:
        e_c_path = os.path.join(calib_dir, f"{tid}_ev3_calib.parquet")
        v_c_path = os.path.join(calib_dir, f"{tid}_vrmt_calib.parquet")

        df_ev3 = pd.read_parquet(e_c_path)
        df_vrmt = pd.read_parquet(v_c_path)

        # 1. Temporal Alignment (independent of the spatial frame)
        la_src = "manually_configured" if cfg.geometry.get("apply_lever_arm_correction", True) else "assumed_zero"
        lx, ly, lz = cfg.geometry.get("lever_arm_offset_xyz", [0.0, 0.0, 0.0])
        lever_arm_log.append({
            "trial_id": tid,
            "lever_arm_x": lx,
            "lever_arm_y": ly,
            "lever_arm_z": lz,
            "lever_arm_source": la_src
        })

        temp_params = estimate_temporal_alignment(df_ev3, df_vrmt, cfg)
        temporal_onset_log.append({
            "trial_id": tid,
            "b_onset": temp_params["b_onset"]
        })
        b_onset = temp_params.get("b_onset", np.nan)
        b_refined = temp_params.get("b_refined", np.nan)
        b_final = temp_params.get("b_final", np.nan)
        lag = temp_params.get("lag", np.nan)
        best_corr = temp_params.get("best_corr", np.nan)
        second_best_corr = temp_params.get("second_best_corr", np.nan)
        peak_confidence_ratio = temp_params.get("peak_confidence_ratio", np.nan)
        overlap_duration_s = temp_params.get("overlap_duration_s", np.nan)
        required_temporal_values = [
            b_onset,
            b_refined,
            b_final,
            lag,
            best_corr,
            second_best_corr,
            peak_confidence_ratio,
            overlap_duration_s,
        ]
        temporal_valid_for_paper = (
            temp_params["status"] == "ok"
            and all(np.isfinite(value) for value in required_temporal_values)
        )
        temporal_invalid_reason = ""
        if not temporal_valid_for_paper:
            temporal_invalid_reason = (
                temp_params["status"]
                if temp_params["status"] != "ok"
                else "non_finite_temporal_diagnostic"
            )
        temporal_cc_log.append({
            "trial_id": tid,
            "b_onset": b_onset,
            "b_refined": b_refined,
            "b_final": b_final,
            "epsilon_t_s": b_onset - b_final,
            "lag": lag,
            "best_corr": best_corr,
            "second_best_corr": second_best_corr,
            "peak_confidence_ratio": peak_confidence_ratio,
            "overlap_duration_s": overlap_duration_s,
            "status": temp_params["status"],
            "valid_for_paper": bool(temporal_valid_for_paper),
            "reason_if_not_valid": temporal_invalid_reason
        })

        # 2. Clock Model
        clock_params = fit_clock_model(df_ev3, df_vrmt, temp_params, cfg)
        clock_log.append({
            "trial_id": tid,
            "selected_model": clock_params["selected_model"],
            "offset_b": clock_params["offset_b"],
            "scale_a": clock_params["scale_a"],
            "candidate_scale_a": clock_params.get("candidate_scale_a", np.nan),
            "candidate_offset_b": clock_params.get("candidate_offset_b", np.nan),
            "temporal_refinement_delta_s": clock_params["temporal_refinement_delta_s"],
            "affine_improvement_ratio": clock_params.get("affine_improvement_ratio", np.nan),
            "constant_mse": clock_params.get("constant_mse", np.nan),
            "affine_mse": clock_params.get("affine_mse", np.nan),
            "candidate_hit_bound": clock_params.get("candidate_hit_bound", False),
            "reason": clock_params["reason"]
        })

        # 3. Spatial registration uses the frozen temporal mapping so the
        # full calibration partition can be paired without temporal leakage.
        spatial_params = estimate_spatial_transform(
            df_ev3, df_vrmt, cfg, temp_params, clock_params
        )
        spatial_log.append({
            "trial_id": tid,
            "theta_rad": spatial_params["theta_rad"],
            "tx": spatial_params["tx"],
            "ty": spatial_params["ty"],
            "method_used": spatial_params.get("method_used", "unspecified")
        })
        trajectory_residual = spatial_params.get("trajectory_residual", np.nan)
        spatial_ok = np.isfinite(trajectory_residual)
        spatial_diag_log.append({
            "trial_id": tid,
            "projection_method": cfg.geometry.get("planar_projection_mode", "direct"),
            "rotation_deg": spatial_params["theta_rad"] * 180.0 / np.pi,
            "translation_x_m": spatial_params["tx"],
            "translation_y_m": spatial_params["ty"],
            "translation_norm_m": float(np.sqrt(spatial_params["tx"]**2 + spatial_params["ty"]**2)),
            "spatial_fit_rmse_mm": trajectory_residual * 1000.0,
            "start_position_residual_mm": spatial_params.get("start_residual", np.nan) * 1000.0,
            "heading_residual_deg": spatial_params.get("heading_residual_deg", np.nan),
            "planarity_residual_mm": spatial_params.get("planarity_residual", np.nan) * 1000.0,
            "trajectory_residual_mm": trajectory_residual * 1000.0,
            "num_fit_samples": spatial_params.get("num_fit_samples", 0),
            "spatial_status": "ok" if spatial_ok else "invalid",
            "valid_for_paper": bool(spatial_ok),
            "reason_if_not_valid": "" if spatial_ok else "rigid_fit_failed"
        })

        # Save all frozen parameters for this trial
        all_params[tid] = {
            "spatial": spatial_params,
            "temporal": {k: v for k, v in temp_params.items() if k not in ["valid_lags", "valid_corr"]},
            "clock": {k: v for k, v in clock_params.items() if k != "plot_data"}
        }

        # Plotting (if budget allows)
        scenario = "_".join(tid.split("_")[:2]) # e.g. D1_T1
        if max_plots == -1 or plots_generated.get(scenario, 0) < max_plots:
            plots_generated[scenario] = plots_generated.get(scenario, 0) + 1

            # 1. Temporal Plot
            if "valid_lags" in temp_params and "valid_corr" in temp_params:
                plt.figure(figsize=(8,4))
                plt.plot(temp_params["valid_lags"], temp_params["valid_corr"])
                plt.axvline(x=temp_params["lag"], color='r', linestyle='--', label=f'Best Lag: {temp_params["lag"]:.2f}s')
                plt.title(f"Cross-Correlation ({tid})")
                plt.xlabel("Lag (s)")
                plt.ylabel("Correlation")
                plt.legend()
                os.makedirs(os.path.join(base_out, "09_temporal", "temporal_offset_plots"), exist_ok=True)
                plt.savefig(os.path.join(base_out, "09_temporal", "temporal_offset_plots", f"{tid}_crosscorr.png"))
                plt.close()

            # 2. Clock Plot
            if "plot_data" in clock_params and clock_params["plot_data"]["t_v"]:
                pd_clock = clock_params["plot_data"]
                plt.figure(figsize=(8,4))
                plt.plot(pd_clock["t_v"], pd_clock["res_const"], label="Constant Offset Residuals", alpha=0.7)
                plt.plot(pd_clock["t_v"], pd_clock["res_affine"], label="Affine Model Residuals", alpha=0.7)
                plt.title(f"Clock Model Residuals ({tid})")
                plt.xlabel("VRMT Time (s)")
                plt.ylabel("Normalized Residual")
                plt.legend()
                os.makedirs(os.path.join(base_out, "10_clock_model", "clock_model_plots"), exist_ok=True)
                plt.savefig(os.path.join(base_out, "10_clock_model", "clock_model_plots", f"{tid}_clock_residuals.png"))
                plt.close()

    # Save diagnostics
    spatial_dir = os.path.join(base_out, "07_spatial")
    temporal_dir = os.path.join(base_out, "09_temporal")
    clock_dir = os.path.join(base_out, "10_clock_model")
    os.makedirs(os.path.join(spatial_dir, "spatial_transform_matrices"), exist_ok=True)

    pd.DataFrame(spatial_log).to_csv(os.path.join(spatial_dir, "spatial_calibration_parameters.csv"), index=False)
    pd.DataFrame(spatial_diag_log).to_csv(os.path.join(spatial_dir, "spatial_diagnostics.csv"), index=False)
    pd.DataFrame(lever_arm_log).to_csv(os.path.join(spatial_dir, "lever_arm_correction_report.csv"), index=False)

    # lever_arm_config_used.csv
    if lever_arm_log:
        pd.DataFrame([lever_arm_log[0]]).to_csv(os.path.join(spatial_dir, "lever_arm_config_used.csv"), index=False)

    pd.DataFrame(temporal_onset_log).to_csv(os.path.join(temporal_dir, "onset_detection_report.csv"), index=False)
    pd.DataFrame(temporal_cc_log).to_csv(os.path.join(temporal_dir, "crosscorr_report.csv"), index=False)

    # ambiguous_alignment_cases.csv
    df_cc = pd.DataFrame(temporal_cc_log)
    if not df_cc.empty:
        ambig = df_cc[df_cc["status"] != "ok"]
        ambig.to_csv(os.path.join(temporal_dir, "ambiguous_alignment_cases.csv"), index=False)
    else:
        pd.DataFrame(columns=[
            "trial_id", "b_onset", "b_refined", "b_final", "epsilon_t_s",
            "lag", "best_corr", "second_best_corr", "peak_confidence_ratio",
            "overlap_duration_s", "status",
            "valid_for_paper", "reason_if_not_valid"
        ]).to_csv(os.path.join(temporal_dir, "ambiguous_alignment_cases.csv"), index=False)

    pd.DataFrame(clock_log).to_csv(os.path.join(clock_dir, "clock_model_comparison.csv"), index=False)
    # Persist the exact Stage 05 clock_log values; this is serialization only.
    persist_fitted_parameters(clock_log, os.path.join(clock_dir, "fitted_parameters.csv"))

    # Save the all_params as json in spatial_dir for now, or just base_out
    # We will also save individual matrices
    for tid, p in all_params.items():
        with open(os.path.join(spatial_dir, "spatial_transform_matrices", f"{tid}_T_vive_to_robot.json"), "w") as f:
            json.dump(_json_safe(p["spatial"]), f, indent=2, allow_nan=False)

    with open(params_out, 'w') as f:
        persisted = {
            tid: {
                "spatial": _json_safe(record["spatial"]),
                "temporal": temporal_params_to_json(record["temporal"]),
                "clock": _json_safe(record["clock"]),
            }
            for tid, record in all_params.items()
        }
        json.dump(persisted, f, indent=2, allow_nan=False)

    prov.record_stage(
        stage_name="05_estimate_parameters",
        config_id=cfg.config_id,
        input_files=[degen_path],
        output_files=[params_out],
        additional_meta={"valid_trials_processed": len(valid_trials)}
    )

    print("Alignment parameters estimated and frozen.")

if __name__ == "__main__":
    main()
