"""
scripts/02_validate_streams.py

Executable script to validate streams, apply schemas, and save as Parquet.
"""

import sys
import os
import csv
import pandas as pd

# Add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offstap.core.config import load_config
from offstap.core.loaders import process_file_pair
from offstap.core.provenance import ProvenanceTracker

def main():
    cfg = load_config()
    prov = ProvenanceTracker(cfg.paths.get("output_provenance", "outputs/provenance"))

    pair_path = os.path.join(cfg.paths.get("output_base", "outputs"), "01_manifest", "pair_manifest.csv")
    if not os.path.exists(pair_path):
        raise FileNotFoundError(f"Error: {pair_path} not found. Run 01_build_manifest.py first.")

    pairs = []
    with open(pair_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(row)

    print(f"Stage 02: Validating and standardizing {len(pairs)} pairings...")

    processed_count = 0
    schema_rows = []
    unit_rows = []
    for p in pairs:
        if p["status"] == "paired":
            out_files = process_file_pair(cfg, p)

            # Record provenance for this pair
            in_files = [f for f in [p["ev3_file"], p["vrmt_file"]] if f]
            out_files = [f for f in out_files if f]

            prov.record_stage(
                stage_name="02_validate_streams",
                config_id=cfg.config_id,
                input_files=in_files,
                output_files=out_files,
                additional_meta={"trial_id": p["trial_id"]}
            )

            processed_count += 1

            for stream_type, parquet_path in zip(("ev3", "vrmt"), out_files):
                if not parquet_path or not os.path.exists(parquet_path):
                    continue
                df = pd.read_parquet(parquet_path)
                schema_cfg = cfg.schema.get(stream_type, {})
                if stream_type == "ev3":
                    required = [schema_cfg.get("time_column", "time")]
                else:
                    required = [
                        schema_cfg.get("wall_time_column", "timestamp"),
                        schema_cfg.get("x_column", "x"),
                        schema_cfg.get("y_column", "y"),
                        schema_cfg.get("z_column", "z"),
                    ]
                required = [c for c in required if c]
                missing = [c for c in required if c not in df.columns]
                schema_rows.append({
                    "trial_id": p["trial_id"],
                    "stream_type": stream_type,
                    "required_columns": ";".join(required),
                    "missing_required_columns": ";".join(missing),
                    "num_rows": len(df),
                    "schema_status": "valid" if not missing and len(df) > 0 else "invalid",
                    "reason_if_invalid": "" if not missing and len(df) > 0 else ("missing_required_column" if missing else "empty_cleaned_stream"),
                })

                time_factor_key = "ev3_time_to_s" if stream_type == "ev3" else "vrmt_time_to_s"
                unit_rows.append({
                    "trial_id": p["trial_id"],
                    "stream_type": stream_type,
                    "time_unit_factor_to_s": cfg.units.get(time_factor_key, 1.0),
                    "position_unit_factor_to_m": cfg.units.get("vrmt_pos_to_m", 1.0) if stream_type == "vrmt" else "",
                    "unit_status": "valid",
                    "reason_if_invalid": "",
                })

    schema_dir = cfg.paths.get("output_schema", "outputs/schema_reports")
    os.makedirs(schema_dir, exist_ok=True)
    pd.DataFrame(schema_rows, columns=[
        "trial_id", "stream_type", "required_columns", "missing_required_columns",
        "num_rows", "schema_status", "reason_if_invalid"
    ]).to_csv(os.path.join(schema_dir, "schema_validation_report.csv"), index=False)
    pd.DataFrame(unit_rows, columns=[
        "trial_id", "stream_type", "time_unit_factor_to_s", "position_unit_factor_to_m",
        "unit_status", "reason_if_invalid"
    ]).to_csv(os.path.join(schema_dir, "unit_validation_report.csv"), index=False)

    print(f"Standardization complete. Successfully processed {processed_count} valid pairs to Parquet format.")

if __name__ == "__main__":
    main()
