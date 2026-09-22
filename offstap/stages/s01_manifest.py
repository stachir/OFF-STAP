"""
scripts/01_build_manifest.py

Executable script to scan directories and build manifest files.
"""

import sys
import os

# Add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offstap.core.config import load_config
from offstap.core.file_index import build_manifest, build_pair_manifest, save_csv
from offstap.core.provenance import ProvenanceTracker

def main():
    cfg = load_config()
    prov = ProvenanceTracker(cfg.paths.get("output_provenance", "outputs/provenance"))

    print("Stage 01: Building file manifests...")
    manifest = build_manifest(cfg)
    pairs = build_pair_manifest(manifest)

    if cfg.dev_mode:
        print("DEV_MODE active: Limiting to first 10 trial pairs for fast execution.")
        pairs = pairs[:10]

    manifest_path = os.path.join(cfg.paths.get("output_base", "outputs"), "01_manifest", "file_manifest.csv")
    pair_path = os.path.join(cfg.paths.get("output_base", "outputs"), "01_manifest", "pair_manifest.csv")
    duplicates_path = os.path.join(cfg.paths.get("output_base", "outputs"), "01_manifest", "duplicate_files.csv")
    missing_path = os.path.join(cfg.paths.get("output_base", "outputs"), "01_manifest", "missing_pairs.csv")
    parsing_errors_path = os.path.join(cfg.paths.get("output_base", "outputs"), "01_manifest", "filename_parsing_errors.csv")

    # Filter subsets
    duplicates = [p for p in pairs if p["status"] in ("duplicate_ev3", "duplicate_vrmt")]
    missing = [p for p in pairs if p["status"] in ("missing_ev3", "missing_vrmt")]
    parsing_errors = [m for m in manifest if m.get("parser_status") == "malformed"]

    import pandas as pd
    save_csv(manifest, manifest_path)
    save_csv(pairs, pair_path)
    pd.DataFrame(duplicates if duplicates else [{"file_path": "none", "status": "none"}]).to_csv(duplicates_path, index=False)
    pd.DataFrame(missing if missing else [{"trial_id": "none", "status": "none"}]).to_csv(missing_path, index=False)
    pd.DataFrame(parsing_errors if parsing_errors else [{"file_path": "none", "parser_status": "none"}]).to_csv(parsing_errors_path, index=False)

    # Write input_checksums.csv
    checksums_path = os.path.join(cfg.paths.get("output_base", "outputs"), "input_checksums.csv") # we're in 01_manifest right now but input_checksums is run-level
    run_dir = os.path.dirname(os.path.dirname(manifest_path)) # up one level from 01_manifest
    checksums_path = os.path.join(run_dir, "input_checksums.csv")
    if manifest:
        pd.DataFrame(manifest)[["file_path", "file_size", "file_checksum"]].rename(columns={"file_checksum": "checksum"}).to_csv(checksums_path, index=False)


    prov.record_stage(
        stage_name="01_build_manifest",
        config_id=cfg.config_id,
        input_files=[], # Input is dirs
        output_files=[manifest_path, pair_path],
        additional_meta={"total_files": len(manifest), "total_pairs": len(pairs), "duplicates": len(duplicates), "missing": len(missing)}
    )

    print(f"Manifests generated. {len(manifest)} files found, {len(pairs)} pairings evaluated.")

if __name__ == "__main__":
    main()
