"""
src/file_index.py

Logic for scanning input directories, building the file manifest,
and pairing EV3/Vive logs deterministically.
"""

import os
import re
import csv
import ast
from collections.abc import Iterable
from typing import List, Dict, Any
from offstap.core.config import Config
from offstap.core.provenance import generate_file_checksum

def parse_filename(filename: str, pattern: str) -> dict:
    match = re.match(pattern, filename)
    if match:
        return match.groupdict()
    return {}

def build_manifest(cfg: Config) -> List[Dict[str, Any]]:
    ev3_dir = cfg.paths.get("input_ev3", "data/ev3")
    vrmt_dir = cfg.paths.get("input_vrmt", "data/vrmt")
    regex = cfg.patterns.get("filename_regex", "")

    manifest = []

    for stream_type, d in [("ev3", ev3_dir), ("vrmt", vrmt_dir)]:
        if not os.path.exists(d):
            continue

        for f in os.listdir(d):
            if not f.endswith(".csv"):
                continue

            path = os.path.join(d, f)
            parsed = parse_filename(f, regex)

            if parsed:
                trial_id = f"{parsed['drive_id']}_{parsed['route_id']}_{parsed['date']}_{parsed['time']}"
                parser_status = "ok"
            else:
                trial_id = "unknown"
                parser_status = "malformed"

            size = os.path.getsize(path)
            # Peek first line for columns
            with open(path, 'r', encoding='utf-8') as file_handle:
                first_line = file_handle.readline().strip()
                cols = first_line.split(',') if first_line else []

            manifest.append({
                "file_path": path,
                "file_name": f,
                "trial_id": trial_id,
                "stream_type": parsed.get("stream_type", stream_type) if parsed else stream_type,
                "parsed_drive": parsed.get("drive_id", ""),
                "parsed_route": parsed.get("route_id", ""),
                "parsed_date": parsed.get("date", ""),
                "parsed_time": parsed.get("time", ""),
                "file_size": size,
                "detected_columns": len(cols),
                "file_checksum": generate_file_checksum(path),
                "parser_status": parser_status
            })

    return manifest

def build_pair_manifest(manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trials = {}

    for row in manifest:
        tid = row["trial_id"]
        if tid == "unknown":
            continue

        if tid not in trials:
            trials[tid] = {"ev3": [], "vrmt": []}

        stype = row["stream_type"]
        if stype in trials[tid]:
            trials[tid][stype].append(row)

    pairs = []

    # We need to assign repetitions correctly per scenario.
    # Group by scenario to keep track of valid pairs.
    scenario_counts = {}

    # Sort trial IDs so repetition assignment is deterministic
    sorted_tids = sorted(trials.keys())

    for tid in sorted_tids:
        streams = trials[tid]
        ev3_files = streams["ev3"]
        vrmt_files = streams["vrmt"]

        # Keep every source entry in the manifest.  The legacy singular file
        # fields below remain for callers that only handle unique pairs, while
        # these collections preserve duplicate-file evidence for validation.
        ev3_paths = [entry.get("file_path", "") for entry in ev3_files]
        vrmt_paths = [entry.get("file_path", "") for entry in vrmt_files]

        # Loaders attach duplicate-key diagnostics to DataFrame.attrs.  When
        # those diagnostics (or source keys) are propagated into a manifest,
        # carry them through the pair record instead of dropping rows/files.
        duplicate_key_count = 0
        duplicate_key_rows = 0
        for stream_name, entries in streams.items():
            propagated = []
            attrs_only = []
            for entry in entries:
                values = entry.get(
                    "source_timestamp_ns",
                    entry.get("source_keys", entry.get("source_key", [])),
                )
                if values is None:
                    continue
                if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
                    values = [values]
                values = list(values)
                if values:
                    propagated.extend(values)
                else:
                    attrs_only.append(entry)
            if propagated:
                seen = set()
                repeated = set()
                for value in propagated:
                    try:
                        marker = value.item() if hasattr(value, "item") else value
                        if marker in seen:
                            repeated.add(marker)
                        else:
                            seen.add(marker)
                    except (TypeError, ValueError):
                        continue
                duplicate_key_count += len(repeated)
                for value in propagated:
                    try:
                        marker = value.item() if hasattr(value, "item") else value
                        if marker is not None and marker in repeated:
                            duplicate_key_rows += 1
                    except (TypeError, ValueError):
                        continue
            for entry in attrs_only:
                duplicate_key_count += int(entry.get("duplicate_source_key_count", 0) or 0)
                duplicate_key_rows += int(entry.get("duplicate_source_key_rows", 0) or 0)

        reasons = []
        if len(ev3_files) > 1:
            reasons.append("duplicate_ev3")
        if len(vrmt_files) > 1:
            reasons.append("duplicate_vrmt")
        if len(ev3_files) == 0 and len(vrmt_files) > 0:
            reasons.append("missing_ev3")
        if len(vrmt_files) == 0 and len(ev3_files) > 0:
            reasons.append("missing_vrmt")
        status = reasons[0] if reasons else "paired"

        # Parse scenario
        parts = tid.split('_')
        d = parts[0] if len(parts) > 0 else "unknown"
        r = parts[1] if len(parts) > 1 else "unknown"
        scenario_id = f"{d}_{r}"

        is_planned_candidate = (status == "paired" and d.startswith("D") and r.startswith("T")
                                and d[1:].isdigit() and r[1:].isdigit()
                                and 1 <= int(d[1:]) <= 9 and 1 <= int(r[1:]) <= 9)

        if is_planned_candidate:
            if scenario_id not in scenario_counts:
                scenario_counts[scenario_id] = 0
            scenario_counts[scenario_id] += 1
            rep = scenario_counts[scenario_id]

            if rep <= 5:
                dataset_tier = "planned_405"
                is_planned_405 = True
                is_extended_log = False
            else:
                dataset_tier = "extended_previous_paper"
                is_planned_405 = False
                is_extended_log = True
        else:
            rep = 1
            if status != "paired":
                dataset_tier = "diagnostic_or_unplanned"
                is_planned_405 = False
                is_extended_log = False
            else:
                # If not part of 9x9 but paired, it's extended
                dataset_tier = "extended_previous_paper"
                is_planned_405 = False
                is_extended_log = True

        # Additional handling for unparseable metadata
        if d == "unknown" or r == "unknown":
            dataset_tier = "unknown"
            is_planned_405 = False
            is_extended_log = False

        pair_status = status
        if dataset_tier == "planned_405":
            analysis_set = "primary_planned_405"
        elif dataset_tier == "extended_previous_paper":
            analysis_set = "extended_sensitivity"
        else:
            analysis_set = "diagnostic_only"

        if pair_status == "paired" and dataset_tier != "unknown":
            compatibility_status = "compatible"
        elif pair_status == "paired":
            compatibility_status = "unknown_dataset_tier"
        elif pair_status in ("duplicate_ev3", "duplicate_vrmt"):
            compatibility_status = "duplicate_trial"
        else:
            compatibility_status = pair_status

        pairs.append({
            "trial_id": tid,
            "status": status,
            "status_reasons": reasons,
            "pair_status": pair_status,
            "ev3_file": ev3_files[0]["file_path"] if ev3_files else "",
            "vrmt_file": vrmt_files[0]["file_path"] if vrmt_files else "",
            "ev3_files": ev3_paths,
            "vrmt_files": vrmt_paths,
            "ev3_checksum": ev3_files[0]["file_checksum"] if ev3_files else "",
            "vrmt_checksum": vrmt_files[0]["file_checksum"] if vrmt_files else "",
            "drive_id": d,
            "route_id": r,
            "dataset_tier": dataset_tier,
            "analysis_set": analysis_set,
            "protocol_source": "planned" if is_planned_405 else ("extended" if is_extended_log else "unknown"),
            "is_planned_405": is_planned_405,
            "is_extended_log": is_extended_log,
            "scenario_id": scenario_id,
            "repetition_id": rep,
            "trial_group_id": f"{scenario_id}_grp",
            "duplicate_status": "duplicate" if any(reason.startswith("duplicate_") for reason in reasons) else "unique",
            "duplicate_source_key_count": duplicate_key_count,
            "duplicate_source_key_rows": duplicate_key_rows,
            "compatibility_status": compatibility_status
        })

    # Also add malformed
    for row in manifest:
        if row["parser_status"] == "malformed":
            pairs.append({
                "trial_id": row["file_name"],
                "status": "malformed_filename",
                "status_reasons": ["malformed_filename"],
                "pair_status": "malformed_filename",
                "ev3_file": row["file_path"] if row["stream_type"] == "ev3" else "",
                "vrmt_file": row["file_path"] if row["stream_type"] == "vrmt" else "",
                "ev3_files": [row["file_path"]] if row["stream_type"] == "ev3" else [],
                "vrmt_files": [row["file_path"]] if row["stream_type"] == "vrmt" else [],
                "ev3_checksum": "",
                "vrmt_checksum": "",
                "drive_id": "unknown",
                "route_id": "unknown",
                "dataset_tier": "unknown",
                "analysis_set": "diagnostic_only",
                "protocol_source": "unknown",
                "is_planned_405": False,
                "is_extended_log": False,
                "scenario_id": "unknown",
                "repetition_id": 0,
                "trial_group_id": "unknown",
                "duplicate_status": "unknown",
                "duplicate_source_key_count": int(row.get("duplicate_source_key_count", 0) or 0),
                "duplicate_source_key_rows": int(row.get("duplicate_source_key_rows", 0) or 0),
                "compatibility_status": "malformed_filename"
            })

    return pairs

def save_csv(data: List[Dict[str, Any]], output_path: str):
    if not data:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    keys = data[0].keys()
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
