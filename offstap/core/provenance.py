"""
src/provenance.py

Handles provenance tracking, checksum generation, and metadata collection.
"""

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

def generate_file_checksum(filepath: str) -> str:
    """Generates a SHA-256 checksum for a given file."""
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def get_runtime_metadata() -> dict:
    """Collects OS and Python runtime metadata."""
    return {
        "python_version": sys.version.split()[0],
        "os": platform.system(),
        "os_release": platform.release(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

class ProvenanceTracker:
    def __init__(self, output_dir: str = "outputs/provenance"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.index_path = os.path.join(self.output_dir, "provenance_index.json")
        self.records = self._load_index()

    def _load_index(self) -> list:
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def add_record(self, record: dict):
        self.records.append(record)
        self._save_index()

    def _save_index(self):
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2)

    def record_stage(self, stage_name: str, config_id: str, input_files: list, output_files: list, additional_meta: dict = None):
        """Records the execution of a processing stage."""
        rec = {
            "stage_name": stage_name,
            "config_id": config_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runtime": get_runtime_metadata(),
            "inputs": [{"path": f, "checksum": generate_file_checksum(f)} for f in input_files],
            "outputs": [{"path": f, "checksum": generate_file_checksum(f)} for f in output_files],
            "metadata": additional_meta or {}
        }
        self.add_record(rec)

        # Log to pipeline_stage_log.csv if it exists
        log_csv = os.path.join(os.path.dirname(self.output_dir), "pipeline_stage_log.csv")
        if os.path.exists(log_csv) and os.environ.get("OFFSTAP_RUNNER_MANAGED_LOG") != "1":
            # We don't have start/end tracking exactly yet, so we'll just log the completion time
            # Format: stage,start_time,end_time,status,error
            end_time = datetime.now(timezone.utc).isoformat()
            with open(log_csv, 'a', encoding='utf-8') as f:
                f.write(f"{stage_name},,{end_time},completed,\n")
