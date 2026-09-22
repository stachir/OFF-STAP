"""
src/config.py

Loads and parses the alignment configuration YAML.
"""

import os
import yaml

from offstap.paths import validate_config

class Config:
    def __init__(self, config_dict: dict):
        self._config = config_dict

        self.version = self._config.get("version", "1.0")
        self.config_id = self._config.get("config_id", "default")
        self.random_seed = self._config.get("random_seed", 42)

        self.paths = self._config.get("paths", {})
        self.patterns = self._config.get("patterns", {})
        self.schema = self._config.get("schema", {})
        self.units = self._config.get("units", {})
        self.geometry = self._config.get("geometry", {})
        self.quality = self._config.get("quality", {})
        self.calibration = self._config.get("calibration", {})
        self.temporal = self._config.get("temporal", {})
        self.clock_model = self._config.get("clock_model", {})
        self.representation = self._config.get("representation", {})
        self.metrics = self._config.get("metrics", {})
        self.plotting = self._config.get("plotting", {})

def load_config(path="config/alignment_config.yaml") -> Config:
    selected = os.environ.get("OFFSTAP_CONFIG", path)
    validate_config(selected)

    with open(selected, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    cfg = Config(data)

    for key, variable in (("input_ev3", "OFFSTAP_EV3_DIR"), ("input_vrmt", "OFFSTAP_VIVE_DIR")):
        if os.environ.get(variable):
            cfg.paths[key] = os.environ[variable]

    run_dir = os.environ.get("PIPELINE_RUN_DIR")
    if run_dir:
        cfg.paths["output_base"] = run_dir
        cfg.paths["output_cleaned"] = os.path.join(run_dir, "03_cleaned")
        cfg.paths["output_aligned"] = os.path.join(run_dir, "13_integrated_logs")
        cfg.paths["output_schema"] = os.path.join(run_dir, "02_schema")
        cfg.paths["output_plots"] = os.path.join(run_dir, "18_plots")
        cfg.paths["output_provenance"] = os.path.join(run_dir, "provenance")

    cfg.dev_mode = False

    return cfg
