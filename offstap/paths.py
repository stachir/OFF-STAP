"""Safe preparation of a public OFF-STAP run directory."""

from dataclasses import dataclass
from pathlib import Path

import yaml


PACKAGED_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "default.yaml"


STAGE_DIRECTORIES = (
    "01_manifest",
    "02_schema",
    "03_cleaned/ev3",
    "03_cleaned/vrmt",
    "04_quality",
    "06_vive_reference/projected_vive",
    "07_spatial/spatial_transform_matrices",
    "08_calibration_evaluation_split",
    "09_temporal/temporal_offset_plots",
    "10_clock_model/clock_model_plots",
    "11_continuous_time_reference/ct_query_results",
    "11_continuous_time_reference/ct_fit_diagnostics",
    "13_integrated_logs/ct_reference",
    "14_metrics",
    "15_hypotheses/H1/h1_plots",
    "15_hypotheses/H2/h2_plots",
    "15_hypotheses/H3/h3_plots",
    "16_config_comparison",
    "17_statistics",
    "18_plots/audit",
    "18_plots/temporal",
    "18_plots/spatial",
    "18_plots/representation",
    "18_plots/results",
    "19_tables",
    "20_tests",
    "provenance",
)


@dataclass(frozen=True)
class RunOptions:
    ev3_dir: Path
    vive_dir: Path
    config: Path
    output_dir: Path
    results: frozenset[str]


def prepare_run(options: RunOptions) -> Path:
    """Validate user-owned paths then create an empty run's stage layout."""
    ev3_dir = Path(options.ev3_dir).resolve()
    vive_dir = Path(options.vive_dir).resolve()
    config = Path(options.config).resolve()
    output_dir = Path(options.output_dir).resolve()

    _validate_input_directory(ev3_dir, "EV3")
    _validate_input_directory(vive_dir, "Vive")
    if ev3_dir == vive_dir:
        raise ValueError("EV3 and Vive input directories must be distinct")
    validate_config(config)
    for input_dir in (ev3_dir, vive_dir):
        if _is_within(output_dir, input_dir) or _is_within(input_dir, output_dir):
            raise ValueError("output directory must not overlap an input directory")
    _validate_output_directory(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in STAGE_DIRECTORIES:
        (output_dir / relative_path).mkdir(parents=True, exist_ok=True)
    return output_dir


def _validate_input_directory(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} input directory not found: {path}")
    if not path.is_dir():
        raise ValueError(f"{label} input path is not a directory: {path}")


def validate_config(path: Path | str) -> dict:
    """Accept only path/identity variants of the packaged scientific configuration."""
    path = Path(path)
    payload = _load_yaml(path)
    default = _load_yaml(PACKAGED_DEFAULT_CONFIG)
    if not isinstance(payload.get("paths"), dict):
        raise ValueError(f"configuration paths must contain a mapping: {path}")
    if not isinstance(payload.get("config_id"), str) or not payload["config_id"]:
        raise ValueError(f"configuration config_id must be a nonempty string: {path}")
    scientific_payload = {key: value for key, value in payload.items() if key not in {"paths", "config_id"}}
    scientific_default = {key: value for key, value in default.items() if key not in {"paths", "config_id"}}
    if scientific_payload != scientific_default:
        raise ValueError("configuration changes frozen scientific settings")
    return payload


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"configuration file not found: {path}")
    if not path.is_file():
        raise ValueError(f"configuration path is not a file: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except OSError as exc:
        raise OSError(f"configuration file is not readable: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"configuration YAML is malformed: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"configuration YAML must contain a mapping: {path}")
    return payload


def _validate_output_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents
