"""Command-line entry point for executing a public OFF-STAP run."""

import argparse
from pathlib import Path

from offstap.paths import PACKAGED_DEFAULT_CONFIG, RunOptions
from offstap.runner import run_core
from offstap.selection import parse_results


DEFAULT_CONFIG = PACKAGED_DEFAULT_CONFIG


def main(argv: list[str] | None = None) -> int:
    """Parse public run options and execute the requested core workflow."""
    parser = argparse.ArgumentParser(description="Execute an OFF-STAP run.")
    parser.add_argument("--ev3-dir", type=Path, required=True, help="directory containing EV3 telemetry")
    parser.add_argument("--vive-dir", type=Path, required=True, help="directory containing Vive tracking logs")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="alignment configuration YAML")
    parser.add_argument("--output-dir", type=Path, required=True, help="new or empty run output directory")
    parser.add_argument("--results", type=parse_results, default=frozenset({"core"}), help="comma-separated products or all")
    args = parser.parse_args(argv)
    run_core(RunOptions(args.ev3_dir, args.vive_dir, args.config, args.output_dir, args.results))
    return 0
