"""Produce the frozen Stage 14 cascade table from canonical artifacts."""

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from offstap.core.cascade import produce_cascade_table
from offstap.core.config import load_config


def main() -> None:
    cfg = load_config("config/alignment_config.yaml")
    base = Path(cfg.paths.get("output_base", "outputs"))
    # Gate membership is run-scoped.  Historical revision tables are never
    # silently imported into a new scientific run.
    gate = base / "19_tables" / "authoritative_population_gate_ledger.csv"
    produce_cascade_table(base, write=True, gate_ledger_path=gate if gate.exists() else None)


if __name__ == "__main__":
    main()
