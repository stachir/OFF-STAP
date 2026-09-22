"""Persist the canonical run-scoped population gate before Stage 14."""

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from offstap.core.config import load_config
from offstap.core.population_gate import persist_trial_gate_ledger


def main() -> None:
    cfg = load_config("config/alignment_config.yaml")
    run = Path(cfg.paths.get("output_base", "outputs"))
    ledger_path, table_path = persist_trial_gate_ledger(run)
    print(f"Authoritative population gate ledger: {ledger_path}")
    print(f"Authoritative population gate table: {table_path}")


if __name__ == "__main__":
    main()
