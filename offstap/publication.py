"""Publish selected current-run artifacts with portable provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import csv

from offstap.selection import ALL, expand_results


REPORT_LIMITATION = (
    "Residuals are pipeline-conditional EV3–Vive differences, not absolute robot accuracy "
    "or evidence of universal pipeline superiority."
)


PUBLICATION_CATALOG: dict[str, list[str]] = {
    "core": [
        "config_used.yaml",
        "config_hash.txt",
        "input_checksums.csv",
        "software_environment.txt",
        "software_manifest.json",
        "pipeline_stage_log.csv",
        "01_manifest/**/*",
        "04_quality/**/*",
        "alignment_parameters.json",
        "13_integrated_logs/ct_reference/**/*",
        "14_metrics/evaluation_metrics.csv",
        "14_metrics/per_trial_metrics.csv",
        "14_metrics/per_phase_metrics.csv",
        "14_metrics/aggregated_metrics.json",
        "provenance/**/*",
    ],
    "h1": [
        "15_hypotheses/H1/**/*", "17_statistics/signed_rank_audits/*_H1_rows.csv",
        "publication_work/h1/**/*",
    ],
    "h2": [
        "15_hypotheses/H2/**/*", "17_statistics/h2_pairwise_*.csv",
        "publication_work/h2/**/*",
    ],
    "h3": [
        "15_hypotheses/H3/**/*", "17_statistics/signed_rank_audits/*_H3_rows.csv",
        "publication_work/h3/**/*",
    ],
    "baseline": ["baseline/**/*"],
    "diagnostics": [
        "00_experiment_design/**/*",
        "13_integrated_logs/sampling_strategy_registry.csv",
        "13_integrated_logs/timestamp_matching_report.csv",
        "13_integrated_logs/query_time_report.csv",
        "13_integrated_logs/common_sampling_stage_comparison.csv",
        "14_metrics/cascade_per_trial_metrics.csv",
        "14_metrics/cascade_per_stage_summary.csv",
        "17_statistics/repeatability_*.csv",
        "17_statistics/alignment_parameter_stability.csv",
        "17_statistics/*scenario*.csv",
        "17_statistics/*variability*.csv",
        "17_statistics/variance_decomposition.csv",
        "17_statistics/method_sensitivity_status.json",
        "18_plots/**/*",
        "19_tables/authoritative_population_gate_ledger.csv",
    ],
    "figures": ["paper_figures/**/*"],
    "tables": ["19_tables/table_*.csv"],
    "report": ["report.md", "report.html"],
}

EXCLUDED_PUBLICATION_PATHS = frozenset({
    "17_statistics/scenario_level_method_sensitivity.csv",
    "18_plots/scenario_grid/method_sensitivity_9x9_heatmap.png",
})


def publish_selected(selected: set[str], catalog: dict[str, list[str]]) -> list[str]:
    """Return catalog entries belonging only to selected public families."""
    return [name for family in sorted(selected) for name in catalog.get(family, [])]


def _requested_families(requested: frozenset[str]) -> set[str]:
    return set(ALL if "all" in requested else requested) | {"core"}


def _sources_for_pattern(run_dir: Path, pattern: str) -> list[Path]:
    if any(character in pattern for character in "*?["):
        candidates = sorted(run_dir.glob(pattern))
    else:
        candidate = run_dir / pattern
        candidates = [candidate] if candidate.exists() else []
    return [
        path
        for path in candidates
        if path.is_file()
        and path.relative_to(run_dir).parts[0] != "results"
        and path.relative_to(run_dir).as_posix() not in EXCLUDED_PUBLICATION_PATHS
    ]


def _csv_has_rows(path: Path) -> bool:
    with path.open(newline="", encoding="utf-8") as stream:
        return next(csv.DictReader(stream), None) is not None


def _materialize_hypothesis_summaries(run_dir: Path, families: set[str]) -> None:
    summary_specs = (
        ("hypothesis_test_summary.csv", "test"),
        ("hypothesis_decision_summary.csv", "hypothesis_id"),
        ("signed_rank_audit_summary.csv", "hypothesis_id"),
        ("sensitivity_planned_vs_extended.csv", "hypothesis_id"),
    )
    for family in sorted(families & {"h1", "h2", "h3"}):
        hypothesis = family.upper()
        destination = run_dir / "publication_work" / family
        destination.mkdir(parents=True, exist_ok=True)
        for filename, column in summary_specs:
            source = run_dir / "17_statistics" / filename
            if not source.is_file():
                continue
            with source.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                rows = [
                    row
                    for row in reader
                    if _hypothesis_family(str(row.get(column, ""))) == hypothesis
                ]
                fieldnames = reader.fieldnames or []
            if not rows:
                continue
            with (destination / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)


def _hypothesis_family(identifier: str) -> str:
    """Map persisted producer identifiers to public H-family routing only."""
    normalized = str(identifier).strip().upper()
    if normalized == "H2_FRIEDMAN":
        return "H2"
    return normalized


def _validate_family(run_dir: Path, family: str, sources: list[Path], *, strict: bool) -> None:
    if not strict:
        return
    if family == "core":
        metrics = run_dir / "14_metrics" / "per_trial_metrics.csv"
        if not metrics.is_file() or not _csv_has_rows(metrics):
            raise RuntimeError("selected core product has no held-out trial metrics")
        return
    if family in {"h1", "h2", "h3"}:
        comparison_names = {
            "h1": "h1_trial_pairs.csv",
            "h2": "h2_trial_comparison.csv",
            "h3": "h3_calibration_mode_comparison.csv",
        }
        comparisons = [path for path in sources if path.name == comparison_names[family]]
        summaries = [path for path in sources if path.name == "hypothesis_test_summary.csv"]
        if not comparisons or not _csv_has_rows(comparisons[0]) or not summaries or not _csv_has_rows(summaries[0]):
            raise RuntimeError(f"selected {family} product lacks comparison or inferential summary rows")
        return
    if family == "baseline":
        trial = run_dir / "baseline" / "baseline_trial_metrics.csv"
        summary = run_dir / "baseline" / "baseline_summary.json"
        if not trial.is_file() or not _csv_has_rows(trial):
            raise RuntimeError("selected baseline trial metrics are missing or empty")
        if not summary.is_file() or not json.loads(summary.read_text(encoding="utf-8")):
            raise RuntimeError("selected baseline summary is missing or empty")
        return
    if family == "report":
        report = run_dir / "report.md"
        if not report.is_file() or REPORT_LIMITATION not in report.read_text(encoding="utf-8"):
            raise RuntimeError("selected report lacks the required pipeline-conditional limitation")
        return
    if family == "tables":
        expected = {
            "table_1_dataset_summary.csv",
            "table_2_quality_filtering.csv",
            "table_3_spatial_diagnostics.csv",
            "table_4_temporal_alignment.csv",
            "table_5_representation_methods.csv",
            "table_6_calibration_modes.csv",
            "table_7_hypothesis_results.csv",
            "table_8_selected_pipeline_configuration.csv",
            "table_9_baseline_vs_proposed_pipeline.csv",
            "table_10_paired_pipeline_difference_summary.csv",
        }
        tables = {path.name: path for path in sources if path.name in expected}
        if set(tables) != expected or any(not _csv_has_rows(path) for path in tables.values()):
            raise RuntimeError("selected tables product requires all ten nonempty current-run tables")
        return
    if family != "figures":
        return
    rendered = [path for path in sources if path.suffix.lower() in {".png", ".pdf", ".svg"}]
    manifest = run_dir / "paper_figures" / "figure_manifest.csv"
    if not rendered or not manifest.is_file():
        raise RuntimeError("selected figures product has no rendered current-run image")
    with manifest.open(newline="", encoding="utf-8") as stream:
        valid_rows = {
            row.get("figure_id", ""): row.get("file_name", "")
            for row in csv.DictReader(stream)
            if str(row.get("valid_for_paper", "")).strip().lower() in {"1", "true", "yes"}
        }
    valid_names = set(valid_rows.values())
    if not any(path.name in valid_names for path in rendered):
        raise RuntimeError("selected figures product has no rendered current-run image marked valid")
    required = {
        "fig_h1_current_run",
        "fig_h2_current_run",
        "fig_h3_current_run",
        "fig_baseline_current_run",
    }
    rendered_names = {path.name for path in rendered}
    if not required <= set(valid_rows) or any(valid_rows[figure_id] not in rendered_names for figure_id in required):
        raise RuntimeError("selected figures product requires rendered current-run H1/H2/H3/baseline images")


def publish_run_results(
    run_dir: Path,
    requested: frozenset[str],
    *,
    catalog: dict[str, list[str]] | None = None,
) -> list[Path]:
    """Copy selected artifacts to ``results/<family>/`` without changing bytes."""
    run_dir = Path(run_dir).resolve()
    selected_catalog = PUBLICATION_CATALOG if catalog is None else catalog
    families = _requested_families(requested)
    if catalog is None:
        _materialize_hypothesis_summaries(run_dir, families)
    results_dir = run_dir / "results"
    if results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty publication destination: {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)

    published: list[Path] = []
    guide_rows: list[tuple[str, str]] = []
    for family in sorted(families):
        patterns = selected_catalog.get(family, [])
        sources: list[Path] = []
        for pattern in patterns:
            sources.extend(_sources_for_pattern(run_dir, pattern))
        sources = sorted(set(sources), key=lambda path: path.relative_to(run_dir).as_posix())
        if not sources:
            if family in selected_catalog:
                raise RuntimeError(f"selected result family has no generated artifacts: {family}")
            continue
        _validate_family(run_dir, family, sources, strict=catalog is None)
        for source in sources:
            if source.stat().st_size == 0:
                raise RuntimeError(f"selected result artifact is empty: {source.relative_to(run_dir)}")
            relative = source.relative_to(run_dir)
            target = results_dir / family / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if target.read_bytes() != source.read_bytes():
                raise RuntimeError(f"published bytes differ from source: {relative}")
            published.append(target)
            guide_rows.append((family, relative.as_posix()))

    guide = results_dir / "core" / "OUTPUT_GUIDE.md"
    guide.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OFF-STAP output guide",
        "",
        "Files are grouped by requested public result family and retain their current-run relative paths.",
        "",
        f"> {REPORT_LIMITATION}",
        "",
        "| Family | Current-run artifact |",
        "| --- | --- |",
        *(f"| `{family}` | `{relative}` |" for family, relative in guide_rows),
        "",
    ]
    guide.write_text("\n".join(lines), encoding="utf-8")
    published.append(guide)
    return sorted(published, key=lambda path: path.relative_to(run_dir).as_posix())


def write_result_index(run_dir: Path, selected: frozenset[str], files: list[Path]) -> Path:
    """Write requested/effective selectors and SHA-256 for published bytes."""
    run_dir = Path(run_dir).resolve()
    index = run_dir / "results" / "index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "requested_results": sorted(selected),
        "effective_results": sorted(expand_results(selected)),
        "files": [
            {
                "path": path.resolve().relative_to(run_dir).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(files, key=lambda item: item.resolve().relative_to(run_dir).as_posix())
        ],
    }
    index.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return index


__all__ = [
    "PUBLICATION_CATALOG",
    "REPORT_LIMITATION",
    "publish_run_results",
    "publish_selected",
    "write_result_index",
]
