"""Public labels and execution order for the frozen core processing stages."""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from offstap.selection import expand_results


@dataclass(frozen=True)
class Stage:
    step: int
    label: str
    module: str
    substeps: tuple[int, ...] = ()


@dataclass(frozen=True)
class CallableProducer:
    """In-process producer positioned relative to the subprocess stage plan."""

    product: str
    after_module: str
    module: str
    function: str
    output_dir: str

    def run(self, source_run: Path) -> dict[str, Path]:
        producer = getattr(import_module(self.module), self.function)
        return producer(source_run, source_run / self.output_dir)


STEP_LABELS = {
    1: "Ingest, identify, pair, and tier",
    2: "Validate, normalize, and gate",
    3: "Partition calibration/evaluation",
    4: "Estimate temporal offset",
    5: "Select clock mapping",
    6: "Fit spatial transform and build reference",
    7: "Query held-out support and verify",
    8: "Compare, infer, and package",
}


CORE_STAGES: tuple[Stage, ...] = (
    Stage(1, STEP_LABELS[1], "offstap.stages.s01_manifest"),
    Stage(2, STEP_LABELS[2], "offstap.stages.s02_validate"),
    Stage(2, STEP_LABELS[2], "offstap.stages.s03_quality"),
    Stage(3, STEP_LABELS[3], "offstap.stages.s04_partition"),
    Stage(4, STEP_LABELS[4], "offstap.stages.s05_parameters", (5, 6)),
    Stage(6, STEP_LABELS[6], "offstap.stages.s06_reference"),
    Stage(7, STEP_LABELS[7], "offstap.stages.s07_align"),
    Stage(7, STEP_LABELS[7], "offstap.stages.s08_metrics"),
)


COMPARISON_STAGE = Stage(8, STEP_LABELS[8], "offstap.stages.s09_comparisons")


DIAGNOSTIC_STAGES: tuple[Stage, ...] = (
    Stage(8, STEP_LABELS[8], "offstap.stages.s10_plots"),
    Stage(8, STEP_LABELS[8], "offstap.stages.s11_dataset_design"),
    Stage(8, STEP_LABELS[8], "offstap.stages.s12_repeatability"),
    Stage(8, STEP_LABELS[8], "offstap.stages.s13_scenario_robustness"),
    Stage(8, STEP_LABELS[8], "offstap.stages.s14b_population_gate"),
    Stage(8, STEP_LABELS[8], "offstap.stages.s14a_cascade"),
    Stage(8, STEP_LABELS[8], "offstap.stages.s14_cascade"),
    Stage(8, STEP_LABELS[8], "offstap.stages.s15_common_sampling"),
)


STATISTICS_STAGE = Stage(8, STEP_LABELS[8], "offstap.stages.s16b_statistics")
FIGURE_STAGE = Stage(8, STEP_LABELS[8], "offstap.figures.generate_figures")
TABLE_STAGE = Stage(8, STEP_LABELS[8], "offstap.stages.s18_tables")
REPORT_STAGE = Stage(8, STEP_LABELS[8], "offstap.stages.s19_report")


HYPOTHESIS_STAGES: tuple[Stage, ...] = (COMPARISON_STAGE, STATISTICS_STAGE)


BASELINE_PRODUCER = CallableProducer(
    product="baseline",
    after_module=CORE_STAGES[-1].module,
    module="offstap.baseline",
    function="run_baseline",
    output_dir="baseline",
)

CALLABLE_PRODUCERS: tuple[CallableProducer, ...] = (BASELINE_PRODUCER,)


def callable_producers_after(
    module: str,
    selected: frozenset[str],
) -> tuple[CallableProducer, ...]:
    """Return selected in-process producers registered after one core stage."""
    effective = expand_results(selected)
    return tuple(
        producer
        for producer in CALLABLE_PRODUCERS
        if producer.after_module == module and producer.product in effective
    )


def build_plan(requested: frozenset[str]) -> tuple[Stage, ...]:
    """Build the importable stage plan for the effective result selection."""
    selected = expand_results(requested)
    plan = list(CORE_STAGES)
    if selected & {"h1", "h2", "h3"}:
        plan.append(COMPARISON_STAGE)
    if "diagnostics" in selected:
        plan.extend(DIAGNOSTIC_STAGES)
    if selected & {"h1", "h2", "h3"}:
        plan.append(STATISTICS_STAGE)
    if "figures" in selected:
        plan.append(FIGURE_STAGE)
    if selected & {"tables", "report"}:
        plan.append(TABLE_STAGE)
    if "report" in selected:
        plan.append(REPORT_STAGE)
    return tuple(plan)
