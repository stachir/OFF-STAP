# OFF-STAP

OFF-STAP is a failure-explicit command-line application for auditable offline spatiotemporal alignment of heterogeneous robot telemetry and motion-tracking logs. It pairs LEGO EV3 telemetry with HTC Vive Tracker measurements, validates the inputs, estimates temporal and spatial mappings, constructs a continuous-time reference, evaluates held-out residuals, and records provenance for every generated result.

This repository contains the curated public application. It does not contain the private development history, manuscript-working files, generated study outputs, or raw experimental logs. The original logs will be published separately by the authors; no dataset URL is stated here until that repository or deposition has a stable public address.

## Scientific scope and limitations

The packaged configuration reproduces the approved OFF-STAP processing contract. A custom configuration may change operational paths and its `config_id`, but the application rejects changes to scientific parameters. This prevents an altered threshold or method from being presented as the same analysis.

Residuals are pipeline-conditional EV3–Vive differences, not absolute robot accuracy or evidence of universal pipeline superiority. The application is intended for offline analysis of logs that satisfy the documented schema. It does not perform online localization or robot control.

## Requirements

- Python 3.14
- Windows, Linux, or macOS
- Enough disk space for the input logs, intermediate Parquet files, selected result families, and plots

Automated tests are configured for all three operating systems in GitHub Actions. A green workflow run is the release-level evidence for platform compatibility; the workflow configuration alone is not a claim that a particular commit passed remotely.

## Installation

Clone or download the repository, then open a terminal in its root.

### Windows PowerShell

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
offstap --help
```

### Linux and macOS

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
offstap --help
```

The module entry point is equivalent:

```bash
python -m offstap --help
```

## Preparing input files

Keep the two sensor streams in separate directories. The output directory must be new or empty and must not overlap either input directory.

```text
my-analysis/
├── input/
│   ├── ev3/
│   │   ├── D1_T1_2025-06-12_13-08-05_ev3.csv
│   │   └── ...
│   └── vive/
│       ├── D1_T1_2025-06-12_13-08-05_vrmt.csv
│       └── ...
└── output/                 # create this as empty, or let OFF-STAP create it
```

Matching files share the drive, route, date, and time prefix. The final stream token is `ev3` or `vrmt`.

The default EV3 schema expects:

| Meaning | Default column | Accepted aliases | Default unit |
| --- | --- | --- | --- |
| Time | `time` | `time_ms` | milliseconds |
| Motor speed | `motor_speed` | ` EV3 state speed` | source-log unit |
| Motor angle | `motor_angle` | ` EV3 state angle` | source-log unit |
| Gyroscope heading | `gyro_heading` | ` Gyro(deg)` | degrees |

The default Vive schema expects:

| Meaning | Default column | Accepted alias | Default unit |
| --- | --- | --- | --- |
| Time | `timestamp` | `Time` | seconds |
| Position | `x`, `y`, `z` | `PoseX`, `PoseY`, `PoseZ` | metres |
| Orientation quaternion | `qx`, `qy`, `qz`, `qw` | — | unit quaternion |

Do not commit private logs to this repository. The supplied `.gitignore` excludes `data/` and `outputs/` by default.

## Running OFF-STAP

A minimal core run:

```bash
offstap \
  --ev3-dir input/ev3 \
  --vive-dir input/vive \
  --output-dir output/core-run \
  --results core
```

PowerShell accepts the same command on one line:

```powershell
offstap --ev3-dir input\ev3 --vive-dir input\vive --output-dir output\core-run --results core
```

Generate every available result family:

```bash
offstap --ev3-dir input/ev3 --vive-dir input/vive --output-dir output/full-run --results all
```

Select several families with a comma-separated value:

```bash
offstap --ev3-dir input/ev3 --vive-dir input/vive --output-dir output/selected-run --results h1,h2,baseline,tables
```

### Command-line options

| Option | Required | Meaning |
| --- | --- | --- |
| `--ev3-dir PATH` | yes | Directory containing EV3 CSV logs. |
| `--vive-dir PATH` | yes | Directory containing Vive/VRMT CSV logs. |
| `--output-dir PATH` | yes | New or empty run directory. It cannot overlap an input directory. |
| `--config PATH` | no | YAML configuration. Defaults to the packaged approved configuration; scientific differences are rejected. |
| `--results NAMES` | no | Comma-separated result selectors, or `all`. Defaults to `core`. |

### Result selectors

| Selector | Published result |
| --- | --- |
| `core` | Pairing, quality, alignment, held-out metrics, configuration, software environment, checksums, stage log, and provenance. |
| `h1` | Current-run paired temporal-refinement comparison and H1 inferential summary. |
| `h2` | Current-run comparison of exactly `zoh`, `linear`, `pchip`, and `spline` on common support, plus H2 inference. |
| `h3` | Current-run calibration-mode comparison and H3 inferential summary. |
| `baseline` | Corrected current-run engineering-baseline trial metrics, exclusions, summaries, and provenance. |
| `diagnostics` | Design balance, repeatability, scenario robustness, population gates, cascade diagnostics, and sampling records. |
| `figures` | Newly rendered current-run figures and their source extracts/manifest. |
| `tables` | Ten newly generated current-run summary tables. |
| `report` | Current-run Markdown and HTML reports with the required interpretation limitation. |
| `all` | Every public result family. |

Dependencies are expanded automatically. For example, presentation products require the current run's numerical and diagnostic sources. The result index records the user's requested selectors separately from the effective dependency closure.

Method sensitivity is explicitly reported as `NOT_AVAILABLE`. The approved processing contract does not define a valid current-run between-method sensitivity computation, so OFF-STAP does not generate placeholder sensitivity values or relabel normalized residuals as sensitivity.

## Processing contract

The stage log uses these eight public labels:

1. Ingest, identify, pair, and tier
2. Validate, normalize, and gate
3. Partition calibration/evaluation
4. Estimate temporal offset
5. Select clock mapping
6. Fit spatial transform and build reference
7. Query held-out support and verify
8. Compare, infer, and package

Steps 4–6 are recorded as explicit methodological substeps even where one implementation stage prepares more than one parameter family. Numerical comparisons are made only after held-out support has passed the required gates.

## Understanding the output

The run directory contains internal stage artifacts and a publication-oriented `results/` directory. Use `results/` when sharing selected outputs.

Important top-level records include:

| File | Meaning |
| --- | --- |
| `run_status.json` | `RUNNING`, `PASS`, or `FAILED`, with a reason when applicable. |
| `pipeline_stage_log.csv` | Ordered processing-contract steps, timestamps, status, and errors. |
| `config_used.yaml` | Effective path-resolved configuration used for the run. |
| `config_hash.txt` | SHA-256 of `config_used.yaml`. |
| `input_checksums.csv` | Input file sizes and SHA-256 values. |
| `software_environment.txt` | Python and operating-system information. |
| `software_manifest.json` | Package and dependency versions. |
| `results/index.json` | Requested/effective selectors and SHA-256 for every published file. |
| `results/core/OUTPUT_GUIDE.md` | Human-readable list of published files grouped by family. |

Common scientific artifacts include:

| Path | Meaning |
| --- | --- |
| `01_manifest/pair_manifest.csv` | Discovered streams, pairing, tier, and analysis-set identity. |
| `04_quality/` | Quality decisions and reason-coded rejections. |
| `alignment_parameters.json` | Selected temporal, clock, and spatial parameters. |
| `13_integrated_logs/ct_reference/` | Integrated held-out trajectories on continuous-time reference support. |
| `14_metrics/per_trial_metrics.csv` | Per-trial residual metrics and hypothesis eligibility. |
| `15_hypotheses/H1/`, `H2/`, `H3/` | Current-run hypothesis comparison products. |
| `baseline/` | Current-run baseline comparison, exclusions, summaries, and hashes. |
| `17_statistics/` | Current-run inferential and diagnostic summaries. |
| `19_tables/table_*.csv` | Ten presentation tables generated from the same run. |
| `paper_figures/` | Rendered figures, source extracts, and figure manifest. |
| `report.md`, `report.html` | Human-readable run report. |

Published files retain their internal relative paths beneath `results/<family>/`. The SHA-256 values in `results/index.json` are calculated after copying and can be used to verify that shared bytes match the run artifacts.

Support hashes identify the exact finite common-support rows used in paired comparisons. A mismatch, missing arm, missing required artifact, empty selected family, or non-finite score causes explicit failure rather than silent substitution.

## Reproducibility and failure behavior

- Start each analysis in a new or empty output directory. OFF-STAP refuses to overwrite a nonempty publication destination or baseline result directory.
- Input, configuration, software, common-support, and output hashes are recorded where applicable.
- H2 has exactly four inferential arms: `zoh`, `linear`, `pchip`, and `spline`.
- Ineligible trials and baseline exclusions are reason-coded.
- A failed required stage writes `FAILED` to `run_status.json`; it is not reported as a partial success.
- If no trial survives a required gate, dependent result families are marked not applicable and the run fails explicitly.
- The application never falls back to a bundled historical output or manuscript figure.

## Troubleshooting

**The output directory is rejected.** Use a new directory, empty the intended directory yourself, and ensure it is outside both input trees.

**Files are not paired.** Check that the EV3 and Vive filenames have the same drive, route, date, and time prefix and end in `_ev3.csv` and `_vrmt.csv` respectively.

**The configuration is rejected.** The public application accepts operational path and `config_id` changes only. Restore the packaged scientific settings for a comparable OFF-STAP run.

**A selected family is missing.** Read `run_status.json`, `pipeline_stage_log.csv`, the quality/rejection records, and any `result_status.json`. Selected products fail closed when their current-run source evidence is absent or empty.

**Plots fail in a headless environment.** The runner sets Matplotlib's noninteractive `Agg` backend. Run through the `offstap` command rather than importing stage scripts manually.

## Development and verification

```bash
python -m pip install . pytest
python -m pytest -q
python -m offstap --help
```

The GitHub Actions workflow repeats installation, CLI help, and the test suite on Windows, Linux, and macOS using Python 3.14.

## Citation

If you use this application in research, cite the accompanying manuscript:

> *Auditable offline spatiotemporal alignment of heterogeneous telemetry and motion tracking logs*.

Publication metadata and a DOI are intentionally not invented here. Update the citation from the final published record when it becomes available.

## License

OFF-STAP is released under the [MIT License](LICENSE).
