"""Public repository documentation, packaging, and hygiene contracts."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_installation_inputs_processing_and_every_result_flag():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for platform in ("Windows", "Linux", "macOS"):
        assert platform in readme
    for flag in (
        "--ev3-dir", "--vive-dir", "--config", "--output-dir", "--results",
    ):
        assert flag in readme
    for result in (
        "core", "h1", "h2", "h3", "baseline", "diagnostics",
        "figures", "tables", "report", "all",
    ):
        assert f"`{result}`" in readme
    for step in range(1, 9):
        assert f"{step}. " in readme
    assert "published separately" in readme
    assert "pip install ." in readme
    assert "results/index.json" in readme


def test_mit_license_and_package_metadata_are_public_ready():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Copyright (c) 2026 OFF-STAP contributors" in license_text
    assert 'license = "MIT"' in pyproject
    assert 'readme = "README.md"' in pyproject
    assert "https://github.com/stachir/OFF-STAP" in pyproject


def test_ci_covers_three_operating_systems_with_python_314():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8"))
    job = workflow["jobs"]["portable"]
    assert set(job["strategy"]["matrix"]["os"]) == {
        "windows-latest", "ubuntu-latest", "macos-latest",
    }
    rendered = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    assert "actions/checkout@v7" in rendered
    assert "actions/setup-python@v7" in rendered
    assert "python-version: '3.14'" in rendered
    assert "python -m pip install ." in rendered
    assert "python -m pytest -q" in rendered


def test_gitignore_excludes_user_data_outputs_and_local_environment():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/", "outputs/", ".venv/", "__pycache__/", ".env", "*.egg-info/"):
        assert pattern in ignored


def test_public_tree_contains_no_private_handoff_or_historical_run_material():
    paths = [path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()]
    assert not any(path.endswith("AGENTS.md") or "CODEX_START_HERE" in path for path in paths)
    assert not any("FULL_CORRECTION" in path or "PERSISTENCE_V3" in path for path in paths)
    assert not any("reviewer" in path.lower() or ".superpowers" in path for path in paths)
    assert not any(path.startswith("data/") or path.startswith("outputs/") for path in paths)


def test_public_source_has_no_local_workspace_paths_or_historical_release_labels():
    excluded = {"test_release_contract.py"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name in excluded or any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix.lower() not in {".py", ".md", ".toml", ".yaml", ".yml", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "C:\\Projekty\\" not in text
        assert "PERSISTENCE_V3" not in text
        assert "FULL_CORRECTION" not in text
        assert "BASELINE_COMPARATOR_CORRECTION_V1" not in text
