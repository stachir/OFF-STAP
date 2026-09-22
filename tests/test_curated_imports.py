from pathlib import Path


def test_curated_modules_import_without_research_checkout():
    """A copied calculation module must be importable from the public package."""
    from offstap.core.baseline_comparator import score_paired_frames
    from offstap.core.spatial import estimate_spatial_transform
    from offstap.core.support import intersect_support

    assert callable(estimate_spatial_transform)
    assert callable(intersect_support)
    assert callable(score_paired_frames)


def test_curated_figure_module_imports_without_research_checkout():
    """Figure helpers must use the public package namespace."""
    from offstap.figures.generate_figures import generate_paper_figures, plot_helpers

    assert callable(generate_paper_figures)
    assert plot_helpers is not None


def test_forbidden_research_names_are_absent_from_package_sources():
    """The distributable package must not expose research-only identifiers."""
    public = Path(__file__).resolve().parents[1] / "offstap"
    text = "\n".join(path.read_text(encoding="utf-8") for path in public.rglob("*.py"))
    forbidden = (
        "run" + "_202",
        "BASELINE" + "_COMPARATOR_CORRECTION_V1",
        "C:" + "\\Projekty",
        "c:" + "\\Projekty",
        "PERSISTENCE_" + ("V" + "3"),
        "AGENTS" + ".md",
    )

    assert not any(name in text for name in forbidden)
