"""Validation and dependency expansion for public result selections."""

NUMERIC = frozenset({"core", "h1", "h2", "h3", "baseline"})
PRESENTATION = frozenset({"figures", "tables", "report"})
ALL = NUMERIC | {"diagnostics"} | PRESENTATION
VALID_RESULTS = ALL
H2_METHODS = ("zoh", "linear", "pchip", "spline")


def parse_results(value: str) -> frozenset[str]:
    """Parse a comma-separated public result selection or the ``all`` shortcut."""
    names = frozenset(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise ValueError("at least one result is required")
    if names == {"all"}:
        return names
    if "all" in names:
        raise ValueError("all cannot be combined with other results")
    unknown = names - VALID_RESULTS
    if unknown:
        raise ValueError(f"unknown result: {', '.join(sorted(unknown))}")
    return names


def expand_results(requested: frozenset[str]) -> frozenset[str]:
    """Return the effective closure needed to produce requested results."""
    selected = set(ALL if "all" in requested else requested)
    unknown = selected - ALL
    if unknown:
        raise ValueError(f"unknown result: {sorted(unknown)}")
    selected.add("core")
    if selected & PRESENTATION:
        selected.update(NUMERIC | {"diagnostics"})
    return frozenset(selected)
