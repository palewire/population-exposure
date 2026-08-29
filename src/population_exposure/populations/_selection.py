"""Exact source and year selection parsing."""

from __future__ import annotations

import re

from population_exposure.populations._sources import SOURCES, SourceSpec

_SELECTION_PATTERN = re.compile(
    r"(?P<source>[a-z0-9]+(?:-[a-z0-9]+)*):(?P<year>[0-9]{4}|latest)"
)


def parse_selection(selection: str) -> tuple[SourceSpec, int]:
    """Parse and validate one exact ``source:year`` selection."""
    if not isinstance(selection, str):
        raise TypeError("population selection must be a string.")
    if selection.endswith(":latest"):
        raise ValueError(
            "Population selections never use 'latest'; choose an explicit year, "
            "for example 'worldpop-global-1km:2020'."
        )
    if selection in SOURCES:
        raise ValueError(
            f"Population source {selection!r} requires an explicit year, for example "
            f"{selection!r} + ':2020'."
        )

    match = _SELECTION_PATTERN.fullmatch(selection)
    if match is None:
        raise ValueError(
            "Population selection must exactly match 'source-id:YYYY'; use "
            "populations.list() to inspect supported source IDs and years."
        )

    source_id = match.group("source")
    source = SOURCES.get(source_id)
    if source is None:
        available = ", ".join(sorted(SOURCES))
        raise ValueError(
            f"Unknown population source {source_id!r}; available sources: {available}."
        )

    year = int(match.group("year"))
    if year not in source.years:
        if source.years == tuple(range(source.years[0], source.years[-1] + 1)):
            supported = f"{source.years[0]}-{source.years[-1]}"
        else:
            supported = ", ".join(str(value) for value in source.years)
        raise ValueError(
            f"{source_id!r} does not support {year}; supported years: {supported}."
        )
    return source, year
