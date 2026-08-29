"""Typed catalog records exposed by :mod:`population_exposure.populations`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Acquisition = Literal["automatic", "earthdata", "manual"]
PopulationMeaning = Literal["residential", "ambient"]


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Stable facts about one built-in population source."""

    source_id: str
    release: str
    title: str
    publisher: str
    supported_years: tuple[int, ...]
    acquisition: Acquisition
    landing_page: str
    doi: str | None
    license: str
    citation: str
    units: str
    meaning: PopulationMeaning
    crs: str
    resolution: str
    format: str
    download_size: str
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SelectionInfo:
    """Resolved facts for one explicit source and year selection."""

    selection: str
    source_id: str
    release: str
    year: int
    title: str
    publisher: str
    acquisition: Acquisition
    landing_page: str
    official_url: str
    doi: str | None
    license: str
    citation: str
    units: str
    meaning: PopulationMeaning
    crs: str
    resolution: str
    format: str
    download_size: str
    expected_filename: str
    notes: tuple[str, ...]
