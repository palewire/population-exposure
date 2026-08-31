"""Discover, acquire, and register curated population-count rasters."""

from population_exposure.populations._api import download, info, list, register
from population_exposure.populations._models import SelectionInfo, SourceInfo

__all__ = [
    "SelectionInfo",
    "SourceInfo",
    "download",
    "info",
    "list",
    "register",
]
