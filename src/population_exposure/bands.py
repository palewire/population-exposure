"""Exposure band definitions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class ExposureBand:
    """One lower-inclusive, upper-exclusive hazard interval."""

    id: str
    lower_bound: float | None
    upper_bound: float | None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class ExposureBands:
    """An ordered set of exhaustive, non-overlapping exposure bands."""

    bands: tuple[ExposureBand, ...]

    def __post_init__(self) -> None:
        """Validate explicit band definitions."""
        if not self.bands:
            raise ValueError("Exposure bands cannot be empty.")

        ids = [band.id for band in self.bands]
        if any(not isinstance(band_id, str) or not band_id.strip() for band_id in ids):
            raise ValueError("Every exposure band must have a non-empty string ID.")
        if len(ids) != len(set(ids)):
            raise ValueError("Exposure band IDs must be unique.")

        for band in self.bands:
            _validate_bound(band.lower_bound)
            _validate_bound(band.upper_bound)
            if (
                band.lower_bound is not None
                and band.upper_bound is not None
                and band.lower_bound >= band.upper_bound
            ):
                raise ValueError(
                    f"Exposure band {band.id!r} must have a lower bound "
                    "that is less than its upper bound."
                )

        if self.bands[0].lower_bound is not None:
            raise ValueError("The first exposure band must start at negative infinity.")
        if self.bands[-1].upper_bound is not None:
            raise ValueError("The last exposure band must end at positive infinity.")

        for previous, current in pairwise(self.bands):
            if previous.upper_bound is None:
                raise ValueError("Only the last exposure band may have no upper bound.")
            if current.lower_bound is None:
                raise ValueError(
                    "Only the first exposure band may have no lower bound."
                )
            if previous.upper_bound < current.lower_bound:
                raise ValueError(
                    f"Exposure bands {previous.id!r} and {current.id!r} have a gap."
                )
            if previous.upper_bound > current.lower_bound:
                raise ValueError(
                    f"Exposure bands {previous.id!r} and {current.id!r} overlap."
                )

    @classmethod
    def from_breaks(
        cls,
        breaks: Sequence[float],
        *,
        ids: Sequence[str],
        labels: Sequence[str] | None = None,
    ) -> Self:
        """Create exhaustive bands from finite, increasing break values."""
        try:
            normalized_breaks = tuple(float(value) for value in breaks)
        except (TypeError, ValueError) as exc:
            raise ValueError("Breaks must contain only numbers.") from exc

        if any(not math.isfinite(value) for value in normalized_breaks):
            raise ValueError("Breaks must contain only finite numbers.")
        if any(lower >= upper for lower, upper in pairwise(normalized_breaks)):
            raise ValueError("Breaks must be strictly increasing.")

        normalized_ids = tuple(ids)
        if len(normalized_ids) != len(normalized_breaks) + 1:
            raise ValueError("Provide exactly one more band ID than break values.")

        normalized_labels: tuple[str | None, ...]
        if labels is None:
            normalized_labels = (None,) * len(normalized_ids)
        else:
            normalized_labels = tuple(labels)
            if len(normalized_labels) != len(normalized_ids):
                raise ValueError("Provide exactly one label for each band.")

        lower_bounds = (None, *normalized_breaks)
        upper_bounds = (*normalized_breaks, None)
        return cls(
            tuple(
                ExposureBand(band_id, lower, upper, label)
                for band_id, lower, upper, label in zip(
                    normalized_ids,
                    lower_bounds,
                    upper_bounds,
                    normalized_labels,
                    strict=True,
                )
            )
        )


def _validate_bound(bound: float | None) -> None:
    if bound is None:
        return
    try:
        finite = math.isfinite(bound)
    except TypeError as exc:
        raise ValueError("Exposure band bounds must be numbers or None.") from exc
    if not finite:
        raise ValueError("Exposure band bounds must be finite numbers or None.")
