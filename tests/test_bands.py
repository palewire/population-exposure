"""Tests for exposure band definitions."""

from __future__ import annotations

import math

import pytest

from population_exposure import ExposureBand, ExposureBands


def test_from_breaks_builds_lower_inclusive_intervals() -> None:
    bands = ExposureBands.from_breaks(
        [-2, 2],
        ids=("below", "near", "above"),
        labels=("Below", "Near", "Above"),
    )

    assert bands.bands == (
        ExposureBand("below", None, -2.0, "Below"),
        ExposureBand("near", -2.0, 2.0, "Near"),
        ExposureBand("above", 2.0, None, "Above"),
    )


def test_from_breaks_accepts_no_breaks() -> None:
    bands = ExposureBands.from_breaks([], ids=("all",))

    assert bands.bands == (ExposureBand("all", None, None),)


@pytest.mark.parametrize(
    ("breaks", "message"),
    [
        ([0, math.nan], "finite"),
        ([0, math.inf], "finite"),
        ([0, -math.inf], "finite"),
        ([0, 0], "strictly increasing"),
        ([1, 0], "strictly increasing"),
        ([0, "bad"], "only numbers"),
    ],
)
def test_from_breaks_rejects_invalid_breaks(breaks, message) -> None:
    with pytest.raises(ValueError, match=message):
        ExposureBands.from_breaks(breaks, ids=("a", "b", "c"))


def test_from_breaks_requires_one_more_id_than_breaks() -> None:
    with pytest.raises(ValueError, match="one more band ID"):
        ExposureBands.from_breaks([0], ids=("only",))


def test_from_breaks_requires_one_label_per_band() -> None:
    with pytest.raises(ValueError, match="one label"):
        ExposureBands.from_breaks([0], ids=("low", "high"), labels=("Low",))


@pytest.mark.parametrize(
    ("bands", "message"),
    [
        ((), "cannot be empty"),
        ((ExposureBand("", None, None),), "non-empty string ID"),
        ((ExposureBand(" ", None, None),), "non-empty string ID"),
        (
            (
                ExposureBand("same", None, 0),
                ExposureBand("same", 0, None),
            ),
            "must be unique",
        ),
        ((ExposureBand("all", math.nan, None),), "finite"),
        ((ExposureBand("all", None, math.inf),), "finite"),
        ((ExposureBand("all", "bad", None),), "numbers or None"),
        ((ExposureBand("zero", 1, 1),), "less than"),
        ((ExposureBand("reversed", 2, 1),), "less than"),
        ((ExposureBand("low", 0, None),), "negative infinity"),
        ((ExposureBand("high", None, 0),), "positive infinity"),
        (
            (
                ExposureBand("low", None, -1),
                ExposureBand("high", 0, None),
            ),
            "have a gap",
        ),
        (
            (
                ExposureBand("low", None, 1),
                ExposureBand("high", 0, None),
            ),
            "overlap",
        ),
        (
            (
                ExposureBand("low", None, None),
                ExposureBand("high", 0, None),
            ),
            "Only the last",
        ),
        (
            (
                ExposureBand("low", None, 0),
                ExposureBand("high", None, None),
            ),
            "Only the first",
        ),
    ],
)
def test_explicit_bands_reject_invalid_collections(bands, message) -> None:
    with pytest.raises(ValueError, match=message):
        ExposureBands(bands)
