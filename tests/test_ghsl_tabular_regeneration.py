"""Unit coverage for the GHSL tabular fixture regeneration helpers."""

from __future__ import annotations

import zipfile

import pytest

from scripts.regenerate_ghsl_tabular_golden import (
    WORKBOOK_COLUMNS,
    _category_for_smod,
    _workbook_header,
    extract_member,
    workbook_totals,
)


@pytest.mark.unit
def test_workbook_header_uses_column_letters_not_dict_insertion_order() -> None:
    """Require ordered header validation independent of XML cell ordering.

    Args:
        None.

    Returns:
        None.

    Examples:
        >>> test_workbook_header_uses_column_letters_not_dict_insertion_order()
    """
    row = {
        "D": "DEGURBA_L1",
        "B": "GADM_ISO",
        "A": "GADM_ID",
        "C": "GADM_NAME",
        "P": "2030",
        "O": "2025",
        "N": "2020",
        "M": "2015",
        "L": "2010",
        "K": "2005",
        "J": "2000",
        "I": "1995",
        "H": "1990",
        "G": "1985",
        "F": "1980",
        "E": "1975",
    }

    assert _workbook_header(row) == WORKBOOK_COLUMNS


@pytest.mark.unit
def test_workbook_totals_requires_the_expected_workbook_member(tmp_path) -> None:
    """Report a missing outer workbook member as a schema error.

    Args:
        tmp_path: Temporary test directory supplied by pytest.

    Returns:
        None.

    Examples:
        >>> test_workbook_totals_requires_the_expected_workbook_member(None)
    """
    archive_path = tmp_path / "country-stats.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("not-the-workbook.txt", "missing")

    with pytest.raises(ValueError, match="Archive lacks required member"):
        workbook_totals(archive_path)


@pytest.mark.unit
def test_extract_member_normalizes_safe_backslash_paths(tmp_path) -> None:
    """Extract a slash-normalized ZIP member when given a backslash path.

    Args:
        tmp_path: Temporary test directory supplied by pytest.

    Returns:
        None.

    Examples:
        >>> test_extract_member_normalizes_safe_backslash_paths(None)
    """
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("layers/grid.tif", "source")

    destination = tmp_path / "grid.tif"
    assert extract_member(archive_path, r"layers\grid.tif", destination) == destination
    assert destination.read_text() == "source"


@pytest.mark.unit
def test_category_for_smod_skips_unclassified_zero() -> None:
    """Allow a zero class only when it has no population.

    Args:
        None.

    Returns:
        None.

    Examples:
        >>> test_category_for_smod_skips_unclassified_zero()
    """
    assert _category_for_smod(0, 0.0, {30: "UC"}) is None


@pytest.mark.unit
def test_category_for_smod_rejects_nonzero_unclassified_population() -> None:
    """Reject an unclassified cell that carries population.

    Args:
        None.

    Returns:
        None.

    Examples:
        >>> test_category_for_smod_rejects_nonzero_unclassified_population()
    """
    with pytest.raises(
        ValueError, match="Unclassified Aruba SMOD cells have non-zero population"
    ):
        _category_for_smod(0, 1.0, {30: "UC"})
