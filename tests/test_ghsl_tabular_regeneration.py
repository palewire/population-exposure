"""Unit coverage for the GHSL tabular fixture regeneration helpers."""

from __future__ import annotations

import json
import zipfile

import pytest
from defusedxml import ElementTree

from scripts.regenerate_ghsl_tabular_golden import (
    COUNTRY_STATS_ARCHIVE,
    WORKBOOK_COLUMNS,
    NoRedirectHandler,
    _category_for_smod,
    _workbook_header,
    _worksheet_rows,
    build_fixture,
    extract_member,
    workbook_totals,
)


@pytest.mark.unit
def test_workbook_cells_require_coordinate_references() -> None:
    """Report a worksheet cell without a coordinate reference clearly.

    Args:
        None.

    Returns:
        None.

    Examples:
        >>> test_workbook_cells_require_coordinate_references()
    """
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    sheet = ElementTree.fromstring(
        b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b"<sheetData><row><c><v>value</v></c></row></sheetData></worksheet>"
    )

    with pytest.raises(ValueError, match="lacks a coordinate reference"):
        list(_worksheet_rows(sheet, [], namespace))


@pytest.mark.unit
def test_regeneration_rejects_source_redirects() -> None:
    """Prevent pinned HTTPS sources from following a redirect.

    Args:
        None.

    Returns:
        None.

    Examples:
        >>> test_regeneration_rejects_source_redirects()
    """
    with pytest.raises(ValueError, match="Redirects are not permitted"):
        NoRedirectHandler().redirect_request(
            None,
            None,
            302,
            "Found",
            None,
            "http://example.test/source",
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


@pytest.mark.unit
def test_build_fixture_treats_missing_workbook_categories_as_zero(
    tmp_path, monkeypatch
) -> None:
    """Allow global workbook totals to omit categories for some countries.

    Args:
        tmp_path: Temporary test directory supplied by pytest.
        monkeypatch: Pytest helper for temporary attribute replacement.

    Returns:
        None.
    """
    output_directory = tmp_path / "fixture"
    gadm_path = tmp_path / "gadm41_ABW_0.json"
    gadm_path.write_text(
        (
            '{"features":[{"properties":{"GID_0":"ABW","COUNTRY":"Aruba"},'
            '"geometry":{"type":"Polygon","coordinates":[]}}]}'
        ),
        encoding="utf-8",
    )
    sources = {
        COUNTRY_STATS_ARCHIVE: tmp_path / COUNTRY_STATS_ARCHIVE,
        "gadm41_ABW_0.json": gadm_path,
        "GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss_V2_0.zip": (
            tmp_path / "GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss_V2_0.zip"
        ),
        "GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip": (
            tmp_path / "GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip"
        ),
    }
    for path in sources.values():
        if path.suffix == ".zip":
            path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        "scripts.regenerate_ghsl_tabular_golden.workbook_totals",
        lambda _: {
            "ABW": {
                "UC": 56903.19754754787,
                "UCL": 45177.75497597072,
                "RUR": 4504.047416000278,
            },
            "BES": {"UC": 1.5},
        },
    )
    monkeypatch.setattr(
        "scripts.regenerate_ghsl_tabular_golden.extract_member",
        lambda archive, member, destination: destination,
    )
    monkeypatch.setattr(
        "scripts.regenerate_ghsl_tabular_golden.global_totals",
        lambda smod_path, population_path: {"UC": 10.0, "UCL": 20.0, "RUR": 30.0},
    )
    monkeypatch.setattr(
        "scripts.regenerate_ghsl_tabular_golden._aruba_rows",
        lambda smod_path, population_path, geometry: (
            [
                {
                    "longitude": "-70.0",
                    "latitude": "12.5",
                    "smod_class": "30",
                    "degurba_l1": "UC",
                    "population": "1.0",
                }
            ],
            {"UC": 10.0, "UCL": 20.0, "RUR": 30.0},
            {"width": 1, "height": 1},
        ),
    )

    build_fixture(output_directory, sources)

    metadata = json.loads(
        (output_directory / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["workbook"]["global"] == {
        "RUR": 4504.047416000278,
        "UC": 56904.69754754787,
        "UCL": 45177.75497597072,
    }
