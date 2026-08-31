"""Tests that missing and partial population support is reported honestly.

Five situations must stay apart from one another: a real count of zero, no
population data at all, some population data, a hazard that reaches past the
population raster, and a hazard entirely outside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import population_exposure as pe

if TYPE_CHECKING:
    from pathlib import Path

NODATA = -9999.0
DATA_FRACTION = "population_data_fraction"
DATA_COMPLETE = "population_data_complete"
COVERAGE_FRACTION = "population_coverage_fraction"
COVERAGE_COMPLETE = "population_coverage_complete"

MIXED = np.array([[NODATA, 2.0], [3.0, 4.0]])
TOP_ROW_MISSING = np.array([[NODATA, NODATA], [3.0, 4.0]])
TOP_ROW_ZERO = np.array([[0.0, 0.0], [3.0, 4.0]])


def write_population(
    path: Path,
    values: np.ndarray,
    *,
    crs: str = "EPSG:3857",
    transform=None,
) -> Path:
    """Write a tiny population-count raster.

    Args:
        path: Where to write the raster.
        values: The cell values, with no-data written as ``-9999``.
        crs: The coordinate system to record.
        transform: An explicit affine transform, or None for one-unit cells
            anchored at the origin.

    Returns:
        pathlib.Path: The raster path.

    Examples:
        >>> write_population(tmp_path / "population.tif", MIXED)  # doctest: +SKIP
    """
    data = np.asarray(values, dtype=float)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float64",
        crs=crs,
        transform=transform or from_origin(0, data.shape[0], 1, 1),
        nodata=NODATA,
    ) as dataset:
        dataset.write(data, 1)
    return path


def write_hazard(
    path: Path,
    shape: tuple[int, int],
    *,
    transform=None,
    crs: str = "EPSG:3857",
) -> Path:
    """Write a tiny one-band hazard raster of ones.

    Args:
        path: Where to write the raster.
        shape: The grid shape, as rows and columns.
        transform: An explicit affine transform, or None for one-unit cells
            whose top edge sits at the row count.
        crs: The coordinate system to record.

    Returns:
        pathlib.Path: The raster path.

    Examples:
        >>> write_hazard(tmp_path / "hazard.tif", (2, 2))  # doctest: +SKIP
    """
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="int16",
        crs=crs,
        transform=transform or from_origin(0, shape[0], 1, 1),
        nodata=-32768,
    ) as dataset:
        dataset.write(np.ones(shape, dtype=np.int16), 1)
    return path


def polygons(*geometries, crs: str = "EPSG:3857") -> gpd.GeoDataFrame:
    """Return a labeled hazard frame built from the given boxes.

    Args:
        *geometries: One or more Shapely polygons.
        crs: The coordinate system to record.

    Returns:
        geopandas.GeoDataFrame: A frame indexed by ``zone-0``, ``zone-1``, and
        so on.

    Examples:
        >>> polygons(box(0, 0, 1, 1)).index.tolist()
        ['zone-0']
    """
    labels = [f"zone-{position}" for position in range(len(geometries))]
    return gpd.GeoDataFrame(
        {"label": labels},
        geometry=list(geometries),
        index=pd.Index(labels, name="zone"),
        crs=crs,
    )


class TestVectorSupport:
    """The five situations, for polygon hazards."""

    def test_a_real_zero_count_stays_zero(self, tmp_path: Path) -> None:
        population = write_population(tmp_path / "population.tif", TOP_ROW_ZERO)

        result = pe.assign_population(polygons(box(0, 1, 2, 2)), population)

        assert result["population"].item() == pytest.approx(0.0)
        assert result[DATA_FRACTION].item() == pytest.approx(1.0)
        assert bool(result[DATA_COMPLETE].item()) is True

    def test_no_population_data_is_refused_by_default(self, tmp_path: Path) -> None:
        population = write_population(tmp_path / "population.tif", TOP_ROW_MISSING)

        with pytest.raises(pe.MissingPopulationDataError) as caught:
            pe.assign_population(polygons(box(0, 1, 2, 2)), population)

        message = str(caught.value)
        assert "no population values anywhere the raster covers it" in message
        assert "'zone-0' has population data for 0.0%" in message
        assert "not evidence that nobody lives there" in message
        assert "allow_missing_population_data=True" in message

    def test_no_population_data_returns_missing_when_allowed(
        self,
        tmp_path: Path,
    ) -> None:
        population = write_population(tmp_path / "population.tif", TOP_ROW_MISSING)

        result = pe.assign_population(
            polygons(box(0, 1, 2, 2)),
            population,
            allow_missing_population_data=True,
        )

        assert np.isnan(result["population"].item())
        assert result[DATA_FRACTION].item() == pytest.approx(0.0)
        assert bool(result[DATA_COMPLETE].item()) is False

    def test_some_population_data_reports_its_share(self, tmp_path: Path) -> None:
        population = write_population(tmp_path / "population.tif", MIXED)

        result = pe.assign_population(polygons(box(0, 0, 2, 2)), population)

        assert result["population"].item() == pytest.approx(9.0)
        assert result[DATA_FRACTION].item() == pytest.approx(0.75)
        assert bool(result[DATA_COMPLETE].item()) is False

    def test_reaching_past_the_raster_is_refused_by_default(
        self,
        tmp_path: Path,
    ) -> None:
        population = write_population(tmp_path / "population.tif", MIXED)

        with pytest.raises(pe.PartialCoverageError, match="reaches outside"):
            pe.assign_population(polygons(box(1, 0, 3, 2)), population)

    def test_reaching_past_the_raster_reports_both_shares_when_allowed(
        self,
        tmp_path: Path,
    ) -> None:
        population = write_population(tmp_path / "population.tif", MIXED)

        result = pe.assign_population(
            polygons(box(1, 0, 3, 2)),
            population,
            allow_partial_coverage=True,
        )

        assert result["population"].item() == pytest.approx(6.0)
        assert result[COVERAGE_FRACTION].item() == pytest.approx(0.5)
        assert bool(result[COVERAGE_COMPLETE].item()) is False
        # Half the feature is off the raster; the covered half has data.
        assert result[DATA_FRACTION].item() == pytest.approx(0.5)
        assert bool(result[DATA_COMPLETE].item()) is False

    @pytest.mark.parametrize(
        "options",
        [
            {},
            {"allow_partial_coverage": True},
            {"allow_missing_population_data": True},
            {"allow_partial_coverage": True, "allow_missing_population_data": True},
        ],
        ids=["strict", "partial", "missing", "both"],
    )
    def test_a_feature_entirely_outside_is_always_an_error(
        self,
        tmp_path: Path,
        options: dict[str, bool],
    ) -> None:
        population = write_population(tmp_path / "population.tif", MIXED)

        with pytest.raises(pe.PartialCoverageError, match="entirely outside"):
            pe.assign_population(polygons(box(9, 9, 10, 10)), population, **options)

    def test_only_the_unsupported_feature_is_named(self, tmp_path: Path) -> None:
        population = write_population(tmp_path / "population.tif", MIXED)

        with pytest.raises(pe.MissingPopulationDataError) as caught:
            pe.assign_population(
                polygons(box(0, 1, 1, 2), box(1, 0, 2, 1)),
                population,
            )

        message = str(caught.value)
        assert "1 hazard feature" in message
        assert "'zone-0'" in message
        assert "'zone-1'" not in message

    def test_the_message_stays_accurate_for_a_partly_covered_feature(
        self,
        tmp_path: Path,
    ) -> None:
        """Half the feature is off the raster, so it does not sit only on no-data."""
        population = write_population(tmp_path / "population.tif", TOP_ROW_MISSING)

        with pytest.raises(pe.MissingPopulationDataError) as caught:
            pe.assign_population(
                polygons(box(1, 1, 3, 2)),
                population,
                allow_partial_coverage=True,
            )

        message = str(caught.value)
        assert "no population values anywhere the raster covers it" in message
        assert "Every cell the raster supplies there is no-data" in message

    def test_a_missing_cell_is_never_rounded_away_on_a_large_feature(
        self,
        tmp_path: Path,
    ) -> None:
        """Completeness is judged against one cell, not against the feature."""
        values = np.ones((64, 64))
        values[0, 0] = NODATA
        population = write_population(tmp_path / "population.tif", values)

        result = pe.assign_population(polygons(box(0, 0, 64, 64)), population)

        assert bool(result[DATA_COMPLETE].item()) is False
        assert result[DATA_FRACTION].item() < 1.0
        assert result[DATA_FRACTION].item() == pytest.approx(1 - 1 / 4096)

    def test_a_sliver_over_no_data_is_never_called_complete(
        self,
        tmp_path: Path,
    ) -> None:
        """A feature smaller than the rounding allowance still has no support."""
        population = write_population(tmp_path / "population.tif", TOP_ROW_MISSING)
        sliver = box(0.5, 1.5, 0.5 + 1e-7, 1.5 + 1e-7)

        with pytest.raises(pe.MissingPopulationDataError):
            pe.assign_population(polygons(sliver), population)

        result = pe.assign_population(
            polygons(sliver),
            population,
            allow_missing_population_data=True,
        )

        assert np.isnan(result["population"].item())
        assert result[DATA_FRACTION].item() == pytest.approx(0.0)
        assert bool(result[DATA_COMPLETE].item()) is False

    def test_data_columns_are_added_without_any_opt_in(self, tmp_path: Path) -> None:
        population = write_population(tmp_path / "population.tif", MIXED)

        result = pe.assign_population(polygons(box(0, 0, 1, 1)), population)

        assert result.columns.tolist() == [
            "label",
            "geometry",
            "population",
            DATA_FRACTION,
            DATA_COMPLETE,
        ]

    def test_the_data_support_columns_never_overwrite_hazard_columns(
        self,
        tmp_path: Path,
    ) -> None:
        population = write_population(tmp_path / "population.tif", MIXED)
        hazard = polygons(box(0, 0, 1, 1))
        hazard[DATA_FRACTION] = 0.5

        with pytest.raises(ValueError, match=DATA_FRACTION):
            pe.assign_population(hazard, population)

    @pytest.mark.parametrize(
        ("name", "options"),
        [
            (DATA_FRACTION, {}),
            (DATA_COMPLETE, {}),
            (COVERAGE_FRACTION, {"allow_partial_coverage": True}),
            (COVERAGE_COMPLETE, {"allow_partial_coverage": True}),
        ],
    )
    def test_the_population_column_cannot_take_a_reported_name(
        self,
        tmp_path: Path,
        name: str,
        options: dict[str, bool],
    ) -> None:
        """Otherwise the reported share would quietly replace the total."""
        population = write_population(tmp_path / "population.tif", MIXED)

        with pytest.raises(ValueError, match="adds that column itself"):
            pe.assign_population(
                polygons(box(0, 0, 1, 1)),
                population,
                population_column=name,
                **options,
            )

    def test_an_incomplete_share_never_reads_as_a_full_one(
        self,
        tmp_path: Path,
    ) -> None:
        """Geodesic measurement is approximate, so it must not reach 1.0 alone."""
        import rasterio

        from population_exposure.vector import _surface_coverage_fractions

        population = write_population(tmp_path / "population.tif", np.ones((2, 2)))
        inside = box(0.25, 0.25, 1.75, 1.75)

        with rasterio.open(population) as reader:
            # The feature really is inside, so the geodesic share measures 1.0.
            # The strict rule is what decides, and it says incomplete here.
            shares = _surface_coverage_fractions(
                [inside],
                reader,
                reader.crs,
                covered=np.array([False]),
            )

        assert shares[0] < 1.0

    def test_a_complete_feature_reports_a_share_of_exactly_one(
        self,
        tmp_path: Path,
    ) -> None:
        """The reported share and the completeness flag must never disagree."""
        population = write_population(
            tmp_path / "population.tif",
            np.ones((2, 2)),
            crs="EPSG:4326",
            transform=from_origin(0, 40, 1, 20),
        )

        result = pe.assign_population(
            polygons(box(0, 0, 2, 40), crs="EPSG:4326"),
            population,
            allow_partial_coverage=True,
        )

        assert result[COVERAGE_FRACTION].item() == 1.0
        assert bool(result[COVERAGE_COMPLETE].item()) is True

    def test_the_coordinate_system_error_still_comes_first(
        self,
        tmp_path: Path,
    ) -> None:
        population = write_population(tmp_path / "population.tif", TOP_ROW_MISSING)
        hazard = polygons(box(0, 1, 2, 2)).to_crs("EPSG:4326")

        with pytest.raises(pe.CrsMismatchError):
            pe.assign_population(hazard, population)


class TestRasterSupport:
    """The five situations, for raster hazards."""

    def test_a_real_zero_count_stays_zero(self, tmp_path: Path) -> None:
        population = write_population(tmp_path / "population.tif", TOP_ROW_ZERO)
        hazard = write_hazard(
            tmp_path / "hazard.tif",
            (1, 2),
            transform=from_origin(0, 2, 1, 1),
        )

        result = pe.assign_population(hazard, population)
        _, aligned = result.read()

        assert result.attrs[DATA_FRACTION] == pytest.approx(1.0)
        assert result.attrs[DATA_COMPLETE] is True
        assert not np.ma.getmaskarray(aligned).any()
        assert float(aligned.sum()) == pytest.approx(0.0)

    def test_no_population_data_is_refused_by_default(self, tmp_path: Path) -> None:
        population = write_population(tmp_path / "population.tif", TOP_ROW_MISSING)
        hazard = write_hazard(
            tmp_path / "hazard.tif",
            (1, 2),
            transform=from_origin(0, 2, 1, 1),
        )

        with pytest.raises(pe.MissingPopulationDataError) as caught:
            pe.assign_population(hazard, population)

        message = str(caught.value)
        assert "no values anywhere it covers the hazard grid" in message
        assert "not evidence that nobody lives there" in message
        assert "allow_missing_population_data=True" in message

    def test_no_population_data_is_reported_when_allowed(self, tmp_path: Path) -> None:
        population = write_population(tmp_path / "population.tif", TOP_ROW_MISSING)
        hazard = write_hazard(
            tmp_path / "hazard.tif",
            (1, 2),
            transform=from_origin(0, 2, 1, 1),
        )

        result = pe.assign_population(
            hazard,
            population,
            allow_missing_population_data=True,
        )
        _, aligned = result.read()

        assert result.attrs[DATA_FRACTION] == pytest.approx(0.0)
        assert result.attrs[DATA_COMPLETE] is False
        assert result.attrs[COVERAGE_COMPLETE] is True
        assert np.ma.getmaskarray(aligned).all()

    def test_the_message_stays_accurate_for_a_partly_covered_grid(
        self,
        tmp_path: Path,
    ) -> None:
        """Half the grid is off the raster, so not every masked cell is no-data."""
        population = write_population(tmp_path / "population.tif", TOP_ROW_MISSING)
        hazard = write_hazard(
            tmp_path / "hazard.tif",
            (1, 4),
            transform=from_origin(0, 2, 1, 1),
        )

        with pytest.raises(pe.MissingPopulationDataError) as caught:
            pe.assign_population(hazard, population, allow_partial_coverage=True)

        message = str(caught.value)
        assert "no values anywhere it covers the hazard grid" in message
        assert "every cell it supplies there is no-data" in message

    def test_a_half_missing_cell_is_not_reported_as_complete(
        self,
        tmp_path: Path,
    ) -> None:
        """Sum resampling unmasks a cell that only partly had values."""
        population = write_population(
            tmp_path / "population.tif",
            np.array([[5.0, NODATA]]),
            transform=from_origin(0, 1, 1, 1),
        )
        hazard = write_hazard(
            tmp_path / "hazard.tif",
            (1, 1),
            transform=from_origin(0, 1, 2, 1),
        )

        result = pe.assign_population(hazard, population)
        _, aligned = result.read()

        # The aligned cell reads as a plain 5.0 with nothing masked, so support
        # has to be measured on the population raster's own cells.
        assert bool(np.ma.getmaskarray(aligned).item()) is False
        assert result.attrs[DATA_FRACTION] == pytest.approx(0.5)
        assert result.attrs[DATA_COMPLETE] is False

    def test_a_coarse_hazard_grid_over_only_nodata_is_refused(
        self,
        tmp_path: Path,
    ) -> None:
        """The raster has values, just not under this hazard."""
        population = write_population(
            tmp_path / "population.tif",
            np.array([[NODATA, NODATA, 5.0]]),
            transform=from_origin(0, 1, 1, 1),
        )
        hazard = write_hazard(
            tmp_path / "hazard.tif",
            (1, 1),
            transform=from_origin(0, 1, 2, 1),
        )

        with pytest.raises(pe.MissingPopulationDataError, match="no values anywhere"):
            pe.assign_population(hazard, population)

    def test_some_population_data_reports_its_share(self, tmp_path: Path) -> None:
        population = write_population(tmp_path / "population.tif", MIXED)
        hazard = write_hazard(tmp_path / "hazard.tif", (2, 2))

        result = pe.assign_population(hazard, population)

        assert result.attrs[DATA_FRACTION] == pytest.approx(0.75)
        assert result.attrs[DATA_COMPLETE] is False
        assert result.attrs[COVERAGE_COMPLETE] is True

    def test_missing_data_is_refused_before_the_alignment_pass(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unsupported hazard should not pay for warping before it fails."""
        from population_exposure import raster as raster_module

        population = write_population(
            tmp_path / "population.tif",
            np.array([[NODATA, NODATA, 5.0]]),
            transform=from_origin(0, 1, 1, 1),
        )
        hazard = write_hazard(
            tmp_path / "hazard.tif",
            (1, 1),
            transform=from_origin(0, 1, 2, 1),
        )

        def fail(*args: object, **kwargs: object) -> float:
            raise AssertionError("population was aligned before the support check")

        monkeypatch.setattr(raster_module, "_aligned_population_total", fail)

        with pytest.raises(pe.MissingPopulationDataError):
            pe.assign_population(hazard, population)

    def test_a_missing_cell_is_never_rounded_away_on_a_large_grid(
        self,
        tmp_path: Path,
    ) -> None:
        """Completeness is judged against one cell, not against the grid."""
        values = np.ones((64, 64))
        values[0, 0] = NODATA
        population = write_population(tmp_path / "population.tif", values)
        hazard = write_hazard(tmp_path / "hazard.tif", (64, 64))

        result = pe.assign_population(hazard, population)

        assert result.attrs[DATA_COMPLETE] is False
        assert result.attrs[DATA_FRACTION] < 1.0
        assert result.attrs[DATA_FRACTION] == pytest.approx(1 - 1 / 4096)

    def test_reaching_past_the_raster_is_refused_by_default(
        self,
        tmp_path: Path,
    ) -> None:
        population = write_population(tmp_path / "population.tif", np.ones((2, 2)))
        hazard = write_hazard(tmp_path / "hazard.tif", (2, 4))

        with pytest.raises(pe.PartialCoverageError) as caught:
            pe.assign_population(hazard, population)

        message = str(caught.value)
        assert "reaches outside the population raster" in message
        assert "50.0% of its grid sits inside" in message
        assert "allow_partial_coverage=True" in message

    def test_reaching_past_the_raster_reports_both_shares_when_allowed(
        self,
        tmp_path: Path,
    ) -> None:
        population = write_population(tmp_path / "population.tif", np.ones((2, 2)))
        hazard = write_hazard(tmp_path / "hazard.tif", (2, 4))

        result = pe.assign_population(
            hazard,
            population,
            allow_partial_coverage=True,
        )

        assert result.attrs[COVERAGE_FRACTION] == pytest.approx(0.5)
        assert result.attrs[COVERAGE_COMPLETE] is False
        assert result.attrs[DATA_FRACTION] == pytest.approx(0.5)
        assert result.attrs[DATA_COMPLETE] is False
        assert result.attrs["population_partial_coverage_allowed"] is True
        assert result.attrs["population_missing_data_allowed"] is False

    @pytest.mark.parametrize(
        "options",
        [
            {},
            {"allow_partial_coverage": True},
            {"allow_missing_population_data": True},
            {"allow_partial_coverage": True, "allow_missing_population_data": True},
        ],
        ids=["strict", "partial", "missing", "both"],
    )
    def test_a_hazard_entirely_outside_is_always_an_error(
        self,
        tmp_path: Path,
        options: dict[str, bool],
    ) -> None:
        population = write_population(tmp_path / "population.tif", MIXED)
        hazard = write_hazard(
            tmp_path / "hazard.tif",
            (2, 2),
            transform=from_origin(20, 22, 1, 1),
        )

        with pytest.raises(pe.PartialCoverageError, match="entirely outside"):
            pe.assign_population(hazard, population, **options)

    def test_partial_coverage_still_reads_lazily_in_blocks(
        self,
        tmp_path: Path,
    ) -> None:
        population = write_population(tmp_path / "population.tif", np.ones((2, 2)))
        hazard = write_hazard(tmp_path / "hazard.tif", (2, 4))

        result = pe.assign_population(
            hazard,
            population,
            allow_partial_coverage=True,
        )
        blocks = list(result.iter_blocks())

        assert blocks
        covered = sum(
            float(aligned.sum()) for _, _, aligned in blocks if aligned.count()
        )
        assert covered == pytest.approx(result.attrs["population_aligned_total"])

    def test_conservation_passing_does_not_mean_complete_support(
        self,
        tmp_path: Path,
    ) -> None:
        """The conservation check is arithmetic; only the shares show coverage."""
        population = write_population(tmp_path / "population.tif", np.ones((2, 2)))
        hazard = write_hazard(tmp_path / "hazard.tif", (2, 4))

        result = pe.assign_population(
            hazard,
            population,
            allow_partial_coverage=True,
        )

        assert result.attrs["population_conservation_relative_difference"] == (
            pytest.approx(0.0)
        )
        assert result.attrs[COVERAGE_COMPLETE] is False
        assert result.attrs[DATA_COMPLETE] is False


class TestOptionDispatch:
    """The public entry point routes and rejects the options consistently."""

    def test_the_missing_data_option_must_be_a_boolean(self, tmp_path: Path) -> None:
        population = write_population(tmp_path / "population.tif", MIXED)

        with pytest.raises(
            TypeError,
            match="allow_missing_population_data must be a boolean",
        ):
            pe.assign_population(
                polygons(box(0, 0, 1, 1)),
                population,
                allow_missing_population_data=1,
            )

    @pytest.mark.parametrize(
        "option",
        ["allow_partial_coverage", "allow_missing_population_data"],
    )
    def test_table_hazards_reject_the_raster_backed_options(self, option: str) -> None:
        hazard = pd.DataFrame({"cell": ["A"]})
        population = pd.DataFrame({"cell": ["A"], "population": [1.0]})

        with pytest.raises(ValueError, match="only to vector and raster hazards"):
            pe.assign_population(
                hazard,
                population,
                cell_columns="cell",
                **{option: True},
            )

    def test_a_vector_path_hazard_accepts_both_opt_ins(self, tmp_path: Path) -> None:
        population = write_population(tmp_path / "population.tif", TOP_ROW_MISSING)
        vector_path = tmp_path / "hazard.geojson"
        polygons(box(1, 1, 3, 2)).to_file(vector_path, driver="GeoJSON")

        result = pe.assign_population(
            vector_path,
            population,
            allow_partial_coverage=True,
            allow_missing_population_data=True,
        )

        assert np.isnan(result["population"].item())
        assert result[DATA_FRACTION].item() == pytest.approx(0.0)
        assert result[COVERAGE_FRACTION].item() == pytest.approx(0.5)

    def test_an_open_reader_hazard_accepts_both_opt_ins(self, tmp_path: Path) -> None:
        population = write_population(tmp_path / "population.tif", TOP_ROW_MISSING)
        hazard = write_hazard(
            tmp_path / "hazard.tif",
            (1, 4),
            transform=from_origin(0, 2, 1, 1),
        )

        with rasterio.open(hazard) as reader:
            result = pe.assign_population(
                reader,
                population,
                allow_partial_coverage=True,
                allow_missing_population_data=True,
            )

        assert result.attrs[COVERAGE_FRACTION] == pytest.approx(0.5)
        assert result.attrs[DATA_FRACTION] == pytest.approx(0.0)

    def test_the_assignment_record_names_both_opt_ins(self, tmp_path: Path) -> None:
        population = write_population(tmp_path / "population.tif", MIXED)

        result = pe.assign_population(
            polygons(box(0, 0, 1, 1)),
            population,
            allow_missing_population_data=True,
        )

        assert result.attrs["population_assignment"] == {
            "method": "exactextract_sum",
            "population_crs": "EPSG:3857",
            "population_band": 1,
            "overlaps_allowed": False,
            "reprojected": False,
            "partial_coverage_allowed": False,
            "missing_population_data_allowed": True,
        }
