"""Offline golden coverage for a verified UNOSAT vector population result."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import pytest
import rasterio
from rasterio.features import geometry_mask

from population_exposure import assign_population
from scripts.regenerate_unosat_vector_golden import extract_members

FIXTURE_DIRECTORY = Path(__file__).parent / "data" / "unosat_fl20221125cod_basankusu"


@pytest.mark.parametrize(
    "member",
    [
        "layer/../../outside.shp",
        r"layer\..\..\outside.shp",
        "layer/C:/outside.shp",
    ],
)
def test_unosat_regeneration_rejects_unsafe_archive_paths(
    tmp_path: Path,
    member: str,
) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr(member, "")

    with pytest.raises(ValueError, match="unsafe path"):
        extract_members(archive, tmp_path / "extracted", ("layer",))


@pytest.mark.component
def test_unosat_basankusu_exactextract_golden() -> None:
    metadata = json.loads((FIXTURE_DIRECTORY / "metadata.json").read_text())
    hazard_path = FIXTURE_DIRECTORY / "hazard.geojson"
    population_path = FIXTURE_DIRECTORY / "population.tif"

    assert (
        hashlib.sha256(hazard_path.read_bytes()).hexdigest()
        == metadata["fixture_files"]["hazard.geojson"]
    )
    assert (
        hashlib.sha256(population_path.read_bytes()).hexdigest()
        == metadata["fixture_files"]["population.tif"]
    )

    hazard = gpd.read_file(hazard_path, engine="pyogrio")
    assert len(hazard) == 1
    assert hazard.crs.to_string() == "EPSG:4326"
    assert hazard.geometry.is_valid.all()
    assert not hazard.geometry.is_empty.any()
    assert hazard["district"].tolist() == ["Basankusu"]
    assert hazard["pcode"].tolist() == ["CD4107"]
    assert hazard.to_crs(6933).area.item() / 1_000_000 == pytest.approx(
        metadata["measured"]["area_sqkm"],
        abs=1e-4,
    )
    assert metadata["measured"]["area_sqkm"] == pytest.approx(
        metadata["published"]["area_sqkm"],
        abs=0.001,
    )
    assert metadata["source_layer"]["geometry_valid_before_repair"] is False
    assert metadata["source_layer"]["repair"] == "shapely.make_valid"

    result = assign_population(hazard, population_path)

    assigned = result["population"].item()
    assert assigned == pytest.approx(
        metadata["measured"]["exactextract_population"],
        abs=1e-4,
    )
    assert assigned - metadata["published"]["population"] == pytest.approx(
        metadata["measured"]["exactextract_difference_from_published"],
        abs=1e-4,
    )
    population_assignment = result.attrs["population_assignment"]
    assert population_assignment["method"] == "exactextract_sum"
    assert population_assignment["population_crs"] == "EPSG:4326"
    assert population_assignment["population_band"] == 1
    assert population_assignment["overlaps_allowed"] is False
    assert result.crs == hazard.crs
    assert 0 < assigned < metadata["measured"]["crop_population"]
    assert assigned - metadata["measured"]["full_source_exactextract_population"] == (
        pytest.approx(
            metadata["measured"]["fixture_difference_from_full_source"],
            abs=1e-4,
        )
    )


@pytest.mark.component
def test_unosat_basankusu_cell_center_reference_is_pinned() -> None:
    metadata = json.loads((FIXTURE_DIRECTORY / "metadata.json").read_text())
    hazard = gpd.read_file(FIXTURE_DIRECTORY / "hazard.geojson", engine="pyogrio")

    with rasterio.open(FIXTURE_DIRECTORY / "population.tif") as population:
        values = population.read(1, masked=True)
        included = geometry_mask(
            hazard.geometry,
            out_shape=values.shape,
            transform=population.transform,
            invert=True,
        )

    cell_center_population = float(values[included].sum(dtype="float64"))
    assert list(values.shape) == metadata["measured"]["crop_raster_shape"]
    assert float(values.sum(dtype="float64")) == pytest.approx(
        metadata["measured"]["crop_population"],
        abs=1e-4,
    )
    assert cell_center_population == pytest.approx(
        metadata["measured"]["cell_center_population"],
        abs=1e-4,
    )
    assert cell_center_population - metadata["published"][
        "population"
    ] == pytest.approx(
        metadata["measured"]["cell_center_difference_from_published"],
        abs=1e-4,
    )
