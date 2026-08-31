"""Real-data ExactExtract-wrapper method comparison against a UNOSAT result."""

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
from population_exposure.populations._archives import extract_members

FIXTURE_DIRECTORY = Path(__file__).parent / "data" / "unosat_fl20221125cod_basankusu"


@pytest.mark.parametrize(
    "member",
    [
        "layer/../../outside.shp",
        r"layer\..\..\outside.shp",
        "layer/C:/outside.shp",
        r"layer\C:\outside.shp",
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


def test_unosat_regeneration_extracts_only_complete_member_prefixes(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("layer.shp", "")
        source.writestr("layer_backup.shp", "")

    destination = tmp_path / "extracted"
    extract_members(archive, destination, ("layer",))

    assert (destination / "layer.shp").is_file()
    assert not (destination / "layer_backup.shp").exists()


def test_extract_members_requires_a_prefix_boundary(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("cod_admin20.shp", "")
        source.writestr("cod_admin2_backup.shp", "")

    with pytest.raises(ValueError, match="does not contain requested members"):
        extract_members(archive, tmp_path / "extracted", ("cod_admin2",))


@pytest.mark.component
def test_unosat_basankusu_exactextract_golden() -> None:
    """Pin the comparator result; do not claim to reproduce UNOSAT's method."""
    metadata = json.loads((FIXTURE_DIRECTORY / "metadata.json").read_text())
    hazard_path = FIXTURE_DIRECTORY / "hazard.geojson"
    population_path = FIXTURE_DIRECTORY / "population.tif"
    assert metadata["evidence"]["category"] == "real-data method comparison"
    assert "plausible comparator" in metadata["evidence"]["does_not_prove"]
    assert (
        "does not reproduce or validate UNOSAT"
        in metadata["evidence"]["does_not_prove"]
    )

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
    """Keep the transparent internal reference distinct from an ArcGIS result."""
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
