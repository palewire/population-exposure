"""Tests for the built-in population dataset catalog."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from filelock import Timeout
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import box

from population_exposure import assign_population, populations
from population_exposure.populations import _api, _cache, _selection, _sources
from population_exposure.populations._http import DownloadResult, sha256_file


def write_population(
    path: Path,
    *,
    values: np.ndarray | None = None,
    year: int = 2020,
    tags: dict[str, str] | None = None,
    width: int = 2,
    height: int = 2,
    include_year: bool = True,
) -> Path:
    """Write a tiny invented count raster."""
    data = (
        np.array([[1.0, 2.0], [3.0, 4.0]])
        if values is None
        else np.asarray(values, dtype="float64")
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float64",
        crs="EPSG:4326",
        transform=from_origin(0, height, 1, 1),
        nodata=-9999,
    ) as dataset:
        dataset.write(data, 1)
        metadata: dict[str, str] = {
            "population_semantics": "count",
            "units": "population count per cell",
        }
        if include_year:
            metadata["year"] = str(year)
        metadata.update(tags or {})
        dataset.update_tags(1, **metadata)
    return path


def use_tiny_source(
    monkeypatch: pytest.MonkeyPatch,
    source_id: str,
    *,
    delivery: str | None = None,
):
    """Replace one source's global grid checks with tiny fixture checks."""
    source = _sources.SOURCES[source_id]
    tiny = replace(
        source,
        delivery=delivery or source.delivery,
        url_template="https://example.test/{year}",
        filename_template=f"{source_id}-{{year}}.tif",
        archive_member_template=f"{source_id}-{{year}}.tif"
        if (delivery or source.delivery) == "zip"
        else None,
        crs="EPSG:4326",
        expected_width=2,
        expected_height=2,
        expected_resolution=(1.0, 1.0),
        expected_bounds=(0.0, 0.0, 2.0, 2.0),
        expected_nodata=(-9999.0,),
        expected_bounds_by_year=MappingProxyType({}),
        expected_nodata_by_year=MappingProxyType({}),
        plausible_total=(0.0, 1_000.0),
        max_download_bytes=1_000_000,
        exact_download_bytes=None,
        publisher_checksum=None,
    )
    catalog = MappingProxyType(
        {
            key: tiny if key == source_id else value
            for key, value in _sources.SOURCES.items()
        }
    )
    monkeypatch.setattr(_sources, "SOURCES", catalog)
    monkeypatch.setattr(_selection, "SOURCES", catalog)
    monkeypatch.setattr(_api, "SOURCES", catalog)
    return tiny


def test_landscan_2024_grid_covers_the_full_global_extent() -> None:
    """Keep LandScan 2024 registration checks aligned with its published grid."""
    source = _sources.LANDSCAN

    assert source.expected_width == 43_200
    assert source.expected_height == 21_600
    assert source.expected_bounds == (
        -180.0,
        -89.99999999280098,
        179.99999998559997,
        89.999999999999,
    )


def fake_downloader_from(
    source_path: Path,
    calls: list[dict[str, object]],
):
    """Return a downloader that installs fixture bytes."""

    def fake_download(
        url,
        partial_path,
        *,
        headers,
        max_bytes,
        exact_bytes,
        publisher_checksum,
    ):
        calls.append(
            {
                "url": url,
                "headers": headers,
                "max_bytes": max_bytes,
                "exact_bytes": exact_bytes,
                "publisher_checksum": publisher_checksum,
            }
        )
        partial_path.write_bytes(source_path.read_bytes())
        return DownloadResult(
            size=partial_path.stat().st_size,
            sha256=sha256_file(partial_path),
        )

    return fake_download


def test_list_and_info_expose_curated_source_facts() -> None:
    sources = populations.list()

    assert [source.source_id for source in sources] == [
        "worldpop-global-1km",
        "ghsl-r2023a-mollweide-1km",
        "gpwv4-r11-count",
        "chambers-hybrid",
        "landscan-global",
    ]
    assert all(source.supported_years for source in sources)
    ghsl = populations.info("ghsl-r2023a-mollweide-1km:2020")
    assert ghsl.release == "R2023A V1.0"
    assert (
        ghsl.title == "GHS-POP R2023A - GHS population grid multitemporal (1975-2030)"
    )
    assert ghsl.crs == "ESRI:54009"
    assert ghsl.meaning == "residential"
    assert ghsl.license.startswith("European Commission reuse notice")
    assert "third-party intellectual property rights" in ghsl.license
    assert ghsl.landing_page == (
        "https://data.jrc.ec.europa.eu/dataset/2ff68a52-5b5b-4a22-8f40-c41da8332cfe"
    )
    assert ghsl.official_url.endswith("_V1_0.zip")
    assert any("do not treat it as independent" in note for note in ghsl.notes)
    assert ghsl.citation.startswith("Schiavina, M., Freire, S., and MacManus, K.")
    assert "Carioli" not in ghsl.citation
    worldpop = populations.info("worldpop-global-1km:2020")
    assert any(
        "not annual local census observations" in note for note in worldpop.notes
    )
    gpw = populations.info("gpwv4-r11-count:2020")
    assert any("not 1 km enumeration" in note for note in gpw.notes)
    chambers = populations.info("chambers-hybrid:2000")
    assert any("2000 seam" in note for note in chambers.notes)
    landscan_2023 = populations.info("landscan-global:2023")
    assert landscan_2023.doi is None
    assert "10.48690/1532445" not in landscan_2023.citation
    landscan_2024 = populations.info("landscan-global:2024")
    assert landscan_2024.meaning == "ambient"
    assert landscan_2024.doi == "10.48690/1532445"
    assert "redistribution" in landscan_2024.license
    assert landscan_2024.citation == (
        "Lebakula, V., Gonzales, J., Stipek, C., Tsybina, E., Zimmer, A., "
        "Nukavarapu, N., Byeonghwa, J., Reynolds, B., Kaufman, J., Fan, J., "
        "Martin, A., Buck, W., Basford, S., Faxon, A., Meade, S., & Urban, M. "
        "(2024). LandScan 2024 [Dataset]. Oak Ridge National Laboratory. "
        "https://doi.org/10.48690/1532445"
    )
    assert any("registrant's responsibility" in note for note in landscan_2024.notes)


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ("worldpop-global-1km", "requires an explicit year"),
        ("worldpop-global-1km:latest", "never use 'latest'"),
        ("worldpop-global-1km:20", "exactly match"),
        ("WORLDPOP-GLOBAL-1KM:2020", "exactly match"),
        ("unknown-source:2020", "Unknown population source"),
        ("worldpop-global-1km:1999", "does not support 1999"),
        ("ghsl-r2023a-mollweide-1km:2025", "does not support 2025"),
        ("gpwv4-r11-count:2019", "supported years"),
    ],
)
def test_selection_must_be_exact(selection: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        populations.info(selection)


def test_selection_must_be_a_string() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        populations.info(2020)  # type: ignore[arg-type]


def test_anonymous_download_writes_receipt_and_uses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "worldpop-global-1km")
    fixture = write_population(tmp_path / "fixture-2020.tif")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(_api, "download_file", fake_downloader_from(fixture, calls))

    result = populations.download(
        "worldpop-global-1km:2020",
        cache_dir=tmp_path / "cache",
    )
    cached = populations.download(
        "worldpop-global-1km:2020",
        cache_dir=tmp_path / "cache",
    )

    assert result == cached
    assert len(calls) == 1
    assert calls[0]["headers"] is None
    receipt = json.loads(result.with_suffix(".tif.json").read_text())
    assert receipt["selection"] == "worldpop-global-1km:2020"
    assert receipt["local_sha256"] == sha256_file(result)
    assert receipt["observed"]["population_total"] == 10.0
    assert receipt["license"].startswith("Creative Commons")
    assert receipt["processing_note"].startswith("Downloaded anonymously")
    assert "_verified_mtime_ns" in receipt


def test_direct_download_reuses_downloader_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "worldpop-global-1km")
    fixture = write_population(tmp_path / "fixture-2020.tif")
    expected_digest = sha256_file(fixture)

    def download_raster(
        url,
        partial_path,
        *,
        headers,
        max_bytes,
        exact_bytes,
        publisher_checksum,
    ):
        del url, headers, max_bytes, exact_bytes, publisher_checksum
        partial_path.write_bytes(fixture.read_bytes())
        return DownloadResult(
            size=partial_path.stat().st_size,
            sha256=expected_digest,
        )

    monkeypatch.setattr(_api, "download_file", download_raster)
    monkeypatch.setattr(
        _api,
        "sha256_file",
        lambda path: pytest.fail(f"Direct raster was rehashed: {path.name}"),
    )

    result = populations.download(
        "worldpop-global-1km:2020",
        cache_dir=tmp_path / "cache",
    )

    receipt = json.loads(result.with_suffix(".tif.json").read_text())
    assert receipt["local_sha256"] == expected_digest


def test_refresh_replaces_a_verified_cache_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "worldpop-global-1km")
    fixture = write_population(tmp_path / "fixture-2020.tif")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(_api, "download_file", fake_downloader_from(fixture, calls))
    cache = tmp_path / "cache"

    first = populations.download("worldpop-global-1km:2020", cache_dir=cache)
    first_receipt = json.loads(first.with_suffix(".tif.json").read_text())
    second = populations.download(
        "worldpop-global-1km:2020",
        cache_dir=cache,
        refresh=True,
    )
    second_receipt = json.loads(second.with_suffix(".tif.json").read_text())

    assert first == second
    assert len(calls) == 2
    assert second_receipt["retrieved_at"] >= first_receipt["retrieved_at"]


def test_failed_refresh_retains_the_last_verified_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "worldpop-global-1km")
    fixture = write_population(tmp_path / "fixture-2020.tif")
    monkeypatch.setattr(
        _api,
        "download_file",
        fake_downloader_from(fixture, []),
    )
    cache = tmp_path / "cache"
    cached = populations.download("worldpop-global-1km:2020", cache_dir=cache)
    before = cached.read_bytes()
    receipt = cached.with_suffix(".tif.json")
    receipt_before = receipt.read_bytes()

    def fail_download(*args, **kwargs):
        raise ValueError("publisher unavailable")

    monkeypatch.setattr(_api, "download_file", fail_download)
    with pytest.raises(ValueError, match="publisher unavailable"):
        populations.download(
            "worldpop-global-1km:2020",
            cache_dir=cache,
            refresh=True,
        )

    assert cached.read_bytes() == before
    assert receipt.read_bytes() == receipt_before


def test_failed_refresh_receipt_write_rolls_back_file_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "worldpop-global-1km")
    first_fixture = write_population(tmp_path / "first-2020.tif")
    monkeypatch.setattr(
        _api,
        "download_file",
        fake_downloader_from(first_fixture, []),
    )
    cache = tmp_path / "cache"
    cached = populations.download("worldpop-global-1km:2020", cache_dir=cache)
    old_bytes = cached.read_bytes()
    receipt = cached.with_suffix(".tif.json")
    old_receipt = receipt.read_bytes()

    second_fixture = write_population(
        tmp_path / "second-2020.tif",
        values=np.array([[10.0, 20.0], [30.0, 40.0]]),
    )
    monkeypatch.setattr(
        _api,
        "download_file",
        fake_downloader_from(second_fixture, []),
    )

    def fail_receipt(*args, **kwargs):
        target_receipt = args[1]
        target_receipt.with_suffix(f"{target_receipt.suffix}.partial").write_text(
            "partial"
        )
        raise OSError("disk full")

    monkeypatch.setattr(_api, "write_receipt", fail_receipt)
    with pytest.raises(OSError, match="disk full"):
        populations.download(
            "worldpop-global-1km:2020",
            cache_dir=cache,
            refresh=True,
        )

    assert cached.read_bytes() == old_bytes
    assert receipt.read_bytes() == old_receipt
    assert not tuple(cache.rglob("*.backup"))
    assert not tuple(cache.rglob("*.partial"))

    empty_cache = tmp_path / "empty-cache"
    with pytest.raises(OSError, match="disk full"):
        populations.download(
            "worldpop-global-1km:2020",
            cache_dir=empty_cache,
        )
    assert not tuple(empty_cache.rglob("*.tif"))
    assert not tuple(empty_cache.rglob("*.json"))


def test_offline_mode_never_downloads_and_environment_is_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "worldpop-global-1km")
    called = False

    def fail_download(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("network called")

    monkeypatch.setattr(_api, "download_file", fail_download)
    monkeypatch.setenv("POPULATION_EXPOSURE_OFFLINE", "true")
    with pytest.raises(ValueError, match="made no network request"):
        populations.download(
            "worldpop-global-1km:2020",
            cache_dir=tmp_path / "cache",
        )
    assert not called

    monkeypatch.setenv("POPULATION_EXPOSURE_OFFLINE", "sometimes")
    with pytest.raises(ValueError, match="POPULATION_EXPOSURE_OFFLINE must be"):
        populations.download(
            "worldpop-global-1km:2020",
            cache_dir=tmp_path / "cache",
        )
    with pytest.raises(TypeError, match="offline must be"):
        populations.download(
            "worldpop-global-1km:2020",
            cache_dir=tmp_path / "cache",
            offline=1,  # type: ignore[arg-type]
        )


def test_offline_valid_cache_skips_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "worldpop-global-1km")
    fixture = write_population(tmp_path / "fixture-2020.tif")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(_api, "download_file", fake_downloader_from(fixture, calls))
    cache = tmp_path / "cache"
    expected = populations.download("worldpop-global-1km:2020", cache_dir=cache)

    monkeypatch.setattr(
        _api,
        "download_file",
        lambda *args, **kwargs: pytest.fail("network called"),
    )
    result = populations.download(
        "worldpop-global-1km:2020",
        cache_dir=cache,
        offline=True,
    )

    assert result == expected
    assert len(calls) == 1


def test_gpw_forwards_explicit_earthdata_token_without_exposing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = use_tiny_source(monkeypatch, "gpwv4-r11-count", delivery="zip")
    fixture = write_population(tmp_path / "fixture-2020.tif")
    archive = tmp_path / "gpw.zip"
    member = source.archive_member(2020)
    assert member is not None
    with ZipFile(archive, "w") as output:
        output.write(fixture, arcname=f"folder/{member}")
    token = "caller-earthdata-token"
    expected_authorization = "Bearer " + token
    received_authorization: list[bool] = []

    def download_archive(
        url,
        partial_path,
        *,
        headers,
        max_bytes,
        exact_bytes,
        publisher_checksum,
    ):
        del url, max_bytes, exact_bytes, publisher_checksum
        received_authorization.append(
            headers == {"Authorization": expected_authorization}
        )
        partial_path.write_bytes(archive.read_bytes())
        return DownloadResult(
            size=partial_path.stat().st_size,
            sha256=sha256_file(partial_path),
        )

    monkeypatch.setattr(_api, "download_file", download_archive)

    result = populations.download(
        "gpwv4-r11-count:2020",
        cache_dir=tmp_path / "cache",
        earthdata_token=token,
    )

    receipt_text = result.with_suffix(".tif.json").read_text()
    assert received_authorization == [True]
    assert "transient user-owned token" in receipt_text

    def reject_download(
        url,
        partial_path,
        *,
        headers,
        max_bytes,
        exact_bytes,
        publisher_checksum,
    ):
        del url, partial_path, max_bytes, exact_bytes, publisher_checksum
        received_authorization.append(
            headers == {"Authorization": expected_authorization}
        )
        raise ValueError("publisher rejected the request")

    monkeypatch.setattr(_api, "download_file", reject_download)
    with pytest.raises(ValueError, match="publisher rejected") as error:
        populations.download(
            "gpwv4-r11-count:2020",
            cache_dir=tmp_path / "cache",
            earthdata_token=token,
            refresh=True,
        )

    assert received_authorization == [True, True]
    token_is_not_disclosed = all(
        token not in text for text in (receipt_text, str(error.value), caplog.text)
    )
    assert token_is_not_disclosed


def test_gpw_requires_a_user_token_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "gpwv4-r11-count", delivery="zip")
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    monkeypatch.setattr(
        _api,
        "download_file",
        lambda *args, **kwargs: pytest.fail("network called"),
    )

    with pytest.raises(ValueError, match="requires your Earthdata token"):
        populations.download(
            "gpwv4-r11-count:2020",
            cache_dir=tmp_path / "cache",
        )


def test_landscan_guides_manual_acquisition_and_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "landscan-global")
    with pytest.raises(ValueError, match="official ORNL LandScan portal"):
        populations.download(
            "landscan-global:2024",
            cache_dir=tmp_path / "cache",
        )

    original = write_population(tmp_path / "landscan-global-2024.tif", year=2024)
    original_bytes = original.read_bytes()
    registered = populations.register(
        "landscan-global:2024",
        original,
        cache_dir=tmp_path / "cache",
    )

    assert registered != original
    assert registered.read_bytes() == original_bytes
    assert original.read_bytes() == original_bytes
    receipt = json.loads(registered.with_suffix(".tif.json").read_text())
    assert receipt["population_meaning"] == "ambient"
    assert receipt["citation"] == populations.info("landscan-global:2024").citation
    assert receipt["license"] == populations.info("landscan-global:2024").license
    assert (
        "source identity remains the registrant's responsibility"
        in receipt["processing_note"]
    )
    assert "original file was not modified" in receipt["processing_note"]

    with pytest.raises(
        ValueError,
        match="structurally checked cached file was retained",
    ):
        populations.download(
            "landscan-global:2024",
            cache_dir=tmp_path / "cache",
            refresh=True,
        )
    assert registered.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("name", "tags", "values", "message"),
    [
        (
            "landscan-global-2023.tif",
            {"year": "2023"},
            np.ones((2, 2)),
            "not requested year",
        ),
        (
            "landscan-global-2024.tif",
            {"population_semantics": "density"},
            np.ones((2, 2)),
            "population_semantics=count",
        ),
        (
            "landscan-global-2024.tif",
            {},
            np.array([[1.0, -1.0], [2.0, 3.0]]),
            "non-negative",
        ),
    ],
)
def test_registration_rejects_wrong_year_density_and_bad_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    tags: dict[str, str],
    values: np.ndarray,
    message: str,
) -> None:
    use_tiny_source(monkeypatch, "landscan-global")
    raster = write_population(
        tmp_path / name,
        year=2024,
        values=values,
        tags=tags,
    )

    with pytest.raises(ValueError, match=message):
        populations.register(
            "landscan-global:2024",
            raster,
            cache_dir=tmp_path / "cache",
        )


def test_registration_requires_existing_year_marked_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "landscan-global")
    with pytest.raises(ValueError, match="does not exist"):
        populations.register(
            "landscan-global:2024",
            tmp_path / "missing.tif",
            cache_dir=tmp_path / "cache",
        )
    raster = write_population(
        tmp_path / "population.tif",
        year=2024,
        include_year=False,
    )
    with pytest.raises(ValueError, match="must identify the requested year"):
        populations.register(
            "landscan-global:2024",
            raster,
            cache_dir=tmp_path / "cache",
        )


def test_chambers_reuses_one_source_for_multiple_years_and_offline_derivation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "chambers-hybrid")
    raw = tmp_path / "source.nc"
    raw.write_bytes(b"invented netcdf fixture")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(_api, "download_file", fake_downloader_from(raw, calls))
    derived_years: list[int] = []

    def derive(source_path: Path, output_path: Path, year: int) -> None:
        assert source_path.read_bytes() == raw.read_bytes()
        derived_years.append(year)
        write_population(output_path, year=year)

    monkeypatch.setattr(_api, "derive_chambers_year", derive)
    cache = tmp_path / "cache"

    first = populations.download("chambers-hybrid:2019", cache_dir=cache)
    second = populations.download(
        "chambers-hybrid:2020",
        cache_dir=cache,
        offline=True,
    )

    assert first != second
    assert len(calls) == 1
    assert derived_years == [2019, 2020]
    receipt = json.loads(second.with_suffix(".tif.json").read_text())
    assert receipt["source_file_sha256"] == sha256_file(raw)
    assert "14 age bands" in receipt["processing_note"]


def test_catalog_selection_and_custom_path_assignment_attrs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "worldpop-global-1km")
    monkeypatch.setenv("POPULATION_EXPOSURE_CACHE_DIR", str(tmp_path / "cache"))
    population = write_population(tmp_path / "worldpop-global-1km-2020.tif")
    populations.register("worldpop-global-1km:2020", population)
    hazard = write_population(
        tmp_path / "hazard.tif",
        values=np.array([[10, 20], [30, 40]]),
    )

    catalog_result = assign_population(hazard, "worldpop-global-1km:2020")
    custom_result = assign_population(hazard, population)

    assert (
        catalog_result.attrs["population_source"]["selection"]
        == "worldpop-global-1km:2020"
    )
    assert catalog_result.attrs["population_source"]["license"].startswith(
        "Creative Commons"
    )
    assert custom_result.attrs["population_source"]["source_id"] == "custom"
    assert len(custom_result.attrs["population_source"]["local_sha256"]) == 64


def test_catalog_selection_attrs_are_attached_to_vector_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "worldpop-global-1km")
    monkeypatch.setenv("POPULATION_EXPOSURE_CACHE_DIR", str(tmp_path / "cache"))
    population = write_population(tmp_path / "worldpop-global-1km-2020.tif")
    populations.register("worldpop-global-1km:2020", population)
    hazard = gpd.GeoDataFrame(
        geometry=[box(0, 0, 2, 2)],
        crs="EPSG:4326",
    )

    result = assign_population(hazard, "worldpop-global-1km:2020")

    assert result["population"].item() == 10.0
    assert result.attrs["population_source"]["year"] == 2020
    assert result.attrs["population_source"]["local_sha256"]


def test_existing_paths_win_and_missing_pathlike_values_do_not_become_selections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unusual = write_population(tmp_path / "worldpop-global-1km:2020")
    hazard = write_population(
        tmp_path / "hazard.tif",
        values=np.array([[10, 20], [30, 40]]),
    )

    with pytest.raises(ValueError, match="must be a GeoTIFF"):
        assign_population(hazard, unusual)

    with pytest.raises(ValueError, match="path does not exist"):
        assign_population(hazard, Path("worldpop-global-1km:2020"))
    with pytest.raises(ValueError, match="path does not exist"):
        assign_population(hazard, "missing-population.tif")
    with pytest.raises(ValueError, match=r"C:\\data\\population.tif"):
        assign_population(hazard, r"C:\data\population.tif")
    with pytest.raises(ValueError, match="https:/"):
        assign_population(hazard, "https://example.test/population.tif")

    home = tmp_path / "home"
    home.mkdir()
    home_population = write_population(home / "population.tif")
    monkeypatch.setenv("HOME", str(home))
    expanded = _api.resolve_for_assignment("~/population.tif")
    assert expanded.source == home_population
    expanded_pathlike = _api.resolve_for_assignment(Path("~/population.tif"))
    assert expanded_pathlike.source == home_population


def test_concurrent_downloads_install_one_cache_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "worldpop-global-1km")
    fixture = write_population(tmp_path / "fixture-2020.tif")
    calls: list[dict[str, object]] = []
    base_downloader = fake_downloader_from(fixture, calls)

    def slow_download(*args, **kwargs):
        time.sleep(0.05)
        return base_downloader(*args, **kwargs)

    monkeypatch.setattr(_api, "download_file", slow_download)
    cache = tmp_path / "cache"
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = tuple(
            pool.map(
                lambda _: populations.download(
                    "worldpop-global-1km:2020",
                    cache_dir=cache,
                ),
                range(2),
            )
        )

    assert paths[0] == paths[1]
    assert len(calls) == 1


def test_invalid_receipt_is_not_treated_as_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "worldpop-global-1km")
    fixture = write_population(tmp_path / "fixture-2020.tif")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(_api, "download_file", fake_downloader_from(fixture, calls))
    cache = tmp_path / "cache"
    result = populations.download("worldpop-global-1km:2020", cache_dir=cache)
    receipt = result.with_suffix(".tif.json")
    receipt.write_text("{}")

    populations.download("worldpop-global-1km:2020", cache_dir=cache)

    assert len(calls) == 2


def test_cache_directory_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "landscan-global")
    monkeypatch.setenv("POPULATION_EXPOSURE_CACHE_DIR", str(tmp_path / "configured"))
    original = write_population(tmp_path / "landscan-global-2024.tif", year=2024)

    result = populations.register("landscan-global:2024", original)

    assert tmp_path / "configured" in result.parents


@pytest.mark.parametrize("cache_root_kind", ["platform-default", "environment"])
def test_shared_cache_is_reused_across_working_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_root_kind: str,
) -> None:
    use_tiny_source(monkeypatch, "worldpop-global-1km")
    fixture = write_population(tmp_path / "fixture-2020.tif")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(_api, "download_file", fake_downloader_from(fixture, calls))
    shared_root = tmp_path / "shared-cache"
    first_project = tmp_path / "first-project"
    second_project = tmp_path / "second-project"
    first_project.mkdir()
    second_project.mkdir()

    if cache_root_kind == "platform-default":
        monkeypatch.delenv("POPULATION_EXPOSURE_CACHE_DIR", raising=False)
        monkeypatch.setattr(
            _cache, "user_cache_path", lambda *args, **kwargs: shared_root
        )
    else:
        monkeypatch.setenv("POPULATION_EXPOSURE_CACHE_DIR", str(shared_root))

    monkeypatch.chdir(first_project)
    first = populations.download("worldpop-global-1km:2020")
    monkeypatch.chdir(second_project)
    second = populations.download("worldpop-global-1km:2020")

    assert first == second
    assert first.is_relative_to(shared_root)
    assert len(calls) == 1


def test_refresh_and_register_option_types_are_checked(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="refresh must be"):
        populations.download(
            "worldpop-global-1km:2020",
            cache_dir=tmp_path,
            refresh=1,  # type: ignore[arg-type]
        )


def test_ghsl_archive_download_extracts_only_the_expected_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = use_tiny_source(
        monkeypatch,
        "ghsl-r2023a-mollweide-1km",
        delivery="zip",
    )
    fixture = write_population(tmp_path / "fixture-2020.tif")
    archive = tmp_path / "ghsl.zip"
    member = source.archive_member(2020)
    assert member is not None
    with ZipFile(archive, "w") as output:
        output.writestr("unrelated.txt", "ignored")
        output.write(fixture, arcname=member)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(_api, "download_file", fake_downloader_from(archive, calls))

    result = populations.download(
        "ghsl-r2023a-mollweide-1km:2020",
        cache_dir=tmp_path / "cache",
    )

    assert result.is_file()
    receipt = result.with_suffix(".tif.json").read_text()
    assert "extracted the exact population-count GeoTIFF" in receipt


def test_archive_download_hashes_the_extracted_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = use_tiny_source(
        monkeypatch,
        "ghsl-r2023a-mollweide-1km",
        delivery="zip",
    )
    fixture = write_population(tmp_path / "fixture-2020.tif")
    archive = tmp_path / "ghsl.zip"
    member = source.archive_member(2020)
    assert member is not None
    with ZipFile(archive, "w") as output:
        output.write(fixture, arcname=member)
    archive_digest = sha256_file(archive)
    member_digest = sha256_file(fixture)
    hashed_paths: list[Path] = []
    original_sha256_file = _api.sha256_file

    def hash_extracted_member(path: Path) -> str:
        hashed_paths.append(path)
        return original_sha256_file(path)

    def download_archive(
        url,
        partial_path,
        *,
        headers,
        max_bytes,
        exact_bytes,
        publisher_checksum,
    ):
        del url, headers, max_bytes, exact_bytes, publisher_checksum
        partial_path.write_bytes(archive.read_bytes())
        return DownloadResult(
            size=partial_path.stat().st_size,
            sha256=archive_digest,
        )

    monkeypatch.setattr(_api, "sha256_file", hash_extracted_member)
    monkeypatch.setattr(_api, "download_file", download_archive)

    result = populations.download(
        "ghsl-r2023a-mollweide-1km:2020",
        cache_dir=tmp_path / "cache",
    )

    receipt = json.loads(result.with_suffix(".tif.json").read_text())
    assert hashed_paths == [result.with_name(f"{result.name}.partial")]
    assert receipt["local_sha256"] == member_digest


def test_bad_archive_member_and_invalid_direct_raster_are_cleaned_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(
        monkeypatch,
        "ghsl-r2023a-mollweide-1km",
        delivery="zip",
    )
    bad_archive = tmp_path / "bad.zip"
    with ZipFile(bad_archive, "w") as output:
        output.writestr("density.tif", b"not the requested count raster")
    monkeypatch.setattr(
        _api,
        "download_file",
        fake_downloader_from(bad_archive, []),
    )
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="must contain exactly one"):
        populations.download(
            "ghsl-r2023a-mollweide-1km:2020",
            cache_dir=cache,
        )
    assert not tuple(cache.rglob("*.partial"))

    use_tiny_source(monkeypatch, "worldpop-global-1km")
    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_text("not a GeoTIFF")
    monkeypatch.setattr(_api, "download_file", fake_downloader_from(corrupt, []))
    with pytest.raises(ValueError, match="not a readable GeoTIFF"):
        populations.download(
            "worldpop-global-1km:2020",
            cache_dir=cache,
        )
    assert not tuple(cache.rglob("*.partial"))


def test_earthdata_environment_token_is_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = use_tiny_source(monkeypatch, "gpwv4-r11-count", delivery="zip")
    fixture = write_population(tmp_path / "fixture-2020.tif")
    archive = tmp_path / "gpw.zip"
    member = source.archive_member(2020)
    assert member is not None
    with ZipFile(archive, "w") as output:
        output.write(fixture, arcname=member)
    token = "environment-earthdata-token"
    expected_authorization = "Bearer " + token
    received_authorization: list[bool] = []

    def download_archive(
        url,
        partial_path,
        *,
        headers,
        max_bytes,
        exact_bytes,
        publisher_checksum,
    ):
        del url, max_bytes, exact_bytes, publisher_checksum
        received_authorization.append(
            headers == {"Authorization": expected_authorization}
        )
        partial_path.write_bytes(archive.read_bytes())
        return DownloadResult(
            size=partial_path.stat().st_size,
            sha256=sha256_file(partial_path),
        )

    monkeypatch.setattr(_api, "download_file", download_archive)
    monkeypatch.setenv("EARTHDATA_TOKEN", token)

    populations.download(
        "gpwv4-r11-count:2020",
        cache_dir=tmp_path / "cache",
    )
    assert received_authorization == [True]


@pytest.mark.parametrize(
    ("earthdata_token", "environment_token"),
    [
        (" ", "environment-earthdata-token"),
        (None, " \t"),
    ],
)
def test_earthdata_whitespace_tokens_fail_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    earthdata_token: str | None,
    environment_token: str,
) -> None:
    use_tiny_source(monkeypatch, "gpwv4-r11-count", delivery="zip")
    monkeypatch.setenv("EARTHDATA_TOKEN", environment_token)
    monkeypatch.setattr(
        _api,
        "download_file",
        lambda *args, **kwargs: pytest.fail("network called"),
    )

    with pytest.raises(ValueError, match="requires your Earthdata token"):
        populations.download(
            "gpwv4-r11-count:2020",
            cache_dir=tmp_path / "cache",
            earthdata_token=earthdata_token,
        )


def test_registering_the_cache_file_again_is_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "landscan-global")
    original = write_population(tmp_path / "landscan-global-2024.tif", year=2024)
    cached = populations.register(
        "landscan-global:2024",
        original,
        cache_dir=tmp_path / "cache",
    )
    before = cached.read_bytes()

    result = populations.register(
        "landscan-global:2024",
        cached,
        cache_dir=tmp_path / "cache",
    )

    assert result == cached
    assert result.read_bytes() == before


def test_registration_size_limit_is_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "landscan-global")
    original = write_population(tmp_path / "landscan-global-2024.tif", year=2024)
    monkeypatch.setattr(_api, "_MAX_REGISTERED_BYTES", 1)

    with pytest.raises(ValueError, match="registration safety limit"):
        populations.register(
            "landscan-global:2024",
            original,
            cache_dir=tmp_path / "cache",
        )


def test_string_path_and_memory_reader_remain_custom_sources(
    tmp_path: Path,
) -> None:
    population = write_population(tmp_path / "population.tif")

    resolved = _api.resolve_for_assignment(str(population))
    with rasterio.open(population) as reader:
        metadata = _api.metadata_for_reader(resolved, reader, total=10.0)
    assert metadata["source_id"] == "custom"
    assert metadata["local_sha256"]

    profile = {
        "driver": "GTiff",
        "width": 2,
        "height": 2,
        "count": 1,
        "dtype": "float64",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 2, 1, 1),
        "nodata": -9999,
    }
    with MemoryFile() as memory:
        with memory.open(**profile) as writer:
            writer.write(np.ones((2, 2)), 1)
        with memory.open() as reader:
            reader_resolved = _api.resolve_for_assignment(reader)
            reader_metadata = _api.metadata_for_reader(
                reader_resolved,
                reader,
                total=4.0,
            )
            assert "local_sha256" not in reader_metadata


def test_download_and_register_lock_timeouts_are_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "worldpop-global-1km")

    class FailingLock:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            raise Timeout("test.lock")

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(_api, "FileLock", FailingLock)
    with pytest.raises(ValueError, match="Timed out waiting"):
        populations.download(
            "worldpop-global-1km:2020",
            cache_dir=tmp_path / "cache",
        )

    original = write_population(tmp_path / "worldpop-global-1km-2020.tif")
    with pytest.raises(ValueError, match="Timed out waiting"):
        populations.register(
            "worldpop-global-1km:2020",
            original,
            cache_dir=tmp_path / "cache",
        )


def test_chambers_refreshes_source_and_cleans_failed_derivation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "chambers-hybrid")
    raw = tmp_path / "source.nc"
    raw.write_bytes(b"invented source")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(_api, "download_file", fake_downloader_from(raw, calls))

    def derive(source_path: Path, output_path: Path, year: int) -> None:
        write_population(output_path, year=year)

    monkeypatch.setattr(_api, "derive_chambers_year", derive)
    cache = tmp_path / "cache"
    populations.download("chambers-hybrid:2020", cache_dir=cache)
    populations.download(
        "chambers-hybrid:2020",
        cache_dir=cache,
        refresh=True,
    )
    assert len(calls) == 2

    def fail_derive(source_path: Path, output_path: Path, year: int) -> None:
        output_path.write_text("partial")
        raise ValueError("bad source structure")

    monkeypatch.setattr(_api, "derive_chambers_year", fail_derive)
    with pytest.raises(ValueError, match="bad source structure"):
        populations.download("chambers-hybrid:2019", cache_dir=cache)
    assert not tuple(cache.rglob("chambers-hybrid-2019.tif.partial"))


def test_chambers_offline_without_shared_source_fails_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "chambers-hybrid")
    monkeypatch.setattr(
        _api,
        "download_file",
        lambda *args, **kwargs: pytest.fail("network called"),
    )

    with pytest.raises(ValueError, match="made no network request"):
        populations.download(
            "chambers-hybrid:2020",
            cache_dir=tmp_path / "cache",
            offline=True,
        )


def test_chambers_shared_source_lock_timeout_is_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_tiny_source(monkeypatch, "chambers-hybrid")
    real_file_lock = _api.FileLock

    class FailingSourceLock:
        def __enter__(self):
            raise Timeout("source.lock")

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def selective_lock(path, *, timeout):
        if Path(path).parent.name == "source":
            return FailingSourceLock()
        return real_file_lock(path, timeout=timeout)

    monkeypatch.setattr(_api, "FileLock", selective_lock)
    with pytest.raises(ValueError, match="Chambers source"):
        populations.download(
            "chambers-hybrid:2020",
            cache_dir=tmp_path / "cache",
        )


def test_archive_expansion_limit_is_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = use_tiny_source(
        monkeypatch,
        "ghsl-r2023a-mollweide-1km",
        delivery="zip",
    )
    member = source.archive_member(2020)
    assert member is not None
    archive = tmp_path / "ghsl.zip"
    with ZipFile(archive, "w") as output:
        output.writestr(member, b"larger than one byte")
    monkeypatch.setattr(_api, "_MAX_EXTRACTED_BYTES", 1)
    monkeypatch.setattr(_api, "download_file", fake_downloader_from(archive, []))

    with pytest.raises(ValueError, match="extraction safety limit"):
        populations.download(
            "ghsl-r2023a-mollweide-1km:2020",
            cache_dir=tmp_path / "cache",
        )
