"""The small, explicit built-in population source catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

from population_exposure.populations._models import (
    Acquisition,
    PopulationMeaning,
    SelectionInfo,
    SourceInfo,
)

Delivery = Literal["geotiff", "zip", "chambers", "manual"]


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Public source facts plus the checks needed to acquire one file."""

    source_id: str
    release: str
    title: str
    publisher: str
    years: tuple[int, ...]
    acquisition: Acquisition
    delivery: Delivery
    landing_page: str
    url_template: str
    doi: str | None
    doi_by_year: Mapping[int, str]
    license: str
    citation_template: str
    units: str
    meaning: PopulationMeaning
    crs: str
    resolution: str
    format: str
    download_size: str
    notes: tuple[str, ...]
    filename_template: str
    archive_member_template: str | None
    expected_width: int | None
    expected_height: int | None
    expected_resolution: tuple[float, float] | None
    expected_bounds: tuple[float, float, float, float] | None
    expected_nodata: tuple[float, ...] | None
    plausible_total: tuple[float, float] | None
    max_download_bytes: int
    exact_download_bytes: int | None = None
    publisher_checksum: str | None = None
    expected_bounds_by_year: Mapping[int, tuple[float, float, float, float] | None] = (
        field(default_factory=lambda: MappingProxyType({}))
    )
    expected_nodata_by_year: Mapping[int, tuple[float, ...] | None] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def source_info(self) -> SourceInfo:
        """Return the public source record."""
        return SourceInfo(
            source_id=self.source_id,
            release=self.release,
            title=self.title,
            publisher=self.publisher,
            supported_years=self.years,
            acquisition=self.acquisition,
            landing_page=self.landing_page,
            doi=self.doi,
            license=self.license,
            citation=self.citation_template.format(year="{year}"),
            units=self.units,
            meaning=self.meaning,
            crs=self.crs,
            resolution=self.resolution,
            format=self.format,
            download_size=self.download_size,
            notes=self.notes,
        )

    def selection_info(self, year: int) -> SelectionInfo:
        """Return the public record for one explicit year."""
        doi = self.doi_by_year.get(year, self.doi)
        return SelectionInfo(
            selection=f"{self.source_id}:{year}",
            source_id=self.source_id,
            release=self.release,
            year=year,
            title=self.title,
            publisher=self.publisher,
            acquisition=self.acquisition,
            landing_page=self.landing_page,
            official_url=self.url_template.format(year=year),
            doi=doi,
            license=self.license,
            citation=self.citation_template.format(year=year),
            units=self.units,
            meaning=self.meaning,
            crs=self.crs,
            resolution=self.resolution,
            format=self.format,
            download_size=self.download_size,
            expected_filename=self.filename_template.format(year=year),
            notes=self.notes,
        )

    def archive_member(self, year: int) -> str | None:
        """Return the expected GeoTIFF member name for an archive."""
        if self.archive_member_template is None:
            return None
        return self.archive_member_template.format(year=year)

    def expected_bounds_for(
        self,
        year: int,
    ) -> tuple[float, float, float, float] | None:
        """Return documented raster bounds for a selected year."""
        if year in self.expected_bounds_by_year:
            return self.expected_bounds_by_year[year]
        return self.expected_bounds

    def expected_nodata_for(self, year: int) -> tuple[float, ...] | None:
        """Return documented raster nodata values for a selected year."""
        if year in self.expected_nodata_by_year:
            return self.expected_nodata_by_year[year]
        return self.expected_nodata


_CC_BY_4 = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
_GLOBAL_TOTAL = (100_000_000.0, 20_000_000_000.0)

WORLDPOP = SourceSpec(
    source_id="worldpop-global-1km",
    release="Global 2000-2020 1 km mosaics",
    title="WorldPop unconstrained global 1 km population counts",
    publisher="WorldPop and CIESIN",
    years=tuple(range(2000, 2021)),
    acquisition="automatic",
    delivery="geotiff",
    landing_page="https://hub.worldpop.org/geodata/listing?id=64",
    url_template=(
        "https://data.worldpop.org/GIS/Population/Global_2000_2020/"
        "{year}/0_Mosaicked/ppp_{year}_1km_Aggregated.tif"
    ),
    doi="10.5258/SOTON/WP00647",
    doi_by_year=MappingProxyType({}),
    license=_CC_BY_4,
    citation_template=(
        "WorldPop and Center for International Earth Science Information Network "
        "(2018). Global High Resolution Population Denominators Project. "
        "https://doi.org/10.5258/SOTON/WP00647"
    ),
    units="population count per cell",
    meaning="residential",
    crs="EPSG:4326",
    resolution="30 arc-seconds (about 1 km at the equator)",
    format="single-band GeoTIFF",
    download_size="roughly 0.8-1.2 GB for one year",
    notes=(
        "This is the unconstrained global mosaic series.",
        "The catalog does not describe this route as the separately published "
        "UN-adjusted series.",
    ),
    filename_template="ppp_{year}_1km_Aggregated.tif",
    archive_member_template=None,
    expected_width=43_200,
    expected_height=18_720,
    expected_resolution=(1 / 120, 1 / 120),
    expected_bounds=(
        -180.001249265,
        -71.99208284398998,
        179.99874929500004,
        84.00791653201003,
    ),
    expected_nodata=(-3.4028234663852886e38,),
    plausible_total=_GLOBAL_TOTAL,
    max_download_bytes=1_500_000_000,
    expected_bounds_by_year=MappingProxyType(
        {
            2020: (
                -180.001249265,
                -72.00041617728999,
                179.99874929500004,
                83.99958319871001,
            ),
        }
    ),
    expected_nodata_by_year=MappingProxyType(
        {
            2000: (3.4028234663852886e38,),
        }
    ),
)

GHSL = SourceSpec(
    source_id="ghsl-r2023a-mollweide-1km",
    release="R2023A V1.0",
    title="GHS-POP R2023A World Mollweide 1 km population counts",
    publisher="European Commission Joint Research Centre",
    years=tuple(range(1975, 2021, 5)),
    acquisition="automatic",
    delivery="zip",
    landing_page="https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php",
    url_template=(
        "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
        "GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_54009_1000/"
        "V1-0/GHS_POP_E{year}_GLOBE_R2023A_54009_1000_V1_0.zip"
    ),
    doi="10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE",
    doi_by_year=MappingProxyType({}),
    license=_CC_BY_4,
    citation_template=(
        "Schiavina, M., Freire, S., Carioli, A., and MacManus, K. (2023). "
        "GHS-POP R2023A - GHS population grid multitemporal (1975-2030). "
        "European Commission, Joint Research Centre. "
        "https://doi.org/10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE"
    ),
    units="population count per cell",
    meaning="residential",
    crs="ESRI:54009",
    resolution="1,000 metres on the World Mollweide equal-area grid",
    format="ZIP archive containing a single-band GeoTIFF",
    download_size="roughly 300 MB for one epoch (the 2020 archive is about 307 MB)",
    notes=(
        "The source ID includes the R2023A release, Mollweide projection, and 1 km "
        "resolution.",
        "The catalog includes estimates through 2020; publisher projections for "
        "2025 and 2030 are intentionally excluded.",
    ),
    filename_template="GHS_POP_E{year}_GLOBE_R2023A_54009_1000_V1_0.tif",
    archive_member_template="GHS_POP_E{year}_GLOBE_R2023A_54009_1000_V1_0.tif",
    expected_width=36_082,
    expected_height=18_000,
    expected_resolution=(1000.0, 1000.0),
    expected_bounds=(-18_041_000.0, -9_000_000.0, 18_041_000.0, 9_000_000.0),
    expected_nodata=(-200.0,),
    plausible_total=_GLOBAL_TOTAL,
    max_download_bytes=750_000_000,
)

GPW = SourceSpec(
    source_id="gpwv4-r11-count",
    release="Revision 11 (v4.11)",
    title="GPWv4 Revision 11 population count",
    publisher="CIESIN, Columbia University and NASA SEDAC",
    years=(2000, 2005, 2010, 2015, 2020),
    acquisition="earthdata",
    delivery="zip",
    landing_page=(
        "https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-count-rev11"
    ),
    url_template=(
        "https://data.earthdata.nasa.gov/nasa-earth/human-dimensions/"
        "sedac-root/downloads/data/gpw-v4/gpw-v4-population-count-rev11/"
        "gpw-v4-population-count-rev11_{year}_30_sec_tif.zip"
    ),
    doi="10.7927/H4JW8BX5",
    doi_by_year=MappingProxyType({}),
    license=_CC_BY_4,
    citation_template=(
        "Center for International Earth Science Information Network - CIESIN - "
        "Columbia University. (2018). Gridded Population of the World, Version 4 "
        "(GPWv4): Population Count, Revision 11. NASA SEDAC. "
        "https://doi.org/10.7927/H4JW8BX5"
    ),
    units="population count per cell",
    meaning="residential",
    crs="EPSG:4326",
    resolution="30 arc-seconds (about 1 km at the equator)",
    format="Earthdata-authenticated ZIP archive containing a single-band GeoTIFF",
    download_size="roughly 405 MB for one 30 arc-second year",
    notes=(
        "This is population count, not density and not the UN-adjusted variant.",
        "A user-owned Earthdata token is used only for the request and is never "
        "stored or written to receipts.",
    ),
    filename_template="gpw_v4_population_count_rev11_{year}_30_sec.tif",
    archive_member_template="gpw_v4_population_count_rev11_{year}_30_sec.tif",
    expected_width=43_200,
    expected_height=21_600,
    expected_resolution=(1 / 120, 1 / 120),
    expected_bounds=(-180.0, -90.0, 180.0, 90.0),
    expected_nodata=(-3.4028234663852886e38,),
    plausible_total=_GLOBAL_TOTAL,
    max_download_bytes=600_000_000,
)

CHAMBERS = SourceSpec(
    source_id="chambers-hybrid",
    release="Zenodo record 6011021",
    title="Hybrid gridded demographic data for the world, 1950-2020",
    publisher="Jonathan Chambers",
    years=tuple(range(1950, 2021)),
    acquisition="automatic",
    delivery="chambers",
    landing_page="https://zenodo.org/records/6011021",
    url_template=(
        "https://zenodo.org/api/records/6011021/files/"
        "demographics_hybrid_1950_2020_15_min.nc/content"
    ),
    doi="10.5281/zenodo.6011021",
    doi_by_year=MappingProxyType({}),
    license=_CC_BY_4,
    citation_template=(
        "Chambers, J. (2022). Hybrid gridded demographic data for the world, "
        "1950-2020 0.25 degree resolution. Zenodo. "
        "https://doi.org/10.5281/zenodo.6011021"
    ),
    units="population count per cell",
    meaning="residential",
    crs="EPSG:4326",
    resolution="0.25 degrees",
    format="derived single-band GeoTIFF from one cached NetCDF-4 source",
    download_size=(
        "4,122,344,510 bytes once for the shared source; each requested year is "
        "derived locally"
    ),
    notes=(
        "The source contains all years and 21 age bands.",
        "One annual total is derived in bounded windows without loading the full "
        "cube into memory.",
        "The dataset is associated with the 2020 Lancet Countdown report, not a "
        "Nature publication.",
    ),
    filename_template="chambers-hybrid-{year}.tif",
    archive_member_template=None,
    expected_width=1_440,
    expected_height=721,
    expected_resolution=(0.25, 0.25),
    expected_bounds=None,
    expected_nodata=(float("nan"),),
    plausible_total=_GLOBAL_TOTAL,
    max_download_bytes=4_122_344_510,
    exact_download_bytes=4_122_344_510,
    publisher_checksum="md5:b0a9c1354435f104743b9a8165df457d",
)

LANDSCAN = SourceSpec(
    source_id="landscan-global",
    release="annual release selected by year",
    title="LandScan Global ambient population",
    publisher="Oak Ridge National Laboratory",
    years=tuple(range(2000, 2025)),
    acquisition="manual",
    delivery="manual",
    landing_page="https://landscan.ornl.gov/",
    url_template="https://landscan.ornl.gov/",
    doi=None,
    doi_by_year=MappingProxyType(
        {
            2023: "10.48690/1531770",
            2024: "10.48690/1532445",
        }
    ),
    license="ORNL LandScan End User License Agreement; redistribution not granted",
    citation_template=(
        "Oak Ridge National Laboratory. ({year}). LandScan Global {year}. "
        "Oak Ridge National Laboratory."
    ),
    units="ambient population count per cell",
    meaning="ambient",
    crs="EPSG:4326",
    resolution="30 arc-seconds (about 1 km at the equator)",
    format="manually acquired single-band GeoTIFF",
    download_size="release-dependent manual download from the ORNL portal",
    notes=(
        "Acquire the selected year from ORNL after registration and license "
        "acceptance, then call populations.register().",
        "The catalog does not automate the form, use undocumented endpoints, or "
        "redistribute LandScan files.",
    ),
    filename_template="landscan-global-{year}.tif",
    archive_member_template=None,
    expected_width=43_200,
    expected_height=20_880,
    expected_resolution=(1 / 120, 1 / 120),
    expected_bounds=(-180.0, -90.0, 180.0, 84.0),
    expected_nodata=None,
    plausible_total=_GLOBAL_TOTAL,
    max_download_bytes=10_000_000_000,
)

SOURCES: MappingProxyType[str, SourceSpec] = MappingProxyType(
    {source.source_id: source for source in (WORLDPOP, GHSL, GPW, CHAMBERS, LANDSCAN)}
)
