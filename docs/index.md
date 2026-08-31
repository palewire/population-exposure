# population-exposure

Add population counts to hazard tables, vector polygons, and raster cells.

## Installation

```console
pip install population-exposure
```

Python 3.11 or newer is required.

## Population registry

The built-in registry provides verified population-count sources. Select one
with its exact `source-id:YYYY` identifier; `pe.populations.info()` reports its
license, citation, and download details before anything is downloaded.

| Source | Population, years, and grid | How counts are placed | Choose it for | Access and main limitation |
| --- | --- | --- | --- | --- |
| [`worldpop-global-1km`](https://hub.worldpop.org/geodata/listing?id=64) | **Residential**, modeled estimates.<br>2000-2020, yearly.<br>30 arc-seconds (about 1 km). | The built-in series is WorldPop's **unconstrained Global 2000-2020 1 km mosaics**. | Annual residential estimates. | Automatic CC BY download (about 0.8-1.2 GB/year).<br>**Limitation:** a modeled grid, not local census observations. |
| [`ghsl-r2023a-mollweide-1km`](https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php) | **Residential** counts.<br>1975-2020, five-year epochs.<br>1 km World Mollweide equal-area grid. | Census and administrative totals are distributed using built-up distribution, volume, and classification. | Comparable historical residential snapshots and area-based work. | Automatic CC BY download (about 300 MB/epoch).<br>**Limitation:** fine cells do not mean fine census inputs. |
| [`gpwv4-r11-count`](https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-count-rev11) | **Residential** census baseline.<br>2000-2020, five-year epochs.<br>30 arc-seconds (about 1 km). | Lightly modeled: census counts are allocated across their source areas. | A transparent census-based baseline. | CC BY download with Earthdata authentication (about 405 MB/epoch).<br>**Limitation:** source-area detail varies by place. |
| [`chambers-hybrid`](https://zenodo.org/records/6011021) | **Residential** counts.<br>1950-2020, yearly.<br>0.25 degrees. | Hybrid of GPWv4, ISIMIP Histsoc, and UN World Population Prospects demographic data. | Long annual history, especially climate work aligned to ERA5. | Automatic CC BY download; 4.1 GB shared source.<br>**Limitation:** coarse spatial detail. |
| [`landscan-global`](https://landscan.ornl.gov/) | **Ambient** population (average 24-hour presence).<br>2000-2024, yearly.<br>30 arc-seconds (about 1 km). | Ambient counts represent where people may be present, rather than home residence. | Disaster response and presence-style exposure. | Manual licensed acquisition; no redistribution.<br>**Limitation:** not a residential population source. |

Choose first by population meaning and date, then by the quality and detail of
the source data at your place and the scale of the analysis. A source's output
grid is not its census precision or a measure of accuracy. For GHS-POP, choose
an epoch close to the hazard date; it estimates residents, not people physically
present.

The registry does not download LandScan or bypass its license terms. GPW needs
an Earthdata token. WorldPop, GHSL, and Chambers download from their publishers,
but their file sizes may still be substantial.

Download a selected raster once, then reuse the returned local path in later
vector or raster assignments:

```python
import population_exposure as pe

population = pe.populations.download("worldpop-global-1km:2020")
```

`pe.populations.download()` verifies and caches the selected raster. By default,
it uses the current user's operating-system cache, so verified rasters are
shared across that user's projects. Calling it again with the same selection
reuses the verified cached file unless `refresh=True`; use `cache_dir=` or
`POPULATION_EXPOSURE_CACHE_DIR` to select another cache root.

LandScan requires a separate, manually acquired annual GeoTIFF. After accepting
the ORNL terms and downloading the 2024 file, register it once to obtain the
local path used below:

```python
landscan_population = pe.populations.register(
    "landscan-global:2024",
    "/path/to/your-2024-landscan.tif",
)
```

## Quick start with a registry source

Use the observed 2024 Hurricane Helene wind swath published by the
[National Hurricane Center](https://www.nhc.noaa.gov/gis/). This post-storm
best-track data is not a forecast cone; 64 knots is hurricane-force wind.

```python
import geopandas as gpd

import population_exposure as pe

url = "zip+https://www.nhc.noaa.gov/gis/best_track/al092024_best_track.zip"
winds = gpd.read_file(url, layer="AL092024_windswath")
hurricane_force = winds[winds["RADII"] == 64].dissolve()
exposed = pe.assign_population(hurricane_force, population)
print(exposed["population"].sum())
```

NWS information is public domain unless specifically noted otherwise; see the
[NOAA/NWS use terms](https://www.weather.gov/disclaimer).

## Raster hazard exposure

The [M 7.1 Ridgecrest earthquake](https://earthquake.usgs.gov/earthquakes/eventpage/ci38457511)
struck near Searles Valley, California, on July 6, 2019. Its official USGS
ShakeMap archive includes a 10 MB Modified Mercalli Intensity (MMI) grid.
The archive uses an ESRI float raster, so this example saves its MMI field as a
GeoTIFF with its documented WGS 84 grid before assigning population.

```python
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import rasterio

import population_exposure as pe

archive_url = (
    "https://earthquake.usgs.gov/product/shakemap/ci38457511/"
    "atlas/1594160054783/download/raster.zip"
)
archive_path = Path("ridgecrest-shakemap-raster.zip")
hazard_directory = Path("ridgecrest-shakemap")
urlretrieve(archive_url, archive_path)
with ZipFile(archive_path) as archive:
    archive.extract("mmi_mean.flt", hazard_directory)
    archive.extract("mmi_mean.hdr", hazard_directory)

with rasterio.open(hazard_directory / "mmi_mean.flt") as source:
    profile = source.profile | {"driver": "GTiff", "crs": "EPSG:4326"}
    with rasterio.open("ridgecrest-mmi.tif", "w", **profile) as destination:
        destination.write(source.read(1), 1)

assignment = pe.assign_population("ridgecrest-mmi.tif", landscan_population)
mmi, people = assignment.read()
pager_vi_or_greater_population = people[mmi >= 5.5].sum()
print(pager_vi_or_greater_population)
```

This is raster-to-raster assignment: population counts are regridded onto the
MMI cells. The final sum uses the PAGER-style MMI VI-or-greater threshold,
which begins at 5.5. It is different from the vector coverage allocation above
and the exact table-coordinate join below.

## `assign_population()` options

`assign_population(hazard, population, *, cell_columns=("longitude",
"latitude"), population_column="population", allow_overlaps=False,
hazard_band=None, conservation_tolerance=1e-6)` selects the matching behavior
from the input types.

| Option | Meaning |
| --- | --- |
| `hazard` | A pandas table, GeoPandas frame, supported vector file, GeoTIFF path, or open Rasterio reader. |
| `population` | A pandas table for table hazards. For maps and rasters, use a one-band population-count GeoTIFF, open Rasterio reader, or exact catalog selection. |
| `cell_columns` | One table key column or a sequence of table key columns. It is used only for table assignment. The default is `longitude` and `latitude`. |
| `population_column` | Name of the new output population column. It defaults to `population` and cannot overwrite an existing hazard column. |
| `allow_overlaps` | Allows overlapping vector polygons. It is `False` by default because adding independent overlapping totals would count shared areas more than once. It applies only to vector hazards. |
| `hazard_band` | A 1-based hazard band number. It is used only for multiband hazard rasters; a one-band raster selects band 1 automatically. |
| `conservation_tolerance` | The allowed relative difference when a population raster is reprojected onto a hazard raster. It applies only to raster hazards and defaults to `1e-6`. |

Table keys must be complete and unique in both inputs. They are never rounded,
trimmed, or otherwise normalized. An unmatched hazard row raises an error
rather than receiving a guessed or missing population value.

Vector inputs must have a coordinate system and valid, non-empty `Polygon` or
`MultiPolygon` geometry. Each polygon must cover valid population data.

## Raster results

`pe.RasterAssignment` records the hazard grid through `shape`, `crs`, `transform`,
`bounds`, and `hazard_band`. Its `attrs` mapping records the assignment method,
population totals, tolerance, and catalog facts when relevant.

- `read(window=None)` returns `(hazard, population)`: two masked NumPy arrays
  for the same cells, optionally limited to a Rasterio window.
- `iter_blocks()` yields `(window, hazard, population)` in bounded blocks, so a
  large raster need not be loaded all at once.

Population is aligned to the hazard coordinate system and grid with
coverage-weighted sum resampling, not interpolation. The package compares the
aligned total with the population covered by the hazard footprint and raises an
error when the difference is above:

```text
conservation_tolerance * max(1, covered_population)
```

Population outside the hazard footprint and population no-data cells are not
part of that covered total. Hazard no-data and areas without population remain
masked. Population inputs must be finite, non-negative **counts per cell**;
rasters explicitly marked as density are rejected instead of being silently
treated as counts.

Paths are opened and closed for each operation. An open Rasterio reader belongs
to the caller and is never closed by this package. Keep caller-owned hazard and
population readers open for as long as a `RasterAssignment` that refers to
them is used.

## Bring your own population raster

Use a one-band GeoTIFF containing finite, non-negative population counts when
you already have an appropriate local dataset. For polygon hazards, pass its
path directly:

```python
import geopandas as gpd

import population_exposure as pe

hazard = gpd.read_file("flood-zones.geojson")
exposed = pe.assign_population(hazard, "population-counts.tif")
```

The result keeps the original geometry and coordinate system. Population cells
crossing a polygon boundary are counted by their covered share, so totals can
be fractional.

## Tabular heat exposure

ERA5-derived temperatures and the Chambers 0.25-degree grid use matching
longitude and latitude cell centers. With both tables keyed by those centers,
`pe.assign_population()` performs an exact cell-coordinate join, not a spatial
overlay. The values below are a small, high-temperature subset.

```python
import pandas as pd

import population_exposure as pe

chambers = pe.populations.info("chambers-hybrid:2020")
print(chambers.resolution)  # 0.25 degrees

temperature = pd.DataFrame(
    {
        "longitude": [-76.0, -75.75],
        "latitude": [38.75, 38.5],
        "daily_max_c": [38.4, 39.1],
    }
)
population = pd.DataFrame(
    {
        "longitude": [-76.0, -75.75],
        "latitude": [38.75, 38.5],
        "population": [20500.0, 17800.0],
    }
)

exposed = pe.assign_population(temperature, population)
hot_population = exposed.loc[exposed["daily_max_c"] >= 38, "population"].sum()
print(hot_population)
```

`exposed` keeps the temperature columns, index, and row order, with a new
`population` column.

## API reference

The public API is intentionally small.

```{eval-rst}
.. autofunction:: population_exposure.assign_population

.. autoclass:: population_exposure.RasterAssignment
   :members: read, iter_blocks

.. automodule:: population_exposure.populations
   :members: list, info, download, register, SourceInfo, SelectionInfo
```

## Links

- [Source code](https://github.com/palewire/population-exposure)
- [Issue tracker](https://github.com/palewire/population-exposure/issues)
- [Changelog](https://github.com/palewire/population-exposure/blob/main/CHANGELOG.md)
- [PyPI package](https://pypi.org/project/population-exposure/)

## About

Ben Welsh and Casey Miller released `population-exposure` in 2026 after developing it for the [Reuters Climate Monitor](https://www.reuters.com/graphics/CLIMATE-AUTOMATED/MONITOR/akpeykqqapr/). GitHub Copilot, an AI-powered text generator, helped
draft this documentation.
