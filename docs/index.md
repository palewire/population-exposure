# population-exposure

Estimate the number of people in any vector polygon or raster cell.

## Installation

```console
pip install population-exposure
```

## Population registry

The built-in registry provides verified population-count sources. Select one
with its exact `source-id:YYYY` identifier; `pe.populations.info()` reports its
license, citation, and download details before anything is downloaded.

| Source ID | Publisher | Available years and grid | Access and size | Best suited for |
| --- | --- | --- | --- | --- |
| `worldpop-global-1km` | [WorldPop and CIESIN](https://hub.worldpop.org/geodata/listing?id=64) | 2000-2020, yearly; 30 arc-seconds (about 1 km) | Automatic CC BY download; roughly 0.8-1.2 GB per year | Annual residential estimates |
| `ghsl-r2023a-mollweide-1km` | [European Commission Joint Research Centre](https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php) | 1975-2020, every 5 years; 1 km World Mollweide equal-area grid | Automatic CC BY download; roughly 300 MB per epoch | Historical residential snapshots |
| `gpwv4-r11-count` | [CIESIN, Columbia University and NASA SEDAC](https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-count-rev11) | 2000, 2005, 2010, 2015, 2020; 30 arc-seconds (about 1 km) | CC BY download requires your Earthdata token; roughly 405 MB per year | Those specific GPW count releases |
| `chambers-hybrid` | [Jonathan Chambers](https://zenodo.org/records/6011021) | 1950-2020, yearly; 0.25 degrees | Automatic CC BY download, but the shared source is 4.1 GB once | Long annual history when a coarser grid is suitable |
| `landscan-global` | [Oak Ridge National Laboratory](https://landscan.ornl.gov/) | 2000-2024, yearly; 30 arc-seconds (about 1 km) | Manually download after ORNL registration and license acceptance; no redistribution | Ambient, 24-hour population estimates |

The registry does not download LandScan or bypass its license terms. GPW needs
an Earthdata token. WorldPop, GHSL, and Chambers download from their publishers,
but their file sizes may still be substantial.

## Quick start with a registry source

Use the observed 2024 Hurricane Helene wind swath published by the
[National Hurricane Center](https://www.nhc.noaa.gov/gis/). This post-storm
best-track data is not a forecast cone; 64 knots is hurricane-force wind.

```python
import geopandas as gpd

import population_exposure as pe

population_path = pe.populations.download("worldpop-global-1km:2020")
url = "zip+https://www.nhc.noaa.gov/gis/best_track/al092024_best_track.zip"
winds = gpd.read_file(url, layer="AL092024_windswath")
hurricane_force = winds[winds["RADII"] == 64].dissolve()
exposed = pe.assign_population(hurricane_force, population_path)
print(exposed["population"].sum())
```

`pe.populations.download()` verifies and caches the selected raster. Calling it
again with the same selection reuses the verified cached file unless
`refresh=True`. NWS information is public domain unless specifically noted
otherwise; see the [NOAA/NWS use terms](https://www.weather.gov/disclaimer).

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

## Tabular data

For advanced use with tables, give the hazard and population tables the same
complete, unique cell key:

```python
import pandas as pd

import population_exposure as pe

hazard = pd.DataFrame({"cell": ["A", "B"], "risk": ["high", "low"]})
population = pd.DataFrame({"cell": ["A", "B"], "population": [100.0, 200.0]})

exposed = pe.assign_population(hazard, population, cell_columns="cell")
```

`exposed` keeps the hazard columns, index, and row order, with a new
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
