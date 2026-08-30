# population-exposure

Estimate the number of people in any vector polygon or raster cell.

## Install

```sh
pip install population-exposure
```

Python 3.11 or newer is required. GeoPandas, Pyogrio, Rasterio, Shapely, and
Exactextract are installed as normal dependencies.

Read the [full documentation](https://palewi.re/docs/population-exposure/).

## Public API

The assignment operation is:

```python
def assign_population(
    hazard,
    population,
    *,
    cell_columns: str | Sequence[str] = ("longitude", "latitude"),
    population_column: str = "population",
    allow_overlaps: bool = False,
    hazard_band: int | None = None,
    conservation_tolerance: float = 1e-6,
) -> pd.DataFrame | gpd.GeoDataFrame | RasterAssignment: ...
```

| Hazard input | Population input | Return type |
|---|---|---|
| `pandas.DataFrame` | `pandas.DataFrame` | `pandas.DataFrame` |
| `geopandas.GeoDataFrame`, GeoJSON, Shapefile, or GeoPackage | catalog selection, local population-count GeoTIFF, or open Rasterio reader | `geopandas.GeoDataFrame` |
| local hazard GeoTIFF or open Rasterio reader | catalog selection, local population-count GeoTIFF, or open Rasterio reader | `RasterAssignment` |

`RasterAssignment` is the documented raster result type. It stores grid
metadata and source references, not a long DataFrame or eagerly loaded global
array.

The documentation uses the conventional `pe` alias.

The `pe.populations` namespace lists, describes, downloads, and registers catalog
data:

```python
import population_exposure as pe

pe.populations.list()
pe.populations.info("worldpop-global-1km:2020")
pe.populations.download("worldpop-global-1km:2020")
pe.populations.register("landscan-global:2024", "/downloads/landscan-global-2024.tif")
```

## Population catalog

Every selection has the exact form `source-id:YYYY`. Bare names, unsupported
years, malformed selections, and `latest` raise an error rather than choosing a
release for you.

| Source ID | Supported years | Meaning | Acquisition |
|---|---|---|---|
| `worldpop-global-1km` | 2000-2020 annually | Residential count, unconstrained 30 arc-second global mosaic | Anonymous WorldPop download |
| `ghsl-r2023a-mollweide-1km` | 1975-2020 every five years | Residential count, R2023A V1.0, World Mollweide 1 km | Anonymous JRC download |
| `gpwv4-r11-count` | 2000, 2005, 2010, 2015, 2020 | Residential count, GPWv4 Revision 11, 30 arc-seconds | NASA Earthdata token or local registration |
| `chambers-hybrid` | 1950-2020 annually | Residential count derived from 21 age bands, 0.25 degrees | Immutable Zenodo download |
| `landscan-global` | 2000-2024 annually | Ambient 24-hour count, 30 arc-seconds | Manual ORNL acquisition and local registration |

Inspect the license, citation, DOI, size, grid, and acquisition method before
downloading:

```python
import population_exposure as pe

selected = pe.populations.info("ghsl-r2023a-mollweide-1km:2020")
print(selected.license)
print(selected.citation)
print(selected.download_size)
print(selected.official_url)
```

WorldPop and GHSL download anonymously from documented publisher routes. GHSL
uses the release-qualified `R2023A` World Mollweide 1 km count grid; the 2025
and 2030 projection grids are not included. GPW uses the population-count
product, not density or the separately published UN-adjusted product. Pass a
user-owned Earthdata token for GPW:

```python
import population_exposure as pe

path = pe.populations.download(
    "gpwv4-r11-count:2020",
    earthdata_token=my_earthdata_token,
)
```

The token is used only in the official request. It is never logged, cached, or
written to a receipt. `EARTHDATA_TOKEN` is also supported for direct assignment
with a GPW selection.

The Chambers source is the 4,122,344,510-byte NetCDF-4 file at Zenodo DOI
`10.5281/zenodo.6011021`. It is downloaded and verified once. A requested
annual count raster is then derived in bounded windows by summing its 21 age
bands, so the full source is not loaded into memory or duplicated for each
year. The dataset is associated with the 2020 *Lancet Countdown* report, not a
Nature publication.

LandScan is different. Register and accept the terms at the
[official ORNL portal](https://landscan.ornl.gov/), download one explicit year,
extract its GeoTIFF, then register a copy:

```python
import population_exposure as pe

path = pe.populations.register(
    "landscan-global:2024",
    "/downloads/landscan-global-2024.tif",
)
```

Registration validates the source, year, count units, grid, values, and
plausible total, then copies the file into the cache without changing the
original. The package does not automate ORNL's form, use undocumented
endpoints or mirrors, redistribute LandScan data, or claim redistribution
rights.

### Cache, receipts, and offline work

The default cache is the operating system's normal application cache location
for the current user. It is not stored in the project or current working
directory, so verified files are shared by every project run by that user.
Each entry is grouped by its exact source, release, and year. A file lock keeps
concurrent processes from installing the same entry at once.

Use `cache_dir=` to select a root for one call, or set
`POPULATION_EXPOSURE_CACHE_DIR` to select a root for every call in processes
that inherit the setting. An explicit `cache_dir=` takes precedence over the
environment setting. Use the same absolute root across projects to share
verified downloads or registered files; a relative environment value is
resolved from each process's current working directory.

Downloads stream to a same-directory partial file, enforce size limits, resume
only after the server advertises and correctly accepts byte ranges, verify
publisher checksums where supplied, and replace the final path only after
validation. SHA-256 is computed when bytes are installed or refreshed. Later
cache checks use the recorded size and file identity/change timestamps so
multi-gigabyte files are not rehashed on every call; assignment still validates
the raster structure and values. A verified cache entry is reused unless
`refresh=True`.

```python
import population_exposure as pe

path = pe.populations.download(
    "worldpop-global-1km:2020",
    offline=True,
)
```

Offline mode makes no network calls. It uses an exact verified cached or
registered file in the selected cache root, or fails with instructions. Set
`POPULATION_EXPOSURE_OFFLINE=1` to make direct catalog assignment offline.

For sources that require manual acquisition, such as LandScan, register the
licensed file once in a shared cache root. Other projects using that root can
reuse the validated copy with the same exact selection; the original licensed
file is not changed, and its license terms still apply.

Every cached raster has an adjacent `.json` receipt with the exact selection,
official and landing URLs, retrieval time, local SHA-256, observed grid and
unit facts, license, citation, DOI, and processing note. Receipts describe cache
files only; they are not a general data-version system.

## Tables

```python
import pandas as pd

import population_exposure as pe

hazard = pd.DataFrame(
    {
        "cell": ["A", "B", "C", "D"],
        "county": ["North", "North", "South", "South"],
        "severity": ["warning", "watch", "warning", pd.NA],
    }
)
population = pd.DataFrame(
    {
        "cell": ["D", "B", "A", "C"],
        "population": [400.5, 200.0, 100.0, 300.25],
    }
)

exposed = pe.assign_population(hazard, population, cell_columns="cell")
```

The output preserves every hazard row and column, its index and order, missing
hazard values, and fractional population values. Cell keys are matched exactly;
they are not rounded or normalized.

| cell | county | severity | population |
|---|---|---|---:|
| A | North | warning | 100.0 |
| B | North | watch | 200.0 |
| C | South | warning | 300.25 |
| D | South | `<NA>` | 400.5 |

Use ordinary pandas operations for analysis:

```python
total = exposed["population"].sum()
by_county = exposed.groupby("county")["population"].sum()
by_severity = exposed.groupby("severity", dropna=False)["population"].sum()
county_severity = exposed.groupby(["county", "severity"], dropna=False)[
    "population"
].sum()
```

## Vector maps

```python
import geopandas as gpd
from shapely.geometry import box

import population_exposure as pe

hazard = gpd.GeoDataFrame(
    {"risk": ["high", "low"]},
    geometry=[box(0, 0, 1.5, 2), box(1.5, 0, 2, 2)],
    crs="EPSG:3857",
)

exposed = pe.assign_population(hazard, "worldpop-global-1km:2020")
```

The vector result preserves the original columns, geometry, CRS, index, and row
order, then appends `population`. Geometries are copied and reprojected to the
population raster CRS only for calculation. Exactextract sums each population
cell in proportion to polygon coverage, so boundary allocations remain
fractional and are never rounded. Population nodata is excluded.

Missing, empty, invalid, or non-polygon geometry raises an error. Every feature
must cover at least one valid population cell. Polygon overlaps raise by
default, because independently calculated totals could then be summed twice.
Use `allow_overlaps=True` only when independent, non-additive feature totals are
intentional.

Vector files are read with Pyogrio. A GeoPackage with multiple layers should be
opened by the caller as a `GeoDataFrame` so the layer choice is explicit.

## Raster maps

```python
import population_exposure as pe

exposed = pe.assign_population("hazard.tif", "ghsl-r2023a-mollweide-1km:2020")

hazard_values, population_values = exposed.read()

for window, hazard_block, population_block in exposed.iter_blocks():
    # Analyze one bounded pair of NumPy masked arrays at a time.
    pass
```

`RasterAssignment` exposes:

- `shape`, `crs`, `transform`, `bounds`, and the selected `hazard_band`;
- `attrs`, including source, covered, and aligned population totals;
- `read(window=None)`, returning hazard and aligned-population NumPy masked
  arrays for the same cells; and
- `iter_blocks()`, yielding `(window, hazard, population)` without loading the
  entire grid.

Population is aligned to the hazard grid and CRS with Rasterio/GDAL's
coverage-weighted `sum` resampling. Bilinear interpolation is never used. The
package compares the aligned total with Exactextract's coverage-aware
population total inside the hazard footprint. The allowed absolute difference
is:

```text
conservation_tolerance * max(1, covered_population)
```

The default tolerance is `1e-6`. Population outside the hazard extent and
population nodata are excluded from the covered total. Hazard nodata and areas
outside population coverage remain masked in reads.

A one-band hazard is selected automatically. Multiband hazards require an
explicit one-based `hazard_band`. Population rasters must contain exactly one
band of finite, non-negative counts. Metadata that explicitly describes density
is rejected; the package does not convert density to counts.

Path inputs are opened and closed for each operation. Caller-owned Rasterio
readers are never closed and must remain open while a `RasterAssignment` uses
them.

Catalog assignments add source ID, release, year, DOI, citation, license,
count/ambient meaning, local hash, observed raster facts, and processing note
under `result.attrs["population_source"]`. Custom local files and caller-owned
readers remain first-class. Their attrs contain only observed facts, local path
and hash where available, and no inferred license or citation.

To use any other local population-count raster:

```python
import population_exposure as pe

exposed = pe.assign_population("hazard.tif", "/data/custom-population-counts.tif")
```

## Scope

The package does not define bands or categories, group results, calculate
shares, convert density rasters, or resolve overlapping polygons. Callers can
use normal pandas, NumPy, or xarray operations after assignment.

See the [full documentation](https://palewi.re/docs/population-exposure/) and
the executable [table](examples/basic.py), [vector](examples/vector.py), and
[raster](examples/raster.py), and [catalog](examples/catalog.py) examples.

## Development

```sh
make bootstrap
make verify
```

The project uses uv, Ruff, ty, pytest, Hypothesis, and pre-commit.

### Live provider downloads

Ordinary tests use local fixtures and never contact population-data providers.
The separate **Live population downloads** workflow runs on the first day of
each month and can also be started manually. Its scheduled run downloads and
validates the current catalog selections for anonymous WorldPop and GHSL,
including receipts and an offline cache reuse.

Manual workflow choices cover WorldPop, GHSL, GPW, and Chambers individually.
GPW needs the `EARTHDATA_TOKEN` repository secret; the workflow fails before
downloading if that manual selection has no token. Its manual live test also
compares sums of exact 30-arc-second source-grid windows against CIESIN's
official 1-degree GPW population-count grid for the same year. It allows only
the publisher's float32 rounding precision for each count, plus a bounded
float64 summation error; low-value cells retain a one-person limit. It also
checks the aggregate total. It does not use the separate UN-WPP-adjusted
product. Chambers is manual only because its one shared source is
4,122,344,510 bytes. LandScan is excluded: its official portal requires
registration and license acceptance, and the package continues to test local
registration with fixtures rather than automating or redistributing it.

### Publishing releases

Before the first release, configure pending trusted publishers at
[TestPyPI](https://test.pypi.org/manage/account/publishing/) and
[PyPI](https://pypi.org/manage/account/publishing/):

| Setting | Value |
|---|---|
| Owner | `palewire` |
| Repository | `population-exposure` |
| Workflow | `publish.yml` |
| Package | `population-exposure` |
| TestPyPI environment | `testpypi` |
| PyPI environment | `pypi` |

Create matching GitHub environments named `testpypi` and `pypi`. Protect the
`pypi` environment with the required release approval policy.

To validate a version manually, choose **Run workflow** for **Publish package**,
select the release tag, and keep the only available target, `testpypi`. Install
the result with:

```sh
python -m pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ population-exposure
```

For PyPI, create a GitHub Release from the exact version tag and publish that
release. Publishing starts only for the `published` release event and uses the
validated artifacts from that workflow run. No PyPI API token is used.
