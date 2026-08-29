# Data model and behavior

`assign_population()` answers one question: how much population belongs to each
input row, polygon feature, or raster cell? It does not classify or aggregate
hazards.

## Tabular assignment

A hazard table contains caller-defined rows and columns. A population table
contains one finite, non-negative population value per cell.

One or more `cell_columns` join population onto each hazard row. Keys are
matched exactly as supplied. Null and duplicate keys are rejected in both
tables, and every hazard row must match one population row. Extra population
rows are allowed.

The returned `DataFrame` preserves hazard columns, row order, index, missing
hazard values, and fractional population. Neither input is mutated.

## Vector assignment

A vector hazard is a `GeoDataFrame` or a local GeoJSON, Shapefile, or GeoPackage
path. The population input is an exact catalog selection, a local one-band
population-count GeoTIFF, or an open Rasterio reader.

The hazard CRS and population CRS are required. A working copy of the polygons
is reprojected to the population CRS. Exactextract weights each population cell
by the fraction covered by each polygon. Population nodata is ignored.

The returned `GeoDataFrame` retains the original geometry, CRS, columns, index,
and feature order. The configured population column is appended as unrounded
floating-point values. Assignment facts are stored under
`result.attrs["population_assignment"]`.

Only valid, non-empty `Polygon` and `MultiPolygon` geometries are accepted.
Every polygon must overlap valid population data. Features with positive-area
overlap are rejected before extraction unless `allow_overlaps=True`. When
overlaps are allowed, feature values are independent and must not be added
together without accounting for duplicated areas.

## Raster assignment

A raster hazard is a local GeoTIFF path or open Rasterio reader. Its population
input may also be an exact catalog selection. Both rasters need a CRS, a finite
invertible transform, positive dimensions, and finite bounds. The population
raster must have exactly one band of finite, non-negative counts. Nodata may be
finite or NaN and is excluded. Metadata that explicitly identifies density is
rejected because density cannot be silently treated as count.

The result is a `RasterAssignment`, not a cell-per-row table. It records the
hazard grid and selected one-based band. `read()` returns paired NumPy masked
arrays. `iter_blocks()` returns the same pair in windows no larger than 256 by
256 cells, which keeps memory use bounded for large rasters.

Population is virtually reprojected to the hazard CRS, transform, width, and
height using Rasterio's `sum` resampling. This distributes source-cell counts
according to coverage rather than interpolating them. Before returning, the
package totals the aligned raster in bounded windows and compares it with an
Exactextract sum over the hazard footprint. The allowed difference is
`conservation_tolerance * max(1, covered_population)`.

`attrs` records the source population total, the total covered by the hazard
footprint, the aligned total, the tolerance, and the resampling method.
Population outside the hazard extent is intentionally excluded. Areas in the
hazard grid without valid population remain masked.

Catalog results also record a `population_source` mapping with the exact source
ID, release, year, DOI, citation, license, population meaning, local SHA-256,
observed raster facts, and processing note. Vector results use the same mapping.
Custom raster paths and caller-owned readers record only observed facts, path
and hash where available, and do not infer publisher or legal metadata.

Single-band hazards select band 1. Multiband hazards require `hazard_band`.
Path inputs are closed after each operation. Caller-owned readers remain open;
they must stay open for later reads from a result that refers to them.

## Catalog resolution

Catalog selections use the exact grammar `source-id:YYYY`. Existing files are
handled as paths before catalog parsing. A missing path-like value fails as a
path unless it exactly uses catalog grammar. Bare source names, `latest`,
unsupported years, and malformed selections fail with guidance.

Automatic sources stream official files to a partial path and install only
after source-specific grid, count, year, value, size, and checksum checks.
Verified cache entries are grouped by source, release, and year. Adjacent JSON
receipts identify the installed bytes. Offline mode never makes a network call.

Chambers annual rasters are derived by reading one requested year's 21 age
bands from the shared NetCDF-4 source in bounded windows. LandScan is never
downloaded automatically: the caller acquires it from ORNL and `register()`
copies and validates it without mutating the original.

## Caller-owned analysis

The package does not define hazard bands or categories, calculate grouped
totals or shares, convert density to counts, or choose precedence for
overlapping features. Use pandas, NumPy, or xarray for those decisions after
assignment.
