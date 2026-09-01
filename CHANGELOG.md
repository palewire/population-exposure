# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0rc1] - 2026-08-31

### Changed

- Report missing and partial population support explicitly. Coverage, which
  asks whether the population grid reaches over the hazard, is now reported
  separately from data support, which asks whether the grid holds values there.
  Coverage is strict by default. Partial data support is allowed and reported
  in every result, because coastlines are made of it; only zero data support
  raises, and that can be allowed explicitly.
- Stop reporting a vector feature that sits entirely on no-data cells as `0.0`.
  It now raises the new `MissingPopulationDataError`, or returns `NaN` with
  `allow_missing_population_data=True`. No-data records the absence of an
  answer, not an empty place, so it is never turned into a count. Real
  zero-count cells are still returned as `0`.
- Add `population_data_fraction` and `population_data_complete` to every vector
  result, so a partly no-data feature says how much real data stood behind its
  total. Ordinary coastal work is unchanged and still allowed. Raster
  assignments report the same two facts about the hazard footprint, measured on
  the population raster's own cells rather than the sum-resampled output, which
  marks a cell valid when any part of it had a value. The data share is valid
  source-cell area in the population raster's coordinate plane, not physical
  Earth-surface area like `population_coverage_fraction`, and neither share is
  a population multiplier.
- Require a hazard raster to sit entirely inside the population raster by
  default. Cells beyond its edge were returned masked, which hid them rather
  than reporting them. `allow_partial_coverage` now applies to raster hazards
  as well as vector ones, and records the covered share.
- Add `population_coverage_fraction`, `population_coverage_complete`,
  `population_data_fraction`, `population_data_complete`,
  `population_partial_coverage_allowed`, and `population_missing_data_allowed`
  to `RasterAssignment.attrs`.
- Align `population_coverage_fraction` with `population_coverage_complete`. A
  fully covered feature now reports a share of exactly `1.0`, so the measured
  share and the completeness flag can no longer disagree.
- Describe the raster conservation check as a computational test of regridding
  arithmetic. It compares totals over available support and cannot establish
  completeness or uncertainty; the coverage and data-support facts do that.
- Require hazard and population inputs to share one coordinate system. A
  mismatch now raises `CrsMismatchError` and explains both the manual and the
  automatic route, instead of silently reprojecting.
- Require vector features to sit inside the population raster. A feature that
  reaches beyond its edge now raises `PartialCoverageError` instead of
  returning a total that quietly omits the missing area.
- Change `conservation_tolerance` to default to `None`, which resolves to
  `1e-6` on a shared coordinate system and `1e-3` when reprojection is used.
  Measured cross-system warping differences were between `6e-5` and `4e-4`.

### Fixed

- Derive Chambers Hybrid annual totals directly from its published NetCDF-4
  layout. The file stores 720 latitude rows and 14 age bands, which GDAL cannot
  read as the 721-row, 21-band raster previously assumed.
- Correct source licenses, citations, and caveats in the population catalog,
  including the GHS-POP R2023A European Commission reuse notice and the
  LandScan 2024 DOI citation.
- Reject same-CRS geographic polygon rings with unsplit edges crossing the
  antimeridian instead of calculating population for the long way around the
  world. Properly split and supported unwrapped-domain polygons remain valid.
- Raster hazards entirely outside the population raster now raise
  `PartialCoverageError`; overlapping no-data and zero-count areas remain
  valid.
- Report `population_coverage_fraction` as a share of physical Earth-surface
  area rather than planar area in the raster's coordinate system. It remains a
  coverage description, not a population multiplier.
- Document the conservative `1e-6` relative area bound for 0.1-degree
  geographic boundary splitting, and distinguish invalid-area and no-area
  failures from the half-Earth geodesic limitation.
- Split longitude-latitude boundary edges before calculating physical area, so
  ordinary wide features follow their intended straight raster boundaries.
- Limit only new vertices created while splitting long longitude-latitude edges,
  across the full geometry, without rejecting already detailed boundaries.
- Keep boundaries curved when reprojecting. Vector features and raster
  footprints now gain points along every edge until the moved boundary is
  within a tenth of one population cell of the true curve. Moving only the
  corners undercounted an ordinary 40-degree box by about 11 percent, and threw
  the raster conservation check off by about 2.5 percent.
- Correct LandScan Global 2024 grid validation to accept the published global
  43,200 by 21,600-cell raster.

### Added

- Add a CI check for the declared minimum runtime dependency versions. It uses
  uv's lowest-direct resolution on Python 3.11 while resolving transitive
  dependencies normally.
- Add `allow_reprojection=True` to opt in to automatic reprojection for vector
  and raster hazards, and `allow_partial_coverage=True` to opt in to partial
  vector results with `population_coverage_fraction` and
  `population_coverage_complete` columns.
- Add the public `CrsMismatchError` and `PartialCoverageError` exceptions, both
  subclasses of `ValueError`.
- Report `population_conservation_relative_difference` and
  `population_reprojected` on raster results, and `reprojected` and
  `partial_coverage_allowed` on vector results.
- Add an offline GPWv4 Revision 11 same-product cross-resolution consistency
  fixture that matches shuffled native cells and separately published
  one-degree totals from the same product.
- Add a maintainer-only, offline UNOSAT FL20221125COD real-data method
  comparison fixture and an explicit source-checked regeneration script.
- Add a single-page Sphinx documentation site, published at
  https://palewi.re/docs/population-exposure/.
- Add `assign_population()` for joining population values to tabular hazard rows
  by exact cell keys.
- Extend `assign_population()` to GeoDataFrames, common vector paths, GeoTIFFs,
  and Rasterio readers using coverage-aware vector sums and count-preserving
  raster alignment.
- Add a lazy, window-readable `RasterAssignment` result for paired hazard and
  aligned-population cells.
- Add a curated `populations` catalog with structural checks for explicit WorldPop,
  GHSL, GPWv4, Chambers, and LandScan source/year selections, safe downloads and
  registration, platform caching, offline use, receipts, and assignment attrs.
- Add validation, examples, plain-language documentation, property tests, and
  package build checks for population assignment.
- Add opt-in live provider download coverage outside pull-request CI, with
  monthly anonymous WorldPop and GHSL checks, manual GPW and Chambers options,
  and documented Earthdata and LandScan limits.
- Compare the manual authenticated GPW fine population-count raster against
  CIESIN's official one-degree count grid for the same catalog year.
- Base GPW parity tolerances on each official coarse float32 count's
  representable precision, while retaining low-value and aggregate checks.
- Add an opt-in USGS PAGER Ridgecrest conditional published-result
  reproduction using the exact public hazard grid and a caller-supplied
  licensed LandScan 2017 file.

### Fixed

- Correct the catalogued grid bounds and nodata values for the official
  WorldPop 1 km mosaics by selected year.

[Unreleased]: https://github.com/palewire/population-exposure/compare/v0.1.0rc1...HEAD
[0.1.0rc1]: https://github.com/palewire/population-exposure/tree/v0.1.0rc1
