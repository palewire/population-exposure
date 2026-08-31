# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

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

- Return `0.0` for vector areas that are spatially covered but contain only
  population no-data cells. Raster hazards entirely outside the population
  raster now raise `PartialCoverageError`; overlapping no-data and zero-count
  areas remain valid.
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

- Add `allow_reprojection=True` to opt in to automatic reprojection for vector
  and raster hazards, and `allow_partial_coverage=True` to opt in to partial
  vector results with `population_coverage_fraction` and
  `population_coverage_complete` columns.
- Add the public `CrsMismatchError` and `PartialCoverageError` exceptions, both
  subclasses of `ValueError`.
- Report `population_conservation_relative_difference` and
  `population_reprojected` on raster results, and `reprojected` and
  `partial_coverage_allowed` on vector results.
- Add an offline GPWv4 Revision 11 cross-resolution tabular golden fixture that
  matches shuffled native cells and independently published one-degree totals.
- Add a maintainer-only, offline UNOSAT FL20221125COD vector golden fixture
  and an explicit source-checked regeneration script.
- Add a single-page Sphinx documentation site, published at
  https://palewi.re/docs/population-exposure/.
- Add `assign_population()` for joining population values to tabular hazard rows
  by exact cell keys.
- Extend `assign_population()` to GeoDataFrames, common vector paths, GeoTIFFs,
  and Rasterio readers using coverage-aware vector sums and count-preserving
  raster alignment.
- Add a lazy, window-readable `RasterAssignment` result for paired hazard and
  aligned-population cells.
- Add a verified `populations` catalog for explicit WorldPop, GHSL, GPWv4,
  Chambers, and LandScan source/year selections, safe downloads and
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
- Add an opt-in USGS PAGER Ridgecrest raster golden reproduction using the
  exact public hazard grid and a caller-supplied licensed LandScan 2017 file.

### Fixed

- Correct the catalogued grid bounds and nodata values for the official
  WorldPop 1 km mosaics by selected year.

[Unreleased]: https://github.com/palewire/population-exposure/commits/main
