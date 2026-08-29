# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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
