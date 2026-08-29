# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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

[Unreleased]: https://github.com/palewire/population-exposure/commits/main
