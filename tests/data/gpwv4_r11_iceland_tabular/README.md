# GPWv4 Revision 11 Iceland 2020 tabular cross-resolution

This is a network-free, real-data **tabular assignment** golden fixture. It is
not a hazard or exposure result. It verifies that
`population_exposure.assign_population` matches native GPW Population Count
values to exact coordinate rows, preserves hazard-row ordering, and reproduces
independently published one-degree CIESIN values after grouping by
`parent_1_degree_cell`.

The fixture contains a 2 by 2 block of complete one-degree cells around
Iceland (global rows 25--26, columns 157--158) and its matching 240 by 240
30-arc-second crop. Its coordinate table contains the crop's finite cells;
the source nodata cells remain in the raster. Both are lossless float32
GeoTIFF derivatives of GPWv4 Revision 11 Population Count 2020 and are
distributed under CC BY 4.0.

The 30-arc-second input is https://data.earthdata.nasa.gov/nasa-earth/human-dimensions/sedac-root/downloads/data/gpw-v4/gpw-v4-population-count-rev11/gpw-v4-population-count-rev11_2020_30_sec_tif.zip. The independent published one-degree
oracle is https://data.earthdata.nasa.gov/nasa-earth/human-dimensions/sedac-root/downloads/data/gpw-v4/gpw-v4-population-count-rev11/gpw-v4-population-count-rev11_2020_1_deg_tif.zip. Both archives need an Earthdata token only during
regeneration. GPW documentation describes the coarse Population Count grids
as aggregations of the native cells. `The shared downloader uses a 60-second response timeout.`

Regenerate only after deliberately accepting the authenticated 405 MB fine
archive:

```sh
EARTHDATA_TOKEN=... uv run python scripts/regenerate_gpwv4_tabular_golden.py \
  --accept-download tests/data/gpwv4_r11_iceland_tabular
```

The script validates cache size and SHA-256 before reuse, deletes a corrupted
cached archive before replacing it, bounds transfers, uses the package's
credential-safe downloader, and uses the shared archive extractor that rejects
unsafe member paths. `metadata.json` pins the source files, crop checksums,
grid facts, published coarse values, and the float32 precision bounds.
