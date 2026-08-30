# UNOSAT FL20221125COD Basankusu vector golden fixture

This is maintainer test data, not a public how-to. It holds one clipped,
repaired 2022 VIIRS maximum surface-water polygon and the 166 x 163-cell
WorldPop 2020 crop needed to test `assign_population`.

The official UNOSAT workbook gives Basankusu's 2022 water area as 393.7350 km2
and exposed population as 9,570.69032327. The source polygon is a single
invalid multipolygon with a self-intersecting ring; regeneration applies
`shapely.make_valid` before clipping it to OCHA COD-AB v01 district `CD4107`.
The fixture's equal-area measurement is 393.7350537448322 km2.

The registry selection `worldpop-global-1km:2020` resolves to the exact
WorldPop URL in `metadata.json`. UNOSAT identifies its population input as
"WorldPop unconstrained UN adjusted [2020]"; the WorldPop listing calls the
same file an unconstrained 1 km population-count mosaic and does not state its
UN-adjustment label. The source file name, CRS, grid, and URL match the
registry, but the published population is not exactly reproducible:

| Method | People | Difference from UNOSAT |
| --- | ---: | ---: |
| UNOSAT workbook | 9,570.69032327 | 0 |
| `exactextract` fractional coverage | 9,693.94675977 | +123.25643650 |
| Cell-center reference | 9,075.67927194 | -495.01105133 |

The reference uses `rasterio.features.geometry_mask`; it is not an ArcGIS
run. The workbook does not document the ArcGIS zonal-statistics edge rule, so
the two measured differences are recorded rather than tuned away. The
full-district totals also differ by 0.2-0.3%, which supports treating the
fixture as a closest defensible reproduction rather than a claim of identity.
The full WorldPop source produces 9,693.94676161; the crop differs by
0.00000184 people, and that measured floating-point difference is also pinned.

Regenerate only after deliberately accepting the roughly 910 MB download:

```sh
uv run python scripts/regenerate_unosat_vector_golden.py \
  --accept-download /tmp/unosat_fl20221125cod_basankusu
```

The script rejects changed source bytes and writes new checksums and measured
results to `metadata.json`. Source URLs, SHA-256 values, product identities,
and layer/date details are all pinned there.
