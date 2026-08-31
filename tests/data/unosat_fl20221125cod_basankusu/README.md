# UNOSAT FL20221125COD Basankusu vector golden fixture

This is maintainer test data, not a public how-to. It is a real-data **method
comparison** fixture: one clipped, repaired 2022 VIIRS maximum surface-water
polygon and the 166 x 163-cell WorldPop 2020 crop needed to test
`assign_population`. It pins this package's coverage-weighted ExactExtract
wrapper result and a transparent cell-center reference. It does not validate
UNOSAT's unpublished processing choices or identify UNOSAT's population file.

The official UNOSAT workbook gives Basankusu's 2022 water area as 393.7350 km2
and exposed population as 9,570.69032327. The source polygon is a single
invalid multipolygon with a self-intersecting ring; regeneration applies
`shapely.make_valid` before clipping it to OCHA COD-AB v01 district `CD4107`.
The fixture's equal-area measurement is 393.7350537448322 km2.

The registry selection `worldpop-global-1km:2020` resolves to the WorldPop
Global 1 km mosaic URL in `metadata.json`. UNOSAT identifies its population
input only as "WorldPop unconstrained UN adjusted [2020]" and supplies no URL,
version, or hash. The registry raster is therefore a plausible comparator, not
a demonstrated source match. The measured disagreement is retained:

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
The full registry WorldPop raster produces 9,693.94676161; the crop differs by
0.00000184 people, and that measured floating-point difference is also pinned.

## Attribution and reuse

`population.tif` is a cropped derivative of WorldPop's 2020 unconstrained
global mosaic. WorldPop licenses its datasets under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); retain the
attribution and citation in `metadata.json` when reusing it.

`hazard.geojson` is a derived test-only geometry, made by repairing the
UNOSAT VIIRS water polygon and clipping it to the OCHA COD-AB boundary. The
OCHA source record licenses the boundary under
[CC BY-IGO 3.0](https://creativecommons.org/licenses/by/3.0/igo/legalcode);
retain its attribution. The linked UNOSAT source artifacts state no specific
reuse license. Their redistribution status is unresolved, so this small
derived geometry is isolated to this fixture and must not be reused as
UNOSAT-licensed data. The repository's MIT license does not grant rights to
either fixture artifact.

Regenerate only after deliberately accepting the roughly 910 MB download:

```sh
uv run python scripts/regenerate_unosat_vector_golden.py \
  --accept-download \
  /tmp/unosat_fl20221125cod_basankusu
```

The script rejects changed source bytes and writes new checksums and measured
results to `metadata.json`. Source URLs, SHA-256 values, citations,
transformations, reuse statements, and layer/date details are pinned there.
