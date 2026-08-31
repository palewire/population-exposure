# GHSL Aruba tabular golden fixture

This is a small, offline test fixture for `assign_population` on exact
longitude/latitude cell keys. It contains 216 GHS-SMOD 2020 and GHS-POP 2020
30-arc-second cells whose centers are inside Aruba according to GADM 4.1.
`cells.csv` is in source-raster row order; the test reverses the population
table before joining so it also pins that the output keeps hazard-row order.

GHS-SMOD is a settlement classification, not a hazard. The test uses its
`smod_class` column solely to test the tabular cell-join method:

| DEGURBA L1 | GHS-SMOD L2 code |
| --- | --- |
| UC | 30 |
| UCL | 21, 22, 23 |
| RUR | 11, 12, 13 |

The global source also has zero-valued class `0` cells. The regeneration
script verifies they have no population and excludes them alongside water
(`10`) and NoData (`-200`).

The European Commission JRC's GHS-COUNTRY-STATS `POP_L1` workbook records
Aruba's 2020 population as UC 56,903.19754755, UCL 45,177.75497597, and RUR
4,504.04741600. A direct sum of the global WGS84 SMOD cells instead gives UC
54,699.13036945, UCL 43,746.81865901, and RUR 6,527.74272612. The test records
these exact differences rather than claiming they should be equal.

This is expected: the Country Statistics methodology says that it creates
SMOD clusters separately within GADM 4.1 country borders before applying
population thresholds. Therefore its workbook cannot be recreated by
overlaying a country boundary on the published global 30-arc-second SMOD
grid. The fixture contains no GADM geometry; GADM is downloaded only by the
regeneration script to make the cell selection explicit.

The cells are a small, attributed derivative of the European Commission JRC
GHSL products. `metadata.json` pins the official URLs, byte sizes, SHA-256
hashes, methodology citations, source-grid details, source totals, and the
official workbook values. The source report authorizes reuse with attribution.

Regenerate after deliberately accepting the approximately 520 MB cold-cache
download:

```sh
uv run --group test python scripts/regenerate_ghsl_tabular_golden.py \
  --accept-download \
  tests/data/ghsl_aruba_tabular
```

The script uses the shared platform cache, validates existing downloads before
reuse, deletes a corrupt named cache entry before retrying, requires exact
response sizes and SHA-256 values, and streams only named safe ZIP members.
