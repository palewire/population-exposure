# China 2019 heatwave golden validation

This maintainer-only validation tests the national result reported by Cai et
al.: **2.20 billion additional heatwave person-days among people aged 65 and
older in mainland China in 2019**, or **13 additional days per older person**,
relative to the 1986-2005 annual baseline. The golden metadata preserves the
numeric comparison even when a method-faithful regeneration does not match the
published value.

It is deliberately outside the user documentation. The committed
`tests/fixtures/china_heatwave_2019/cells.csv` is a small, reviewable
derivative containing only the 0.5-degree China cells needed to exercise
`population_exposure.assign_population`. The normal test suite reads it
without a network request. No large source data is committed.
A passing offline test means package assignment still reproduces the recorded
cell aggregation; source acquisition and climate processing are rechecked only
when a maintainer runs the opt-in regeneration.

## Primary method

The peer-reviewed China-report appendix is the controlling source:

- Cai W, Zhang C, Suen HP, et al. *The 2020 China report of the Lancet
  Countdown on health and climate change*. **Lancet Public Health**.
  2021;6:e131-e147. DOI
  [`10.1016/S2468-2667(20)30256-5`](https://doi.org/10.1016/S2468-2667(20)30256-5);
  [PMC7966675](https://pmc.ncbi.nlm.nih.gov/articles/PMC7966675/).
- Its [supplementary appendix 2](https://pmc.ncbi.nlm.nih.gov/articles/instance/7966675/bin/mmc2.pdf),
  pp. 3-5, states that the calculation uses ERA5
  and the Chambers (2020) hybrid population data. It defines the warm season as
  **May 1 through September 30**, not June-August. For each grid, the threshold
  is the 92.5th percentile of daily maximum temperature over those warm-season
  days in 1986-2005. A heatwave is at least three consecutive days above the
  threshold, and every day in a qualifying event is counted. It then defines
  annual cell exposure as heatwave days multiplied by that year's population,
  averages those exposures over 1986-2005, and subtracts that baseline from
  2019 exposure.
- The corresponding [2020 global report](https://pmc.ncbi.nlm.nih.gov/articles/PMC7616803/)
  appendix, pp. 6-8, documents the
  0.5-degree grid, ERA5 hourly inputs, person-days as heatwave days multiplied
  by people affected. It describes a different **exposure to change** baseline
  treatment. Where the reports conflict, this validation follows the China
  appendix's direct instruction to average annual heatwave exposure.

The calculation implemented here is therefore:

```text
exposure_year[cell] =
    heatwave_days_year[cell] * population_65_plus_year[cell]

additional_person_days =
    sum(exposure_2019[cell])
    - mean(
        sum(exposure_1986[cell]),
        ...,
        sum(exposure_2005[cell]),
      )
```

## Reproduction finding

The authoritative CDS run recorded in `golden.json` does **not** reproduce the
published additional-exposure result:

- method-defined additional exposure: **1,687,404,917.62 person-days**,
  **512,595,082.38 below** 2.20 billion;
- method-defined additional days per older person: **9.6152**, not 13;
- absolute 2019 exposure before subtracting the baseline:
  **2,272,963,932.94 person-days**, or **12.9519 days per older person**;
- mean annual 1986-2005 exposure subtracted by the appendix method:
  **585,559,015.32 person-days**.

The published pair of 2.20 billion and 13 days per person is much closer to the
reproduced **absolute** 2019 exposure than to the appendix-defined change, but
even the absolute total is 72,963,932.94 person-days outside the paper's
0.01-billion rounding interval. The paper's second time-series check also does
not reconcile the methods: it reports 71.8 million additional person-days in
2000, while this run produces 187.13 million using the appendix subtraction and
132.13 million when the same year's population weights the heatwave-day
anomaly.

Changing the threshold comparison to the appendix table's `>=` interpretation
produces 1,687,238,841.01 additional person-days, only 166,076.62 below the
strict result, so ties do not explain the discrepancy. The exact original 2020
mask, analysis code, percentile interpolation, and daily time zone remain
unpublished. This validation preserves the mismatch as a finding rather than
changing the stated method or broadening tolerance to make the result pass.

The original Chambers record is
[`10.5281/zenodo.3768003`](https://doi.org/10.5281/zenodo.3768003), file
`demographics_1950_2020.nc`. Its final `age_band_lower_bound=65` coordinate is
already a cumulative 65-and-older band; it must not be treated as a five-year
65-69 band. The later 0.25-degree Zenodo record `6011021` has a different grid
and age layout and is not substituted for the source cited by the paper.

The national mask uses UN M49 code 156 from the
[GPWv4 Revision 11 National Identifier Grid](https://doi.org/10.7927/H4TD9VDP)
at 30 arc-minutes. This excludes separately coded Hong Kong (344), Taiwan
(158), and Macao (446) from the national total. That choice is consistent with
the reported 13-day national denominator; the report still presents Hong Kong
and Taiwan separately in provincial results.

The China appendix does not identify its boundary file. The GPW identifier
grid is the traceable replacement because the global method says that the
Chambers cells were matched to countries with that grid. SEDAC explicitly
describes the identifier grid as input-data coverage rather than an official
country boundary. A later study from the same China team reports 3,829
mainland 0.5-degree cells and says its administrative boundaries came from
China's National Geographic Information Public Service Platform, but its data
and code are available only on request:
[Chen et al. (2022), PMC9465423](https://pmc.ncbi.nlm.nih.gov/articles/PMC9465423/).
The missing original 2020 mask is therefore recorded as a remaining source of
uncertainty, not treated as an exact input. Regeneration records both the raw
replacement-mask cell count and the smaller count with finite Chambers
population. It also requires the replacement count to remain within 5% of the
independently reported 3,829-cell mainland grid.

## Recorded assumptions

The paper and appendix do not publish the original analysis code, name the
China boundary file, state the percentile interpolation rule, or state the
daily time zone. `golden.json` records the reproducible choices made here:

- historical ERA5 values are requested from the official Copernicus Climate
  Data Store daily-statistics service and sampled at the exact 0.5-degree
  Chambers coordinates;
- daily maxima use 00:00-23:00 UTC and all 24 hourly values, matching xarray
  resampling in later public Lancet Countdown heatwave code;
- the percentile uses NumPy's documented linear interpolation;
- qualifying runs reset at each May-September season boundary;
- threshold comparison is strict `>`, matching the methods sentence ("higher
  than") and later public Lancet Countdown code. The appendix's justification
  paragraph also uses `>=`; ties at a calculated percentile should be reviewed
  if source encoding changes. Regeneration records the complete `>=`
  sensitivity result beside the controlling strict result.

The later implementation evidence is
[`zeliest/lancet_countdown_heatwave_2024`](https://github.com/zeliest/lancet_countdown_heatwave_2024)
at commit `4fe60956a690d7b32de61d04774357d637f922a1`. Its ERA5 preparation
resamples hourly UTC values to daily maxima; `source/heatwave_indices.py`
compares values with `>` and adds the full run length. Its later exposure
notebook uses the global report's different baseline treatment, which is not
used here.

These source and method differences are not hidden by a loose threshold. The
person-day check uses only the publication's stated precision: 2.20 billion is
an interval of plus or minus 0.005 billion. The per-person check similarly uses
half a day around the reported whole number.

## Regeneration

Regeneration reads about 1.66 GB from Zenodo and about 250 MB of regional daily
statistics from the Copernicus Climate Data Store on a cold cache. The CDS
requests are split into resumable annual batches. Around 3 GB of free space is
recommended. CDS queue time dominates: an annual batch typically processes in
about 4-6 minutes once started, but can wait much longer. This regeneration's
concurrently submitted annual requests took about 5 hours 20 minutes to clear
the CDS queue; the final fully cached command completed in 9.8 seconds. The
committed serial command can therefore take several hours on a cold cache.

Configure `~/.cdsapirc` for ERA5 access and set `EARTHDATA_TOKEN` for the small
SEDAC country-mask download, then run:

```sh
make regenerate-china-heatwave
```

The command uses the locked `validation` dependency group. It verifies source
sizes and digests and writes candidates under the cache first. It records
whether each numeric comparison is inside the publication's stated precision;
an outside result remains a visible finding rather than being hidden by a
broader tolerance.

The companion repository
[`palewire/cee-agriculture-climate-analysis`](https://github.com/palewire/cee-agriculture-climate-analysis)
currently contains only the shared Python project template and no relevant
climate-data extraction convention, so this workflow does not depend on it.
