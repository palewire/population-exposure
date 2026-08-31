# USGS PAGER Ridgecrest fixture

This is a real-data **conditional published-result reproduction** fixture.
`mmi.tif` is a deterministic GeoTIFF derivative of the public USGS PAGER
Ridgecrest `grid.xml` artifact. It checks that this package reads the published
hazard grid with its documented grid transformation. It does not reproduce a
PAGER result by itself, validate PAGER's methods, or provide LandScan data.

The opt-in live test compares PAGER's published exposure bands only when the
caller supplies a licensed LandScan Global 2017 GeoTIFF. The source record does
not expose the annual LandScan release used by PAGER, so 2017 is a documented
inference rather than a verified PAGER input. Its half-person tolerance is for
numeric rounding only; it is not widened for source differences.

USGS states that [U.S. government information is public domain unless otherwise
noted](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits).
The source URL, SHA-256 digest, USGS attribution, and exact conversion from XML
coordinates to the GeoTIFF are recorded in `metadata.json`. Reuse of the
fixture must retain that attribution. The repository's MIT license does not
license LandScan, and no LandScan values are included here.
