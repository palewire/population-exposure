"""Configuration for the Sphinx documentation builder."""

from datetime import UTC, datetime
from importlib.metadata import metadata
from importlib.metadata import version as distribution_version

distribution = metadata("population-exposure")
project = distribution["Name"]
author = distribution.get("Author") or distribution.get("Author-email", "")
version = distribution_version(project)
release = version
year = datetime.now(UTC).year
copyright = f"{year}, {author}"

language = "en"
templates_path = ["_templates"]
html_static_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = True

nitpicky = True
nitpick_ignore = [
    ("py:class", "Affine"),
    ("py:class", "BoundingBox"),
    ("py:class", "CRS"),
    ("py:class", "DatasetReader"),
    ("py:class", "Path"),
    ("py:class", "PathLike"),
    ("py:class", "RasterSource"),
    ("py:class", "Window"),
    ("py:class", "gpd.GeoDataFrame"),
    ("py:class", "np.ma.MaskedArray"),
    ("py:class", "pd.DataFrame"),
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/stable/", None),
}

linkcheck_timeout = 10
linkcheck_retries = 2
# NASA SEDAC's authoritative GPW page times out under automated link checking.
linkcheck_ignore = [
    r"https://sedac\.ciesin\.columbia\.edu/data/set/gpw-v4-population-count-rev11"
]

html_theme = "palewire"
html_baseurl = "https://palewi.re/docs/population-exposure/"
html_sidebars = {"**": []}
html_theme_options = {"canonical_url": html_baseurl}
