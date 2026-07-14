# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Configuration for generating readthedocs docstrings."""
from __future__ import absolute_import

from importlib.metadata import version as _get_distribution_version
from datetime import datetime
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from doc_utils.jumpstart_doc_utils import create_jumpstart_model_table  # noqa: E402

project = "sagemaker"
version = _get_distribution_version(project)

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
]

# Add any paths that contain templates here, relative to this directory.
# The _templates/layout.html override injects a <meta name="robots" content="noindex">
# tag into every page's <head> so search engines drop the deprecated V2 docs from
# their results (search de-indexing lever).
templates_path = ["_templates"]

# Canonical URL. Sphinx emits <link rel="canonical"> on each page using this base URL,
# declaring the V3 ("stable") page as the authoritative URL for each V2 page so search
# engines and automation consolidate signals on V3. A page built at
# /en/v2/<path>.html points its canonical at https://sagemaker.readthedocs.io/en/stable/<path>.html
# (same relative path under the V3 tree). Only accurate where a same-path V3 page exists.
html_baseurl = "https://sagemaker.readthedocs.io/en/stable/"

source_suffix = ".rst"  # The suffix of source filenames.
master_doc = "index"  # The master toctree document.

copyright = "%s, Amazon" % datetime.now().year

# The full version, including alpha/beta/rc tags.
release = version

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_trees = ["_build"]

pygments_style = "default"

autoclass_content = "both"
autodoc_default_flags = ["show-inheritance", "members", "undoc-members"]
autodoc_member_order = "bysource"

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 6,
    "includehidden": True,
    "titles_only": False,
}


html_static_path = ["_static"]

htmlhelp_basename = "%sdoc" % project

# For Adobe Analytics
html_js_files = [
    "https://a0.awsstatic.com/s_code/js/3.0/awshome_s_code.js",
    "https://cdn.datatables.net/1.10.23/js/jquery.dataTables.min.js",
    "https://kit.fontawesome.com/a076d05399.js",
    "js/datatable.js",
]

html_css_files = [
    "https://cdn.datatables.net/1.10.23/css/jquery.dataTables.min.css",
    "theme_overrides.css",
    "pagination.css",
    "search_accessories.css",
]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"python": ("http://docs.python.org/", None)}

# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# autosummary
autosummary_generate = True

# autosectionlabel
autosectionlabel_prefix_document = True

autodoc_mock_imports = ["pyspark", "feature_store_pyspark", "py4j"]


def setup(app):
    sys.stdout.write("Generating JumpStart model table...")
    sys.stdout.flush()
    create_jumpstart_model_table()
