# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
from unittest.mock import MagicMock


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        if name == "__version__":
            return "1.4.0"
        else:
            return MagicMock()


MOCK_MODULES = ['tensorflow', 'tensorflow.core', 'tensorflow.core.framework', 'tensorflow.python',
                'tensorflow.python.framework', 'tensorflow_serving', 'tensorflow_serving.apis']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

version = '1.1.0'
project = u'sagemaker'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest',
              'sphinx.ext.intersphinx', 'sphinx.ext.todo',
              'sphinx.ext.coverage', 'sphinx.ext.autosummary',
              'sphinx.ext.napoleon']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffix = '.rst'  # The suffix of source filenames.
master_doc = 'index'  # The master toctree document.

copyright = u'%s, Amazon' % datetime.now().year

# The full version, including alpha/beta/rc tags.
release = version

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_trees = ['_build']

pygments_style = 'default'

autoclass_content = "both"
autodoc_default_flags = ['show-inheritance', 'members', 'undoc-members']
autodoc_member_order = 'bysource'

if 'READTHEDOCS' in os.environ:
    html_theme = 'default'
else:
    html_theme = 'haiku'
html_static_path = ['_static']
htmlhelp_basename = '%sdoc' % project

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'http://docs.python.org/': None}

# autosummary
autosummary_generate = True
