import os
import sys
from datetime import datetime

# Add the source directories to Python path
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../sagemaker-core/src'))
sys.path.insert(0, os.path.abspath('../sagemaker-train/src'))
sys.path.insert(0, os.path.abspath('../sagemaker-serve/src'))
sys.path.insert(0, os.path.abspath('../sagemaker-mlops/src'))

project = 'SageMaker Python SDK V3'
copyright = f'{datetime.now().year}, Amazon Web Services'
author = 'Amazon Web Services'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_nb',
    'sphinx_book_theme',
    'sphinx_design',
    'sphinx_copybutton'
]

templates_path = ['_templates']
exclude_patterns = [
    '_build', 
    'Thumbs.db', 
    '.DS_Store', 
    'sagemaker-core/docs/*', 
    'sagemaker-core/CHANGELOG.md', 
    'sagemaker-core/CONTRIBUTING.md',
]

# Suppress specific warnings
suppress_warnings = [
    'myst.header',       # header level warnings from notebooks
    'toc.not_readable',  # toctree warnings for symlinked files
    'ref.python',        # "more than one target found" for duplicate class names
    'autosummary',       # autosummary import failures for internal modules
]

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'repository_url': 'https://github.com/aws/sagemaker-python-sdk',
    'use_repository_button': True,
    'use_issues_button': True,
    'use_edit_page_button': False,
    'path_to_docs': 'docs/',
    'show_navbar_depth': 2,
    'show_toc_level': 2,
    'collapse_navbar': True,
    'announcement': 'This is V3 documentation. <a href="https://sagemaker.readthedocs.io/en/v2/">View V2 docs</a>',
}

html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['feedback.js']

html_context = {
    'display_github': True,
    'github_user': 'aws',
    'github_repo': 'sagemaker-python-sdk',
    'github_version': 'master',
    'conf_py_path': '/docs/',
    'version_warning': True,
    'version_warning_text': 'This is the V3 documentation. For V2 documentation, visit the legacy docs.',
}

nb_execution_mode = 'off'
nb_execution_allow_errors = True

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'private-members': False,
}

# Generate autosummary stubs recursively
autosummary_generate = True

# Suppress internal/implementation modules not intended for users
exclude_patterns += [
    '*/telemetry*',
    '*/tools*',
    '*/container_drivers*',
    '*/runtime_environment*',
    '*/model_server*',
    '*/detector*',
    '*/validations*',
    '*/image_retriever*',
]

# Modules that fail to import due to runtime dependencies or side effects
autodoc_mock_imports = [
    'triton_python_backend_utils',
    'sagemaker.serve.model_server.in_process_model_server.app',
    'sagemaker.serve.model_server.multi_model_server.inference',
    'sagemaker.serve.model_server.tensorflow_serving.inference',
    'sagemaker.serve.model_server.torchserve.inference',
    'sagemaker.serve.model_server.torchserve.xgboost_inference',
    'sagemaker.serve.model_server.triton.model',
]

suppress_warnings = ['autodoc.import_error']
