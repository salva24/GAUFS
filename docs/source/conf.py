# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------

project = 'GAUFS'
copyright = '2026, Salvador de la Torre Gonzalez'
author = 'Salvador de la Torre Gonzalez'
release = '0.3.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
]

autosummary_generate = True
autosummary_imported_members = False

autodoc_typehints = 'description'

templates_path = ['_templates']
exclude_patterns = []

# -- Napoleon configuration (NumPy style docstrings) -------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True

napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = False

napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False

napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

napoleon_attr_annotations = True

# -- Autodoc configuration ---------------------------------------------------

autodoc_default_options = {
    'members': False,
    'undoc-members': False,
    'special-members': None,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
    'inherited-members': False,
}

# -- Suppress non-critical warnings ------------------------------------------

suppress_warnings = [
    'autosummary.import',
    'autosummary.stub',
    'toc.not_included',
]

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']
