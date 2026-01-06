# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------

project = 'GAUFS'
copyright = '2026, Salvador de la Torre Gonzalez'
author = 'Salvador de la Torre Gonzalez'
release = '0.3.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosummary',
]

autosummary_generate = True
autodoc_typehints = 'description'

templates_path = ['_templates']
exclude_patterns = []

# Use Google style docstrings by default (Napoleon)
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Show type hints in descriptions
autodoc_typehints = 'description'

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']

suppress_warnings = ['app.add_object', 'toc.not_included']