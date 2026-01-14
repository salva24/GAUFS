import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'GAUFS'
author = 'Salvador de la Torre Gonzalez'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',  
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',      
    'sphinx.ext.viewcode',      
]

autosummary_generate = True
autodoc_typehints = 'description'

templates_path = ['_templates']
exclude_patterns = []

suppress_warnings = ['autosummary.import']

html_theme = 'alabaster'
html_static_path = ['_static']
