import os
import sys

sys.path.insert(0, os.path.abspath("../../humancompatible"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "HumanCompatible.Detect"
copyright = "2025, AutoFair Project"
author = "AutoFair Project"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # important for numpy/google style docstrings...
    "sphinx.ext.todo",  # Good for TODO notes (optional)
    "sphinx.ext.viewcode",  # Adds links to source code (optional, highly recommended)
    "myst_parser",
]

# Configure napoleon to enable both NumPy and Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True  # Include __init__ docstrings
# Default False, change if you want private members
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True  # Include __str__, __repr__ etc.
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
# Set to True if you want type hints to be processed by Napoleon
napoleon_preprocess_types = False
napoleon_type_aliases = None

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_theme = "alabaster"
html_static_path = ["_static"]
