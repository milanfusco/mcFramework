# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup ----------------------------------------------------------------
from __future__ import annotations
import os
import sys
from datetime import datetime

# -- Import the package to document ----------------------------------------------
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mcframework'
author = 'Milan Fusco'
copyright = f"{datetime.now():%Y}, {author}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Core
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    
    # Docs quality
    "sphinx_autodoc_typehints",
    "numpydoc",
    "myst_parser",
    "sphinx.ext.doctest",
    
    # UX polish
    "sphinx_copybutton",
    "sphinx_design",
    # "sphinx_gallery.gen_gallery",  # if/when we add a gallery
]

nitpicky = True
nitpick_ignore = [
    ("py:class", "MetricSet"),
    ("py:class", "mcframework.stats_engine._MetricT"),
    ("py:obj", "mcframework.stats_engine._MetricT"),
    ("py:func", "mcframework.stats_engine._clean"),
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_prev_next": False,
    "navigation_depth": 2,
    # "logo": {"text": "mcframework"}, # or set an SVG logo via html_logo
}
# Where to keep local assets (badges, images)
# html_static_path = ["_static"]

# -- Autodoc / Autosummary ------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration
autosummary_generate = True
autosummary_imported_members = True      # include re-exported symbols
autodoc_member_order = "bysource"        # keep source order for readability
autodoc_typehints = "description"        # types in the doc body, not signature
autoclass_content = "both"               # class + __init__ docstrings together
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "undoc-members": False,
}
autodoc_class_signature = "mixed"        # show class signature at the class

# -- Numpydoc / Napoleon --------------------------------------------------------
numpydoc_show_class_members = False      # let autosummary make member pages
numpydoc_attributes_as_param_list = True

# -- MyST (Markdown) ------------------------------------------------------------
myst_enable_extensions = ["dollarmath", "amsmath"]  # $...$ and $$...$$
myst_heading_anchors = 3


# -- MathJax / LaTeX macros -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/math.html#module-sphinx.ext.mathjax
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "preamble": r"""
            \usepackage{amsmath}
            \usepackage{amssymb}
            \newcommand{\olsi}[1]{\,\overline{\!{#1}}}
        """,
        "macros": {
            "E": r"\mathbb{E}",
            "Var": r"\mathrm{Var}",
            "SE": r"\mathrm{SE}",
            "xBar": r"\,\overline{\!{x}}",
            "Xbar": r"\,\overline{\!{X}}",
        },
    }
}
# (LaTeX PDF build options only matter if we add a latex builder)
latex_elements: dict[str, str] = {}

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Copybutton (skip prompts in code blocks) -----------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
