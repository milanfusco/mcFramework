# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup ----------------------------------------------------------------
from __future__ import annotations

import dataclasses
import os
import sys
from datetime import datetime

# -- Import the package to document ----------------------------------------------
sys.path.insert(0, os.path.abspath("../../src"))


# -- Autodoc event: skip dataclass fields (already in docstring Attributes) ------
_DATACLASS_FIELD_CACHE: dict[type, set[str]] = {}


def _get_dataclass_fields(cls: type) -> set[str]:
    """Get cached set of field names for a dataclass."""
    if cls not in _DATACLASS_FIELD_CACHE:
        if dataclasses.is_dataclass(cls):
            _DATACLASS_FIELD_CACHE[cls] = {f.name for f in dataclasses.fields(cls)}
        else:
            _DATACLASS_FIELD_CACHE[cls] = set()
    return _DATACLASS_FIELD_CACHE[cls]


def autodoc_skip_member_handler(app, what, name, obj, skip, options):
    """Skip dataclass fields since they're documented in the Attributes section."""
    if skip:
        return skip
    
    # Only process attributes
    if what != "attribute":
        return skip
    
    # Try to find the parent class
    # For class attributes, obj might be the descriptor or the value
    parent = getattr(obj, "__objclass__", None)
    
    # If we can't get parent directly, try from the fget of property
    if parent is None and hasattr(obj, "fget"):
        parent = getattr(obj.fget, "__objclass__", None)
    
    if parent is not None and name in _get_dataclass_fields(parent):
        return True  # Skip dataclass field
    
    return skip


def setup(app):
    """Connect event handlers."""
    app.connect("autodoc-skip-member", autodoc_skip_member_handler)

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
    # NOTE: sphinx_autodoc_typehints removed - conflicts with numpydoc on dataclasses
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
    # Module references (Sphinx doesn't always resolve these)
    ("py:mod", "mcframework"),
    ("py:mod", "mcframework.core"),
    ("py:mod", "mcframework.stats_engine"),
    ("py:mod", "mcframework.sims"),
    ("py:mod", "mcframework.utils"),
    # Metric is a Protocol, not a regular class
    ("py:class", "mcframework.stats_engine.Metric"),
    ("py:obj", "mcframework.stats_engine.Metric"),
]

# Ignore attribute/method references to dataclass fields (slots don't expose as attrs)
nitpick_ignore_regex = [
    # Full path references to dataclass attributes
    (r"py:attr", r"mcframework\.stats_engine\.StatsContext\..*"),
    (r"py:attr", r"mcframework\.stats_engine\.ComputeResult\..*"),
    (r"py:attr", r"mcframework\.core\.SimulationResult\..*"),
    # Short references (~ prefix strips module, leaves Class.attr)
    (r"py:attr", r"^StatsContext\.\w+$"),
    (r"py:attr", r"^ComputeResult\.\w+$"),
    (r"py:attr", r"^SimulationResult\.\w+$"),
    (r"py:attr", r"^(n_simulations|metrics|rng|ess|n|alpha)$"),
    (r"py:meth", r"^alpha$"),
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_prev_next": False,
    "navigation_depth": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/milanfusco/mcframework",
            "icon": "fa-brands fa-github",
        },
    ],
}
# Where to keep local assets (badges, images)
# html_static_path = ["_static"]

# -- Autodoc / Autosummary ------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration
autosummary_generate = True
autosummary_imported_members = True      # include re-exported symbols
autodoc_member_order = "bysource"        # keep source order for readability
autodoc_typehints = "signature"          # types only in signature, not body
autoclass_content = "class"              # only class docstring (avoids dataclass duplication)
autodoc_default_options = {
    "members": True,
    "inherited-members": False,          # don't show inherited for cleaner output
    "show-inheritance": True,
    "undoc-members": False,
}
autodoc_class_signature = "separated"    # signature separate from description

# -- Numpydoc ----------------------------------------------------------------
numpydoc_show_class_members = False      # let autosummary make member pages
numpydoc_class_members_toctree = False   # don't create separate toctree for members
numpydoc_xref_param_type = True          # create cross-references for types

# Words to NOT create cross-references for
numpydoc_xref_ignore = {
    # Common descriptor words
    "of", "or", "default", "optional", "keyword-only",
    # Shape descriptors  
    "shape", "length", "size",
    # Other common words that aren't types
    "mapping", "iterable", "sequence", "callable",
    "any", "scalar", "array-like", "array_like",
}

# Map short names to full cross-reference paths
numpydoc_xref_aliases = {
    # mcframework classes
    "Simulations": "mcframework.sims",
    "Utils": "mcframework.utils",
    "Core": "mcframework.core",
    "StatsEngineModule": "mcframework.stats_engine",
    "SimulationResult": "mcframework.core.SimulationResult",
    "MonteCarloSimulation": "mcframework.core.MonteCarloSimulation",
    "MonteCarloFramework": "mcframework.core.MonteCarloFramework",
    "StatsEngine": "mcframework.stats_engine.StatsEngine",
    "StatsContext": "mcframework.stats_engine.StatsContext",
    "FnMetric": "mcframework.stats_engine.FnMetric",
    "Metric": "mcframework.stats_engine.Metric",
    "ComputeResult": "mcframework.stats_engine.ComputeResult",
    "CIMethod": "mcframework.stats_engine.CIMethod",
    "NanPolicy": "mcframework.stats_engine.NanPolicy",
    "BootstrapMethod": "mcframework.stats_engine.BootstrapMethod",
    # NumPy types
    "ndarray": "numpy.ndarray",
    "Generator": "numpy.random.Generator",
    "SeedSequence": "numpy.random.SeedSequence",
    # Python builtins (via intersphinx)
    "int": ":py:class:`int`",
    "float": ":py:class:`float`",
    "bool": ":py:class:`bool`",
    "str": ":py:class:`str`",
    "dict": ":py:class:`dict`",
    "list": ":py:class:`list`",
    "tuple": ":py:class:`tuple`",
    "None": ":py:obj:`None`",
}

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
