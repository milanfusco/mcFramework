# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]
### Bug Fixes
- Fix CI workflow and benchmark script bugs
- Reduce simulation count to prevent OOM errors in profiling demo
- Add missing newline in demo.py for code style consistency

### Documentation
- Update README.md
- Enhance documentation for execution backends and GPU support
- Clarify Torch backend imports in __init__.py
- Update release-drafter.yml to remove pull_request triggers
- Add CUDA and MPS backends for Monte Carlo simulations
- Update dtype policy in Monte Carlo simulation for device-specific handling
- Added legacy directory for outdated docs

### Features
- Add cuRAND support for CUDA backend in benchmarking
- Add PyTorch profiler integration for performance analysis
- Enhance Monte Carlo simulation for GPU support
- Add CUDA performance benchmark demo
- Enhance package metadata and dependencies for GPU support
- Enhance Torch CUDA backend with adaptive batching and cuRAND support
- Introduce seperate Torch backends for GPU-accelerated Monte Carlo simulations
- Add Apple Silicon performance benchmark demo
- Enhance Monte Carlo simulation with Torch backend support
- Implement Torch backend support for Monte Carlo simulations
- Add Safety CLI scan workflow for vulnerability checks
- Add WIP Features section to README

### Maintenance
- Update CHANGELOG.md and add cliff.toml for changelog generation
- Enhance benchmarking calculations and visualizations
- Improve timing accuracy in Monte Carlo simulation
- Remove unnecessary blank lines in Torch backend test files
- Remove egg-info files for mcframework
- Add GPU support to CI workflow
- Refine Sphinx dependencies in pyproject.toml
- Update sphinx-autosummary-accessors dependency in documentation
- Add sphinx-autosummary-accessors dependency for documentation generation
- Remove obsolete generated API documentation files
- Remove obsolete documentation build artifacts
- Standardize simulation kwargs naming across Torch backends
- Clean up imports and remove unnecessary whitespace in simulation.py
- Remove duplicate dead code
- Streamline Torch backend integration in Monte Carlo simulation
- Reorganize Torch dependency in pyproject.toml
- Add .safety-project.ini and update dependencies in pyproject.toml
- Add torch dependency to pyproject.toml
- Extract backends module and evolve to backend= API
- **deps**: Bump actions/download-artifact from 5 to 7

### Performance
- Add MPS benchmark image

### Testing
- Introduce a simple simulation class for Windows compatibility
- Update CI workflow to run Torch tests for MPS and CPU backends
- Migrate Torch backend tests to separate files for better organization
- Adjust simulation count in Torch backend test for performance
- Enhance Torch backend tests for unused kwargs handling
- Update tests for Torch backend utilities and device validation
- Add comprehensive CUDA backend tests for performance and error handling
- Replace internal generator method with backend utility in tests
- Add deprecation tests for explicit backend usage in Monte Carlo simulation
- Enhance Torch backend tests and update simulation error handling
- Migrate tests to backend= API and add coverage

## [0.1.1] – 2026-01-13
### Bug Fixes
- Fix documentation link case sensitivity
- Fix formatting of links in README.md
- Fix documentation link for mcFramework
- Fix state comparison in RNG tests for Black-Scholes simulations
- Fix Sphinx documentation build command in CI workflow by removing unnecessary warning flag

### Documentation
- Revise project plan and table formatting
- Update SYSTEM_DESIGN.rst
- Update README
- Update documentation for clarity and structure
- Update CHANGELOG.md for initial public release of mcframework (0.1.0)
- Update Docs badge link in README.md
- Update README to improve documentation structure and add links
- Revise Monte Carlo framework class diagram
- Update CHANGELOG for version 0.5.0 and add UML diagrams for mcframework
- Update doctest configuration in conf.py to improve testing setup
- Update .gitignore to include additional generated documentation files and images
- Update documentation and improve API references
- Update .gitignore to include additional Sphinx build directories for generated API documentation
- Update .gitignore to exclude all build artifacts from documentation
- Update documentation and improve .gitignore for better project organization
- Update URL in README
- Revise README for improved description and clarity
- Update src/mcframework/stats_engine.py
- Update CI workflows for documentation deployment and validation
- Update CI workflow for coverage reporting and modify performance test assertions
- Update CI workflow to use latest GitHub actions versions for improved reliability
- Update README and CI workflows; enhance parallel execution handling
- Update .gitignore to exclude coverage files and remove obsolete coverage reports
- Update CI badges in QUICK_REFERENCE.md
- Update .gitignore and McFramework.iml to exclude IDE files

### Features
- Add installation instructions to getting-started.md
- Add badges to README for project status
- Enhance project metadata and documentation
- Enhance README with detailed documentation and installation instructions
- Add project plan and system design to documentation for McFramework
- Add project plan and system design documentation for McFramework
- Enhance documentation and examples for doctests in mcframework
- Enhance documentation and configuration for mcframework
- Enhance Black-Scholes GUI with statistical analysis features
- Enhance Black-Scholes GUI with option pricing features and layout improvements
- Enhance CandlestickChart with crosshair and tooltip features
- Enhance Black-Scholes GUI with new features and layout improvements
- Enhance Black-Scholes GUI with new features and UI improvements
- Add GUI application for Black-Scholes Monte Carlo simulations
- Add test to ensure RNG state is preserved in calculate_greeks
- Enhance Sphinx documentation configuration
- Enhance mcframework with Black-Scholes simulations and tests
- Add ticker-based Black-Scholes analysis demo and visualizations
- Add comprehensive tests for Black-Scholes sim
- Add Black-Scholes simulation demo and enhance demo files
- Enhance error handling and testing in core and stats engine
- Enhance stats engine with ComputeResult class and improve validation
- Enhance stats_engine.py with new CIResult class and update tests
- Add documentation deployment badge to README
- Enhance CI workflow and update documentation deployment
- Enhance statistical context and testing in stats_engine.py and test files
- Enhance test coverage and readability in test_core.py
- Enhance statistical functions and improve context handling in Monte Carlo simulations
- Add GitHub Actions workflows for CI/CD pipeline
- Enhance .gitignore and improve error handling in stats_engine
- Add CHANGELOG.md and CONTRIBUTING.md; update README.md with installation instructions and dependencies
- Add `eps` parameter and improve statistic engine handling
- Add StatsContext and enhance statistical metrics with new features

### Maintenance
- **deps**: Bump actions/upload-artifact from 5 to 6
- **deps**: Bump actions/stale from 9 to 10
- **deps**: Bump github/codeql-action from 3 to 4
- **deps**: Bump actions/checkout from 5 to 6
- Refactor links formatting in README
- Improve test files by adding missing newlines for consistency
- Refactor demo scripts and improve imports
- Refactor README by removing outdated sections
- Refactor SYSTEM_DESIGN.md for clarity and conciseness
- Revert "Adding a Neutron Transport simulation model and demo "
- Refactor Pylint configuration and enhance type annotations in mcframework
- Refactor Black-Scholes GUI components and enhance styling
- Refactor and reorganize simulation modules in mcframework
- **deps**: Bump actions/upload-artifact from 4 to 5
- Remove obsolete autosummary documentation for unused functions in mcframework.utils
- Refactor Monte Carlo simulation percentile computation and enhance tests
- Refactor imports in test files for improved clarity
- Refactor Monte Carlo simulation and stats engine for improved clarity and functionality
- Refactor Monte Carlo simulation and enhance stats handling
- Refactor CI workflows for documentation handling
- Remove unused import of the time module in test_performance_and_concurrency.py to improve code cleanliness.
- Remove unused import of the math module in test_core.py to enhance code cleanliness.
- Comment out performance test for parallel execution speed due to environment-dependent variance
- Delete .DS_Store
- Refactor StatsEngine context handling and enhance statistical functions
- Remove unnecessary blank line in test_integration.py for improved code cleanliness.
- Refactor imports and improve code readability
- Delete GITHUB_ACTIONS_SETUP_COMPLETE.md
- Rename LICENSE  to LICENSE
- Delete .idea directory
- Delete trace
- Delete build directory
- Refactor confidence interval handling and enhance percentile assertions in tests
- Initial commit
---

## Development History

<details>
<summary>Pre-release development milestones (click to expand)</summary>

The following sections document the incremental development history leading to version 0.1.0.

### dev-0.5.0 – PySide6 GUI Application

#### Added
- PySide6-based GUI application for interactive Black-Scholes Monte Carlo simulations
- Market data tab with live candlestick charts, crosshair, and tooltip features
- Option pricing calculator with Greeks visualization
- Monte Carlo simulation controls with real-time path visualization
- 3D option price surfaces for volatility and time sensitivity analysis
- Dark theme stylesheet for improved visual aesthetics
- Toast notification system for user feedback
- Empty state widgets for enhanced UX when no data is available
- Recent tickers functionality in sidebar for quick access
- `StatsConfig` class for encapsulating statistical settings
- Integration with `StatsEngine` for comprehensive statistical displays in GUI
- Optional GUI dependencies in `pyproject.toml` (`mcframework[gui]`)
- Comprehensive documentation for GUI usage and features

#### Changed
- Enhanced doctest configuration in `conf.py` for improved testing setup
- Updated examples in `ComputeResult`, `StatsEngine`, and utility functions to include `StatsContext`
- Refactored Pylint configuration with additional message controls and design constraints
- Enhanced type annotations in core simulation functions

#### Fixed
- Improved output formatting in documentation examples for consistency

### dev-0.4.0 – Black-Scholes Simulations & Module Reorganization

#### Added
- `BlackScholesSimulation` class for European option pricing via Monte Carlo
- `BlackScholesPathSimulation` class for path-dependent option analysis
- Greeks calculation methods (`delta`, `gamma`, `theta`, `vega`, `rho`)
- Ticker-based Black-Scholes analysis demo (`demoTickerBlackScholes.py`)
- Black-Scholes simulation demo with visualizations (`demoBlackScholes.py`)
- `TICKER_DEMO_README.md` with usage instructions
- Comprehensive test suite for Black-Scholes simulations (`test_black_scholes.py`)
- Test for RNG state preservation in `calculate_greeks` method
- Neutron transport simulation module (experimental, on feature branch)

#### Changed
- Reorganized simulation modules: distributed `sims.py` into dedicated modules
- `sims/black_scholes.py` - Black-Scholes simulations
- `sims/pi.py` - Pi estimation simulation
- `sims/portfolio.py` - Portfolio simulation
- Introduced new `sims/__init__.py` for streamlined imports
- Updated public API to reflect new module organization
- Enhanced Black-Scholes simulation classes to support ticker-based analysis

#### Fixed
- Fixed state comparison logic in RNG tests for Black-Scholes simulations
- Regression test for early exercise handling with matrix singularity

### dev-0.3.2 – Stats Engine Refactoring & Documentation

#### Added
- `ComputeResult` class to encapsulate results from stats engine (metrics, skipped metrics, errors)
- Cross-field validation for `ess` and `n_bootstrap` parameters in `StatsContext`
- Tests for `_compute_stats_with_engine` behavior when engine is `None`
- Tests for `ComputeResult` representation and error handling
- Edge case tests for `StatsContext` and statistical functions
- New "Getting Started" guide for user onboarding
- Autosummary templates for enhanced class documentation
- GitHub icon link in Sphinx HTML theme

#### Changed
- Enhanced `ci_mean` and `ci_mean_bootstrap` to return `_CIResult` dataclass
- Statistical functions now return `None` for empty inputs instead of zero
- Improved representation methods in `ComputeResult` and `StatsEngine`
- Expanded documentation with detailed descriptions of classes and functions
- Improved docstrings across simulation classes for consistency
- Refactored Monte Carlo simulation with dedicated methods for percentile and statistics handling
- Updated type hints to use new union syntax
- Enhanced parallel execution logic with clearer separation of thread/process handling

#### Removed
- Unused `_validate_ctx` function from codebase
- Obsolete autosummary documentation for unused functions

#### Fixed
- Improved handling of `KeyboardInterrupt` in `MonteCarloSimulation` for proper future cancellation
- Error handling in `MonteCarloSimulation._compute_percentiles_block`

### dev-0.3.1 – CI/CD & Documentation Fixes

#### Added
- Documentation deployment badge in `README.md`
- Separate workflows for documentation deployment and validation
- Concurrency settings for deployment process

#### Changed
- Upgraded `upload-pages-artifact` action to version 4
- Modified branches to include `test` for documentation validation workflow
- Refactored CI workflows for documentation handling

#### Fixed
- Fixed Sphinx documentation build command in CI workflow by removing unnecessary treat-warnings-as-errors flag
- Commented out environment-dependent performance test for parallel execution speed (soft test)

#### Removed
- `GITHUB_ACTIONS_SETUP_COMPLETE.md` (merged into workflow documentation)

### dev-0.3.0 – CI/CD and Documentation Overhaul

#### Added
- GitHub Actions CI/CD pipeline with linting, testing, and documentation building
- Documentation deployment workflow (automated GitHub Pages publishing)
- Documentation validation workflow
- Dependabot configuration for automated dependency updates
- CI status badges in `README.md`
- GitHub Actions documentation
- Permissions for GitHub Actions to manage contents and deploy pages

#### Changed
- Updated CI workflows to use latest GitHub Actions versions
- Bumped `actions/upload-artifact` from 4 to 5
- Bumped `actions/checkout` from 5 to 6

#### Removed
- Obsolete coverage report files
- Build directory from version control
- Temporary `trace` file

### dev-0.2.1 – Stability & Cleanup Patch

#### Fixed
- Removed duplicate stats variable declaration in `core.py`
- Added missing validation for `ctx.eps` in `chebyshev_required_n`
- Removed dead code in `build_default_engine`
- Removed `<<DEBUG>>` marker from production code
- Fixed misplaced `test_parallel_cores` in `conftest.py`
- Added validation for positive `eps` in `StatsContext.__post_init__`
- Removed unused imports in test files (time, math)

#### Removed
- IDE-specific `.idea` directory
- Python cache directories (`__pycache__`)
- Temporary trace files

### dev-0.2.0 – Feature Expansion & Major Refactor

#### Added
- Installation instructions in `README.md`
- `CHANGELOG.md` for version tracking
- `CONTRIBUTING.md` with development guidelines
- Validation for `n_workers`, `confidence`, and `eps` parameters
- Class constants `_PARALLEL_THRESHOLD` and `_CHUNKS_PER_WORKER`
- `_CIResult` dataclass for confidence interval representation
- `_ensure_ctx` helper for clean context creation
- New `MCSimulation` methods for computing statistics and percentiles
- Edge case tests (`test_stats_engine_edge_cases.py`)
- Tests for result formatting with engine stats and metadata
- Tests ensuring statistical outputs are plain Python floats
- Import for `erfinv` from `scipy.special` for improved statistics

#### Changed
- Standardized type hints to PEP 585 (`dict`, `tuple`)
- Refactored percentile tracking to use method parameters
- Renamed `todo` to `metrics_to_compute` in `StatsEngine`
- Updated `ci_mean` and `ci_mean_bootstrap` to use `_CIResult`
- `_clean` refactored to return normalized `StatsContext`
- Improved statistical functions to use new context system
- Confidence interval handling expanded to support tuples and lists
- Improved simulation error handling and fallback behavior
- Refactored imports across modules (`__init__.py`, `core.py`, `stats_engine.py`)
- Enhanced parallel execution with better thread management
- Improved documentation formatting in `core.py`
- Enhanced test coverage and readability

#### Removed
- `_requested_percentiles_for_last_run` and `_engine_defaults_used_for_last_run` instance variables

### dev-0.1.0 – Initial Development

#### Added
- Core Monte Carlo simulation framework with abstract base class
- Built-in simulations: Pi estimation and Portfolio simulation
- Comprehensive statistics engine with multiple metrics
- Support for parallel execution using threads and processes
- Reproducible RNG seeding with `SeedSequence`
- Flexible confidence interval methods (z, t, bootstrap, Chebyshev)
- Framework for registering and comparing multiple simulations
- Comprehensive test suite
- Sphinx documentation

</details>
