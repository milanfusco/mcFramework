# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] – First PyPI Release

Initial public release of mcframework, a lightweight, reproducible, and deterministic Monte Carlo simulation framework.

### Added

#### Core Framework

- Core Monte Carlo simulation framework with abstract base class (`MonteCarloSimulation`)
- Comprehensive statistics engine (`StatsEngine`) with multiple metrics
- Support for parallel execution using threads and processes
- Reproducible RNG seeding with `SeedSequence`
- Flexible confidence interval methods (z, t, bootstrap, Chebyshev)
- Framework for registering and comparing multiple simulations
- `ComputeResult` class to encapsulate results from stats engine
- `StatsContext` for statistical configuration with cross-field validation
- `_CIResult` dataclass for confidence interval representation
- Class constants `_PARALLEL_THRESHOLD` and `_CHUNKS_PER_WORKER`

#### Simulations

- Built-in simulations: Pi estimation (`PiEstimation`) and Portfolio simulation (`PortfolioSimulation`)
- `BlackScholesSimulation` class for European option pricing via Monte Carlo
- `BlackScholesPathSimulation` class for path-dependent option analysis
- Greeks calculation methods (`delta`, `gamma`, `theta`, `vega`, `rho`)
- Organized simulation modules: `sims/black_scholes.py`, `sims/pi.py`, `sims/portfolio.py`

#### GUI Application

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
- Optional GUI dependencies in `pyproject.toml` (`mcframework[gui]`)

#### CI/CD & Automation

- Comprehensive CI/CD pipeline with multiple workflows:
  - `ci.yml` for linting (Ruff, Pylint), testing (Python 3.10–3.12), type checking (mypy), and building
  - `publish.yml` for PyPI publishing using trusted publishing (OIDC)
  - `codeql.yml` for automated security scanning
  - `release-drafter.yml` for automated release note generation
  - `stale.yml` for automated issue/PR cleanup
  - `docs-deploy.yml` for GitHub Pages deployment
  - `docs-validate.yml` for documentation validation on PRs
- Cross-platform CI testing (macOS, Windows) for Python 3.12
- Dependabot configuration for automated dependency updates
- Release drafter configuration for automated changelog generation

#### Documentation

- Sphinx documentation with pydata theme
- `PROJECT_PLAN.md` with detailed project roadmap and feature planning
- `SYSTEM_DESIGN.md` with architecture documentation and design decisions
- Mermaid architecture diagrams for design patterns and system views:
  - Class diagrams for core and stats modules
  - Sequence diagrams for simulation and bootstrap flows
  - Component and layered architecture diagrams
  - Design pattern diagrams (Adapter, Strategy, Template Method, Registry)
  - State lifecycle and execution flow diagrams
- "Getting Started" guide for user onboarding
- Autosummary templates for enhanced class documentation
- GitHub icon link in Sphinx HTML theme
- PyPI version, CI, coverage, and documentation badges in README
- `CHANGELOG.md` for version tracking
- `CONTRIBUTING.md` with development guidelines
- Demo files: `demo.py`, `demoBlackScholes.py`, `demoTickerBlackScholes.py`
- `TICKER_DEMO_README.md` with usage instructions

#### Testing

- Comprehensive test suite with pytest
- Test coverage reporting with pytest-cov and Codecov integration
- Edge case tests (`test_stats_engine_edge_cases.py`)
- Comprehensive test suite for Black-Scholes simulations (`test_black_scholes.py`)
- Tests for RNG state preservation in `calculate_greeks` method
- Tests for `ComputeResult` representation and error handling
- Tests ensuring statistical outputs are plain Python floats

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
- Renamed `todo` → `metrics_to_compute` in `StatsEngine`
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
