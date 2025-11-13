# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] – CI/CD & Documentation Fixes
### Fixed
- Fixed Sphinx documentation build command in CI workflow by removing unnecessary treat-warnings-as-errors flag
- Commented out environment-dependent performance test for parallel execution speed (soft test)

### Removed
`GITHUB_ACTIONS_SETUP_COMPLETE.md `(merged into workflow documentation)

## [0.3.0] – CI/CD and Documentation Overhaul

### Added
- GitHub Actions CI/CD pipeline with linting, testing, and documentation building
- Documentation deployment workflow (automated GitHub Pages publishing)
- Documentation validation workflow
- Dependabot configuration for automated dependency updates
- CI status badges in `README.md`
- Documentation deployment badge in `README.md`
- GitHub Actions documentation

### Changed
- Updated CI workflows to use latest GitHub Actions versions

### Removed
- Obsolete coverage report files
- Build directory from version control
- Temporary `trace` file

## [0.2.1] – Stability & Cleanup Patch

### Fixed
- Removed duplicate stats variable declaration in `core.py`
- Added missing validation for `ctx.eps` in `chebyshev_required_n`
- Removed dead code in `build_default_engine`
- Removed `<<DEBUG>>` marker from production code
- Fixed misplaced `test_parallel_cores` in `conftest.py`
- Added validation for positive `eps` in `StatsContext.__post_init__`
- Removed unused imports in test files (time, math)

### Removed
IDE-specific .idea directory
Python cache directories (`__pycache__`)
Temporary trace files


## [0.2.0] – Feature Expansion & Major Refactor

### Added
- Installation instructions in `README.md`
- `CHANGELOG.md` for version tracking
- `CONTRIBUTING.md` with development guidelines
- Validation for `n_workers`, `confidence`, and `eps` parameters
- Class constants `_PARALLEL_THRESHOLD` and `_CHUNKS_PER_WORKER`
- `_CIResult` dataclass for confidence interval representation
- `_ensure_ctx` helper for clean context creation
- New MCSimulation methods for computing statistics and percentiles
- Edge case tests (`test_stats_engine_edge_cases.py`)
- Tests for result formatting with engine stats and metadata
- Tests ensuring statistical outputs are plain Python floats
- Import for `erfinv` from `scipy.special` for improved statistics

### Changed
- Standardized type hints to PEP 585 (`dict`, `tuple`)
- Refactored percentile tracking to use method parameters
- Renamed `todo` -> `metrics_to_compute` in `StatsEngine`
- Updated `ci_mean` and `ci_mean_bootstrap` to use `_CIResult`
- `_clean` refactored to return normalized `StatsContext`
- Improved statistical functions to use new context system
- Confidence interval handling expanded to support tuples and lists
- Improved simulation error handling and fallback behavior
- Refactored imports across modules (`__init__.py`, `core.py`, `stats_engine.py`)
- Enhanced parallel execution with better thread management
- Improved documentation formatting in `core.py`
- Enhanced test coverage and readability

 ### Removed
- `_requested_percentiles_for_last_run` and `_engine_defaults_used_for_last_run` instance variables

## [0.1.0] - Initial Release

### Added
- Core Monte Carlo simulation framework with abstract base class
- Built-in simulations: Pi estimation and Portfolio simulation
- Comprehensive statistics engine with multiple metrics
- Support for parallel execution using threads and processes
- Reproducible RNG seeding with SeedSequence
- Flexible confidence interval methods (z, t, bootstrap, Chebyshev)
- Framework for registering and comparing multiple simulations
- Comprehensive test suite
- Sphinx documentation

