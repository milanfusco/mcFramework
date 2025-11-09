# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Installation instructions in README.md
- CHANGELOG.md for version tracking
- CONTRIBUTING.md with development guidelines
- Validation for `n_workers`, `confidence`, and `eps` parameters
- Class constants `_PARALLEL_THRESHOLD` and `_CHUNKS_PER_WORKER` for configuration

### Changed
- Standardized type hints to PEP 585 style (`dict` instead of `Dict`, `tuple` instead of `Tuple`)
- Refactored percentile tracking to use method parameters instead of instance state
- Renamed `todo` variable to `metrics_to_compute` in StatsEngine for clarity
- Improved pytest coverage configuration to properly target source code

### Fixed
- Removed duplicate `stats` variable declaration in `core.py`
- Fixed missing validation for `ctx.eps` in `chebyshev_required_n`
- Removed dead code in `build_default_engine`
- Removed `<<DEBUG>>` marker from production code
- Fixed misplaced `test_parallel_cores` function in conftest.py
- Added validation for `eps` being positive in `StatsContext.__post_init__`

### Removed
- Instance variables `_requested_percentiles_for_last_run` and `_engine_defaults_used_for_last_run`

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

