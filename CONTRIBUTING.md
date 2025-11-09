# Contributing to mcframework

Thank you for your interest in contributing to mcframework! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python >= 3.10
- Git

### Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/mcframework.git
   cd mcframework
   ```

3. Install in development mode with all dependencies:
   ```bash
   pip install -e ".[dev,test,docs]"
   ```

### Development Environment

The project uses:
- **pytest** for testing
- **pylint** and **ruff** for linting
- **mypy** for type checking
- **Sphinx** for documentation

## Code Style

### Python Style Guide

- Follow PEP 8 conventions
- Use PEP 585 type hints (e.g., `dict` instead of `Dict`, `list` instead of `List`)
- Maximum line length: 110 characters
- Use NumPy/SciPy docstring style for all public functions and classes

### Type Hints

- All public functions should have complete type hints
- Use `from __future__ import annotations` at the top of modules
- Prefer built-in types (`dict`, `list`, `tuple`) over `typing` module equivalents

### Docstrings

Use NumPy/SciPy style docstrings with sections:
- Summary line
- Extended description (if needed)
- Parameters
- Returns
- Raises (if applicable)
- See Also (if applicable)
- Notes (if applicable)
- Examples

Example:
```python
def my_function(x: np.ndarray, ctx: StatsContext) -> float:
    r"""
    Compute something interesting from data.

    Parameters
    ----------
    x : ndarray
        Input data array.
    ctx : StatsContext
        Configuration context.

    Returns
    -------
    float
        The computed result.

    Examples
    --------
    >>> my_function(np.array([1, 2, 3]), ctx)
    2.0
    """
```

## Testing

### Running Tests

Run all tests with coverage:
```bash
pytest --cov=src/mcframework --cov-report=term-missing --cov-report=xml
```

Run specific test files:
```bash
pytest tests/test_core.py -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_<functionality>_<scenario>`
- Use fixtures from `conftest.py` for common setup
- Aim for high coverage (>80%)
- Include edge cases and error conditions

### Test Organization

- `test_core.py` - Core framework functionality
- `test_stats_engine.py` - Statistical metrics
- `test_sims.py` - Built-in simulations
- `test_errors_and_edge_cases.py` - Error handling
- `test_integration.py` - End-to-end tests

## Code Quality

### Linting

Run pylint:
```bash
pylint src/mcframework
```

Run ruff:
```bash
ruff check src/mcframework
```

### Type Checking

Run mypy (optional but recommended):
```bash
mypy src/mcframework
```

## Documentation

### Building Documentation

Build HTML documentation:
```bash
cd docs
sphinx-build -b html source _build/html
```

View documentation:
```bash
open _build/html/index.html  # macOS
```

### Documentation Guidelines

- Update docstrings for any new or modified functions
- Add examples to docstrings when helpful
- Update `docs/source/` RST files if adding new modules
- Include mathematical formulas using reStructuredText math directive

## Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation as needed
   - Update CHANGELOG.md under `[Unreleased]`

3. **Run tests and linting**:
   ```bash
   pytest --cov=src/mcframework
   pylint src/mcframework
   ruff check src/mcframework
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**:
   - Provide a clear description of the changes
   - Reference any related issues
   - Ensure CI passes

### Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests when relevant

## Project Structure

```
mcframework/
├── src/mcframework/
│   ├── __init__.py       # Public API
│   ├── core.py           # Core framework classes
│   ├── stats_engine.py   # Statistical metrics
│   ├── sims.py          # Built-in simulations
│   └── utils.py         # Utility functions
├── tests/               # Test suite
├── docs/                # Sphinx documentation
├── demo.py             # Example usage
└── pyproject.toml      # Project configuration
```

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Minimal code example to reproduce
- Expected vs. actual behavior
- Error messages and tracebacks

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Assume good intentions

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.

Thank you for contributing to mcframework!

