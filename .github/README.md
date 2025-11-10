# GitHub Actions Workflows

This directory contains CI/CD workflows for the McFramework project.

## Workflows

### 1. CI Workflow (`ci.yml`)

Runs on every push and pull request to main/master/develop branches.

**Jobs:**

- **Lint**: Checks code quality using Ruff and Pylint
  - Runs on Python 3.12
  - Ruff checks for code style and common errors
  - Pylint runs with project configuration (non-blocking)

- **Test**: Runs the full test suite with coverage
  - Matrix testing on Python 3.10, 3.11, and 3.12
  - Generates coverage reports
  - Uploads coverage to Codecov (Python 3.12 only)

- **Test-Multiplatform**: Tests on macOS and Windows
  - Ensures cross-platform compatibility
  - Runs on Python 3.12

- **Docs**: Builds Sphinx documentation
  - Validates that documentation compiles without errors
  - Uploads documentation as artifacts

- **Build**: Verifies package can be built and installed
  - Creates wheel and source distributions
  - Tests installation from wheel

### 2. Publish Workflow (`publish.yml`)

Publishes the package to PyPI when a release is created.

**Triggers:**
- Automatically on GitHub release publication
- Manually via workflow_dispatch (publishes to Test PyPI)

## Setup Instructions

### Required Secrets

To use these workflows fully, you need to configure the following secrets in your GitHub repository settings (Settings → Secrets and variables → Actions):

#### For Coverage Reporting (Optional)

1. **CODECOV_TOKEN**: 
   - Sign up at [codecov.io](https://codecov.io)
   - Add your repository
   - Copy the upload token
   - Add it as a repository secret

#### For PyPI Publishing (Optional)

2. **PYPI_API_TOKEN**:
   - Create an account on [PyPI](https://pypi.org)
   - Go to Account Settings → API tokens
   - Create a token with "Entire account" scope (or specific project)
   - Add it as a repository secret

3. **TEST_PYPI_API_TOKEN** (for testing):
   - Create an account on [Test PyPI](https://test.pypi.org)
   - Create an API token
   - Add it as a repository secret

### Running Workflows

#### CI Workflow
- Runs automatically on push/PR to main, master, or develop branches
- Can be triggered manually from Actions tab

#### Publish Workflow
- Automatically runs when you create a GitHub Release
- For testing: Manually trigger from Actions tab (publishes to Test PyPI)

### Publishing a Release

1. Update version in `src/mcframework/__init__.py` and `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit changes and push
4. Create a new release on GitHub:
   - Go to Releases → Draft a new release
   - Create a new tag (e.g., `v0.1.0`)
   - Add release notes
   - Click "Publish release"
5. The publish workflow will automatically build and upload to PyPI

## Status Badges

Add these badges to your main README.md to show workflow status:

```markdown
[![CI](https://github.com/yourusername/mcframework/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/mcframework/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/yourusername/mcframework/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/mcframework)
```

Replace `yourusername` with your actual GitHub username.

## Customization

### Modifying Python Versions
Edit the matrix in `ci.yml`:
```yaml
matrix:
  python-version: ['3.10', '3.11', '3.12']
```

### Adjusting Branch Triggers
Edit the `on` section in `ci.yml`:
```yaml
on:
  push:
    branches: [ main, your-branch ]
```

### Disabling Jobs
Comment out or remove jobs you don't need from the workflow files.

## Troubleshooting

### Codecov Upload Fails
- Ensure CODECOV_TOKEN is set correctly
- The workflow is set to `fail_ci_if_error: false`, so builds won't fail

### Pylint Warnings
- Pylint runs with `continue-on-error: true` and uses `pylint-exit`
- It won't block CI, but warnings will be visible in logs

### Documentation Build Fails
- Check that all Sphinx dependencies are in `[project.optional-dependencies.docs]`
- The `-W` flag treats warnings as errors for documentation quality

### Package Build Fails
- Verify `pyproject.toml` configuration
- Ensure all dependencies are properly specified
- Check that `__version__` is defined in `__init__.py`

