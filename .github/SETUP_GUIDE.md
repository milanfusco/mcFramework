# GitHub Actions Setup Guide for McFramework

## 🎉 What's Been Created

Your project now has a complete CI/CD pipeline with the following workflows:

### 1. **CI Workflow** (`workflows/ci.yml`)
   - ✅ Linting with Ruff and Pylint
   - ✅ Testing on Python 3.10, 3.11, and 3.12
   - ✅ Cross-platform testing (Ubuntu, macOS, Windows)
   - ✅ Code coverage reporting with Codecov integration
   - ✅ Documentation building with Sphinx
   - ✅ Package build verification

### 2. **Publish Workflow** (`workflows/publish.yml`)
   - ✅ Automatic PyPI publishing on releases
   - ✅ Manual Test PyPI publishing for testing
   - ✅ Package validation with twine

### 3. **Dependabot Configuration** (`dependabot.yml`)
   - ✅ Automatic GitHub Actions updates
   - ✅ Automatic Python dependency updates
   - ✅ Grouped dependency PRs for easier management

### 4. **Status Badges** (in README.md)
   - ✅ CI status badge
   - ✅ Codecov coverage badge
   - ✅ Python version badge
   - ✅ License badge

## 🚀 Quick Start

### Step 1: Push to GitHub

First, commit and push these new workflow files:

```bash
git add .github/
git add README.md
git commit -m "ci: Add GitHub Actions workflows"
git push origin main  # or your branch name
```

### Step 2: Watch Your First Build

1. Go to your repository on GitHub
2. Click on the "Actions" tab
3. You should see the CI workflow running automatically

The workflows will run on:
- Every push to `main`, `master`, or `develop` branches
- Every pull request to these branches
- Manual triggers (via workflow_dispatch)

## 🔧 Optional: Configure Secrets

### For Code Coverage (Codecov)

1. **Sign up for Codecov:**
   - Go to [codecov.io](https://codecov.io)
   - Sign in with your GitHub account
   - Add your repository

2. **Get your upload token:**
   - In Codecov, go to your repository settings
   - Copy the "Repository Upload Token"

3. **Add to GitHub:**
   - Go to your GitHub repository
   - Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `CODECOV_TOKEN`
   - Value: Paste your token
   - Click "Add secret"

4. **Update badge in README:**
   Replace `yourusername` with your GitHub username in the Codecov badge URL

### For PyPI Publishing (When Ready to Publish)

1. **Create PyPI API Token:**
   - Create account on [pypi.org](https://pypi.org)
   - Go to Account Settings → API tokens
   - Create a new token (scope: Entire account or specific project)
   - **IMPORTANT:** Copy the token now (starts with `pypi-`)

2. **Add to GitHub:**
   - GitHub repository → Settings → Secrets and variables → Actions
   - Name: `PYPI_API_TOKEN`
   - Value: Paste your PyPI token

3. **Optional - Test PyPI:**
   - Create account on [test.pypi.org](https://test.pypi.org)
   - Create API token
   - Add as `TEST_PYPI_API_TOKEN` secret
   - Use workflow_dispatch to test publishing

## 📝 Update Badge URLs

In your README.md, replace `yourusername` with your actual GitHub username:

```markdown
[![CI](https://github.com/yourusername/mcframework/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/mcframework/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/yourusername/mcframework/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/mcframework)
```

For example, if your GitHub username is `john-doe`:
```markdown
[![CI](https://github.com/john-doe/mcframework/actions/workflows/ci.yml/badge.svg)](https://github.com/john-doe/mcframework/actions/workflows/ci.yml)
```

## 🎯 Publishing Your First Release

When you're ready to publish to PyPI:

1. **Update version numbers:**
   ```bash
   # In pyproject.toml
   version = "0.1.0"
   
   # In src/mcframework/__init__.py
   __version__ = "0.1.0"
   ```

2. **Update CHANGELOG.md** with release notes

3. **Commit and push:**
   ```bash
   git add pyproject.toml src/mcframework/__init__.py CHANGELOG.md
   git commit -m "chore: Bump version to 0.1.0"
   git push
   ```

4. **Create a GitHub Release:**
   - Go to your repository on GitHub
   - Click "Releases" → "Create a new release"
   - Click "Choose a tag" → Type `v0.1.0` → "Create new tag"
   - Fill in release title: `v0.1.0 - Initial Release`
   - Add release notes (can copy from CHANGELOG)
   - Click "Publish release"

5. **Watch the magic happen:**
   - The publish workflow will automatically trigger
   - Your package will be built and uploaded to PyPI
   - Check the Actions tab to monitor progress

## 🔍 Understanding the Workflows

### CI Workflow Stages

```
Lint Job
└─ Run Ruff (code style)
└─ Run Pylint (code quality)

Test Job (Matrix: Python 3.10, 3.11, 3.12)
└─ Install dependencies
└─ Run pytest with coverage
└─ Upload coverage to Codecov (3.12 only)

Test-Multiplatform Job (Matrix: macOS, Windows)
└─ Run pytest on different OS

Docs Job
└─ Build Sphinx documentation
└─ Upload as artifact

Build Job
└─ Build wheel and source distribution
└─ Verify installation
```

### What Gets Tested

- **Code Style:** Ruff checks for PEP 8 compliance and common issues
- **Code Quality:** Pylint performs deep static analysis
- **Functionality:** All tests in `tests/` directory
- **Coverage:** Line coverage of your source code
- **Documentation:** Ensures docs build without errors
- **Package:** Verifies the package can be built and installed
- **Cross-platform:** Tests on Linux, macOS, and Windows

## 🛠️ Customization

### Change Python Versions

Edit the matrix in `ci.yml`:

```yaml
matrix:
  python-version: ['3.10', '3.11', '3.12', '3.13']  # Add 3.13
```

### Change Trigger Branches

Edit the `on` section in `ci.yml`:

```yaml
on:
  push:
    branches: [ main, develop, staging ]  # Add more branches
  pull_request:
    branches: [ main, develop ]
```

### Disable Jobs

Comment out jobs you don't need:

```yaml
# docs:
#   name: Build Documentation
#   runs-on: ubuntu-latest
#   ...
```

### Add More Operating Systems

Extend the matrix:

```yaml
matrix:
  os: [ubuntu-latest, macos-latest, windows-latest, macos-14]  # ARM Mac
```

## 🐛 Troubleshooting

### Issue: CI fails on first run

**Solution:** Check the Actions logs for specific errors:
- Dependency installation issues: Update `pyproject.toml`
- Test failures: Fix failing tests
- Linting errors: Run locally first: `ruff check .` and `pylint src/`

### Issue: Codecov upload fails

**Solution:** This is okay! The workflow is configured with `fail_ci_if_error: false`
- Add `CODECOV_TOKEN` secret when ready
- Without token, coverage is still generated locally

### Issue: Documentation build fails

**Solution:** 
```bash
# Test locally
pip install -e ".[docs]"
sphinx-build -b html docs/source docs/_build/html -W
```

### Issue: Dependabot PRs are overwhelming

**Solution:** Adjust `dependabot.yml`:
```yaml
schedule:
  interval: "monthly"  # Change from weekly
open-pull-requests-limit: 3  # Reduce from 5
```

## 📊 Monitoring Your CI/CD

### GitHub Actions Dashboard
- View all workflow runs in the "Actions" tab
- Filter by workflow, branch, or status
- Download logs and artifacts

### Codecov Dashboard
- View coverage trends over time
- See which files/functions need more tests
- Get coverage reports on pull requests

### Dependabot Alerts
- Check "Security" tab for vulnerability alerts
- Review and merge Dependabot PRs regularly
- Configure auto-merge for minor updates (optional)

## 🎓 Next Steps

1. **Test locally before pushing:**
   ```bash
   pytest --cov=src/mcframework -v
   ruff check src/ tests/
   pylint src/mcframework tests/
   ```

2. **Set up branch protection rules:**
   - Settings → Branches → Add rule
   - Require status checks to pass before merging
   - Select "Lint Code" and "Test Python 3.12"

3. **Enable Dependabot security updates:**
   - Settings → Security & analysis
   - Enable "Dependabot alerts" and "Dependabot security updates"

4. **Consider adding:**
   - Pre-commit hooks for local linting
   - Performance benchmarks
   - Integration tests with real Monte Carlo scenarios
   - Automated changelog generation

## 📚 Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Codecov Documentation](https://docs.codecov.com/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Dependabot Documentation](https://docs.github.com/en/code-security/dependabot)

## ✅ Checklist

- [ ] Workflows pushed to GitHub
- [ ] First CI run successful
- [ ] Badge URLs updated with correct username
- [ ] Codecov token configured (optional)
- [ ] PyPI token configured (when ready to publish)
- [ ] Branch protection rules set up (recommended)
- [ ] Tested creating a release (when ready)

---

**Need help?** Check the [README.md](.github/README.md) in the `.github` directory for more details on each workflow.

