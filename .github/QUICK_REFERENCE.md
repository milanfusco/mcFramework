# GitHub Actions Quick Reference

## 🚀 Common Commands

### Local Testing Before Push

```bash
# Run all tests with coverage
pytest --cov=src/mcframework --cov-report=term-missing -v

# Check code style with Ruff
ruff check src/ tests/

# Check code quality with Pylint
pylint src/mcframework tests/

# Build documentation
sphinx-build -b html docs/source docs/_build/html -W

# Build package
python -m build
```

### Git Workflow with CI

```bash
# Create a feature branch
git checkout -b feature/my-feature

# Make changes and commit
git add .
git commit -m "feat: Add new feature"

# Push and create PR
git push -u origin feature/my-feature
# Then create PR on GitHub - CI will run automatically

# After PR is merged, CI runs on main branch
```

### Manually Trigger Workflows

**From GitHub Web Interface:**
1. Go to Actions tab
2. Select workflow (e.g., "CI")
3. Click "Run workflow" dropdown
4. Select branch
5. Click "Run workflow" button

**Using GitHub CLI:**
```bash
# Install GitHub CLI first: https://cli.github.com/

# Trigger CI workflow
gh workflow run ci.yml

# Trigger on specific branch
gh workflow run ci.yml --ref develop

# List workflow runs
gh run list

# Watch a workflow run
gh run watch
```

## 🔍 Viewing Results

### Check Workflow Status

```bash
# List recent runs
gh run list --limit 10

# View specific run
gh run view <run-id>

# View logs
gh run view <run-id> --log

# Download artifacts
gh run download <run-id>
```

### Check Coverage

**Locally:**
```bash
pytest --cov=src/mcframework --cov-report=html
open htmlcov/index.html  # macOS
```

**On GitHub:**
- Check PR comments for coverage report
- Visit Codecov dashboard (after setup)

## 📦 Release Workflow

### Version Bump Checklist

```bash
# 1. Update version
# Edit: pyproject.toml, src/mcframework/__init__.py

# 2. Update changelog
# Edit: CHANGELOG.md

# 3. Commit changes
git add pyproject.toml src/mcframework/__init__.py CHANGELOG.md
git commit -m "chore: Bump version to 0.2.0"
git push

# 4. Create release on GitHub
# - Go to Releases → New release
# - Tag: v0.2.0
# - Generate release notes
# - Publish

# 5. Workflow automatically publishes to PyPI
```

### Test PyPI Publishing

```bash
# Manually trigger publish workflow (goes to Test PyPI)
gh workflow run publish.yml

# Check it worked
pip install --index-url https://test.pypi.org/simple/ mcframework
```

## 🔧 Troubleshooting

### Re-run Failed Jobs

**Web Interface:**
1. Go to Actions tab
2. Click on failed run
3. Click "Re-run failed jobs" or "Re-run all jobs"

**CLI:**
```bash
gh run rerun <run-id>
gh run rerun <run-id> --failed  # Only failed jobs
```

### Debug Workflow

Add to workflow file for debugging:

```yaml
- name: Debug Info
  run: |
    echo "Python version: $(python --version)"
    echo "Pip version: $(pip --version)"
    pip list
    env
```

### Common Fixes

**Tests fail on CI but pass locally:**
```bash
# Ensure you're testing with the same Python version
python --version

# Clear cache and reinstall
pip cache purge
pip install -e ".[test]"
pytest
```

**Linting fails on CI:**
```bash
# Run exactly what CI runs
ruff check src/ tests/
pylint src/mcframework tests/

# Auto-fix what's possible
ruff check src/ tests/ --fix
```

**Package build fails:**
```bash
# Test build locally
python -m pip install build
python -m build

# Check the built package
pip install dist/*.whl
python -c "import mcframework; print(mcframework.__version__)"
```

## 📊 Status Badges

### Update Badges in README

```markdown
# Replace 'yourusername' with your GitHub username

[![CI](https://github.com/yourusername/mcframework/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/mcframework/actions/workflows/ci.yml)

# Customize branch
[![CI](https://github.com/yourusername/mcframework/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/yourusername/mcframework/actions/workflows/ci.yml)

# Codecov
[![codecov](https://codecov.io/gh/yourusername/mcframework/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/mcframework)
```

## 🔐 Managing Secrets

### Add a Secret

**Web Interface:**
1. Repository → Settings
2. Secrets and variables → Actions
3. New repository secret
4. Add name and value

**CLI:**
```bash
# Set secret from command line
gh secret set CODECOV_TOKEN

# Set from file
gh secret set CODECOV_TOKEN < token.txt

# List secrets (values are hidden)
gh secret list
```

### Update a Secret

```bash
# Same command overwrites
gh secret set CODECOV_TOKEN

# Delete a secret
gh secret delete CODECOV_TOKEN
```

## 🎯 Workflow Customization

### Skip CI on Specific Commits

```bash
git commit -m "docs: Update README [skip ci]"
# or
git commit -m "docs: Update README [ci skip]"
```

### Run Specific Job Only

Add conditions to jobs:

```yaml
jobs:
  test:
    if: contains(github.event.head_commit.message, '[run-tests]')
    ...
```

## 📈 Performance Tips

### Speed Up CI

1. **Use caching:**
   ```yaml
   - uses: actions/setup-python@v5
     with:
       cache: 'pip'  # ✅ Already enabled
   ```

2. **Run jobs in parallel:**
   - Lint, test, and docs jobs run in parallel ✅

3. **Matrix strategy:**
   - Tests run on multiple Python versions in parallel ✅

4. **Artifacts for debug:**
   ```yaml
   - uses: actions/upload-artifact@v4
     if: failure()
     with:
       name: test-results
       path: test-output/
   ```

## 🎓 Best Practices

### Commit Message Convention

```bash
# Format: <type>: <description>

# Types:
feat: New feature
fix: Bug fix
docs: Documentation
test: Tests
chore: Maintenance
ci: CI/CD changes
refactor: Code refactoring

# Examples:
git commit -m "feat: Add variance reduction technique"
git commit -m "fix: Correct confidence interval calculation"
git commit -m "docs: Update API reference"
git commit -m "ci: Add Python 3.13 to test matrix"
```

### Branch Protection

**Recommended Settings:**
1. Require pull request before merging
2. Require status checks: "Lint Code", "Test Python 3.12"
3. Require conversation resolution
4. Do not allow bypassing

**Setup:**
```bash
# Settings → Branches → Add rule
# Branch name pattern: main
# Check: "Require status checks to pass before merging"
# Select: Lint Code, Test Python 3.12
```

## 📱 Notifications

### Email Notifications

**Configure in GitHub:**
Settings → Notifications → Actions
- Choose when to receive emails
- Only failures, or all runs

### Slack Integration (Optional)

Add to workflow:

```yaml
- name: Notify Slack
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK_URL }}
```

## 🆘 Help & Resources

### Useful Links

- **Workflow Syntax:** https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions
- **GitHub CLI:** https://cli.github.com/manual/
- **Action Marketplace:** https://github.com/marketplace?type=actions
- **Codecov Docs:** https://docs.codecov.com/

### Getting Help

1. Check workflow logs in Actions tab
2. Review [SETUP_GUIDE.md](SETUP_GUIDE.md)
3. Review [README.md](README.md)
4. GitHub Actions documentation
5. Create issue in repository

---

**Pro Tip:** Bookmark this page for quick reference! 🔖

