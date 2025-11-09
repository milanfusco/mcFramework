# GitHub Actions Workflows Overview

## 📊 Workflow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         GitHub Push/PR                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        CI Workflow (ci.yml)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │   Lint Job   │  │   Test Job   │  │  Docs Build Job  │    │
│  │              │  │              │  │                  │    │
│  │ • Ruff       │  │ • Python 3.10│  │ • Sphinx         │    │
│  │ • Pylint     │  │ • Python 3.11│  │ • Upload Artifact│    │
│  │              │  │ • Python 3.12│  │                  │    │
│  │              │  │ • Coverage   │  │                  │    │
│  │              │  │ • Codecov    │  │                  │    │
│  └──────────────┘  └──────────────┘  └──────────────────┘    │
│                                                                 │
│  ┌──────────────────────────┐  ┌──────────────────────────┐  │
│  │ Test-Multiplatform Job   │  │     Build Job            │  │
│  │                          │  │                          │  │
│  │ • macOS-latest           │  │ • Build wheel            │  │
│  │ • Windows-latest         │  │ • Build source dist      │  │
│  │                          │  │ • Test installation      │  │
│  └──────────────────────────┘  └──────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   Publish Workflow (publish.yml)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Triggered by:                                                  │
│  • GitHub Release (→ PyPI)                                     │
│  • Manual Dispatch (→ Test PyPI)                               │
│                                                                 │
│  Steps:                                                         │
│  1. Build package (wheel + source)                             │
│  2. Verify with twine check                                    │
│  3. Upload to PyPI/Test PyPI                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              Dependabot (dependabot.yml)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Weekly Updates:                                                │
│  • GitHub Actions versions                                      │
│  • Python dependencies (grouped)                                │
│                                                                 │
│  Auto-creates PRs with:                                         │
│  • Updated versions                                             │
│  • Changelog links                                              │
│  • Security info                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Workflow Triggers

### CI Workflow

| Trigger | Branches | When |
|---------|----------|------|
| **Push** | main, master, develop | Every commit pushed |
| **Pull Request** | main, master, develop | When PR is opened/updated |
| **Manual** | Any branch | Via workflow_dispatch |

### Publish Workflow

| Trigger | When | Publishes To |
|---------|------|--------------|
| **Release** | New GitHub release created | PyPI (production) |
| **Manual** | Via workflow_dispatch | Test PyPI |

## 🔄 Job Dependencies & Parallelization

```
All jobs run in PARALLEL (no dependencies):

├── Lint Job (2-3 minutes)
├── Test Job - Python 3.10 (5-7 minutes)
├── Test Job - Python 3.11 (5-7 minutes)
├── Test Job - Python 3.12 (5-7 minutes) + Codecov upload
├── Test-Multiplatform - macOS (6-8 minutes)
├── Test-Multiplatform - Windows (6-8 minutes)
├── Docs Job (3-5 minutes)
└── Build Job (2-3 minutes)

Total Time: ~8 minutes (all run in parallel)
```

## 📈 Test Coverage Flow

```
Developer → Commits Code → Push to GitHub
                                │
                                ▼
                         CI Workflow Runs
                                │
                                ▼
                    pytest --cov runs on all platforms
                                │
                                ▼
                    coverage.xml generated
                                │
                                ▼
              Uploaded to Codecov (Python 3.12 only)
                                │
                                ▼
                    Codecov analyzes coverage
                                │
                                ▼
           Comment posted on PR with coverage report
                                │
                                ▼
                Badge updated in README.md
```

## 🚀 Release Flow

```
Developer → Update Version → Commit & Push
                                    │
                                    ▼
                        Create GitHub Release
                                    │
                                    ▼
                    Publish Workflow Triggered
                                    │
                                    ▼
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
        Build Package                           Verify Package
        (wheel + source)                        (twine check)
                │                                       │
                └───────────────────┬───────────────────┘
                                    ▼
                            Upload to PyPI
                                    │
                                    ▼
                    Package Available Worldwide
                                    │
                                    ▼
                    pip install mcframework
```

## 🔍 Linting Flow

```
Code Changes
      │
      ▼
┌─────────────┐
│ Ruff Check  │ ← Fast syntax & style checks
└─────┬───────┘
      │ PASS
      ▼
┌─────────────┐
│Pylint Check │ ← Deep static analysis
└─────┬───────┘
      │ PASS/WARN
      ▼
   Success ✅
```

**Ruff checks:**
- Import sorting
- PEP 8 style
- Unused imports
- Syntax errors

**Pylint checks:**
- Code quality
- Design patterns
- Complexity
- Documentation

## 💾 Caching Strategy

```
First Run:
├── Download Python ⏱️ 30s
├── Install pip packages ⏱️ 2-3 min
└── Run tests ⏱️ 3-4 min
Total: ~6 minutes

Subsequent Runs (with cache):
├── Download Python ⏱️ 30s
├── Restore pip cache ⏱️ 10s ← CACHED!
└── Run tests ⏱️ 3-4 min
Total: ~4 minutes

💾 Cache saves ~2 minutes per run!
```

## 🛡️ Security Features

### Dependabot
- **Monitors:** All dependencies + GitHub Actions
- **Frequency:** Weekly
- **Auto-creates:** Security update PRs
- **Grouped:** Minor/patch updates bundled

### Secrets Management
- **Required:** 
  - `CODECOV_TOKEN` (optional, for coverage)
  - `PYPI_API_TOKEN` (when publishing)
  - `TEST_PYPI_API_TOKEN` (for testing)
- **Stored:** Encrypted in GitHub
- **Access:** Only available during workflow runs

## 📊 Matrix Testing

### Python Versions Matrix

```python
Python 3.10 → Ubuntu Latest ✅
Python 3.11 → Ubuntu Latest ✅  
Python 3.12 → Ubuntu Latest ✅ + Coverage Upload
```

### Platform Matrix

```
Python 3.12 → macOS Latest   ✅
Python 3.12 → Windows Latest ✅
Python 3.12 → Ubuntu Latest  ✅ (covered in main test job)
```

**Why this strategy?**
- Python version testing on Linux (fastest)
- Platform testing on latest stable Python
- Coverage on most common deployment target (Linux + Python 3.12)

## 🎨 Badge Status

Badges in README.md show real-time status:

```markdown
[![CI](badge-url)]         → Green: All checks pass
                             Red: Something failed
                             Yellow: In progress

[![codecov](badge-url)]    → Shows coverage percentage
                             Green: >80%
                             Yellow: 60-80%
                             Red: <60%

[![Python 3.10+](badge)]   → Minimum Python version

[![License](badge)]        → Project license
```

## 📦 Artifacts Generated

### CI Workflow
- **Documentation HTML** (docs job)
  - Available for 90 days
  - Download from Actions tab

### Publish Workflow
- **Wheel file** (.whl)
- **Source distribution** (.tar.gz)
- Published to PyPI (not stored in GitHub)

## ⚙️ Configuration Files

```
.github/
├── workflows/
│   ├── ci.yml           ← Main CI/CD workflow
│   └── publish.yml      ← PyPI publishing
├── dependabot.yml       ← Dependency updates
├── README.md            ← Workflow documentation
├── SETUP_GUIDE.md       ← Step-by-step setup
├── QUICK_REFERENCE.md   ← Command reference
└── WORKFLOWS_OVERVIEW.md ← This file!
```

## 🎯 Success Criteria

A successful CI run means:
- ✅ Code passes Ruff style checks
- ✅ Code passes Pylint quality checks
- ✅ All tests pass on Python 3.10, 3.11, 3.12
- ✅ Tests pass on Linux, macOS, Windows
- ✅ Documentation builds without errors
- ✅ Package can be built and installed
- ✅ Code coverage maintained/improved

## 🔧 Customization Points

Easy to customize:

1. **Add Python versions:** Edit matrix in `ci.yml`
2. **Add OS platforms:** Edit matrix in `ci.yml`
3. **Change branches:** Edit `on:` section
4. **Add jobs:** Copy existing job structure
5. **Modify linting:** Update pyproject.toml
6. **Adjust coverage:** Update pytest config

## 📈 Metrics Tracked

- **Test Results:** Pass/Fail for each test
- **Code Coverage:** Line coverage percentage
- **Lint Score:** Ruff and Pylint findings
- **Build Time:** Duration of each job
- **Platform Compatibility:** Pass rate per OS

## 🚦 Status Checks

GitHub can require these checks before merging PRs:

**Recommended required checks:**
- ✅ Lint Code
- ✅ Test Python 3.12

**Optional required checks:**
- Test Python 3.10
- Test Python 3.11
- Build Package
- Build Documentation

Configure in: **Settings → Branches → Branch protection rules**

---

**Need more details?** Check out:
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Complete setup instructions
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command cheat sheet
- [README.md](README.md) - Workflow details

