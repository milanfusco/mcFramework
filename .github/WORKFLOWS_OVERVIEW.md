# CI/CD Workflows

Here's what all the GitHub Actions do. Reference this when something breaks or you need to add stuff.

## What's What

| Workflow | What it does | When it runs |
|----------|--------------|--------------|
| `ci.yml` | Tests, linting, type checking | Every push/PR |
| `publish.yml` | Pushes to PyPI | When you create a release |
| `docs-deploy.yml` | Deploys docs to GitHub Pages | Push to main |
| `docs-validate.yml` | Makes sure docs build | PRs only |
| `codeql.yml` | Security scanning | Push/PR + weekly |
| `release-drafter.yml` | Writes release notes for you | When PRs merge |
| `stale.yml` | Closes old issues/PRs | Daily |

---

## CI (`ci.yml`)

The big one. Runs on every push and PR.

### Jobs (all run in parallel)

```
┌─────────────────────────────────────────────────────────────┐
│  lint        test           typecheck       build          │
│  ────        ────           ─────────       ─────          │
│  Ruff        Py 3.10        mypy            wheel + sdist  │
│  Pylint      Py 3.11                        test install   │
│              Py 3.12 + coverage                            │
│                                                             │
│  test-multiplatform                                         │
│  ──────────────────                                         │
│  macOS + Windows (Py 3.12 only)                            │
└─────────────────────────────────────────────────────────────┘
```

Takes a little while since everything runs at once, Windows takes the longest.

### Triggers

- Push to main/master/develop
- Any PR to those branches
- Manual run from Actions tab

### Notes

- **Ruff** fails the build if it finds issues
- **Pylint** just warns (set to `continue-on-error`)
- **mypy** also just warns for now (set to `continue-on-error`)
- Coverage only uploads from the Py 3.12 run (for now)

---

## Publishing (`publish.yml`)

Pushes the package to PyPI. Uses trusted publishing so no API tokens needed.

### How to release

1. Bump version in `pyproject.toml`
2. Commit and push
3. Create a GitHub Release with tag `v0.x.x`
4. Workflow runs automatically → package on PyPI

### Manual options

You can also trigger manually from Actions tab:

- Pick `testpypi` to test the release flow first
- Pick `pypi` to force publish without a release

---

## Docs

Two workflows here:

**`docs-deploy.yml`** - Builds and deploys to GitHub Pages on push to main.

**`docs-validate.yml`** - Just builds on PRs to catch broken docs before merge. Uploads the built HTML as an artifact you can download.

Make sure GitHub Pages is enabled (Settings → Pages → Source: GitHub Actions).

---

## Security (`codeql.yml`)

GitHub's code scanner. Runs on every push/PR plus weekly on Sundays.

Checks for the usual stuff: injection attacks, bad crypto, path traversal, etc.

Check results in Security tab → Code scanning alerts.

---

## Release Notes (`release-drafter.yml`)

Auto-generates release notes from your PRs. Label your PRs and they'll show up in the right section:

| Label | Shows up as |
|-------|-------------|
| `feature`, `enhancement` | 🚀 Features |
| `bug`, `fix` | 🐛 Bug Fixes |
| `docs` | 📚 Documentation |
| `dependencies` | 📦 Dependencies |

Also auto-labels PRs based on what files changed:

- Touch `docs/*` → gets `documentation` label
- Touch `tests/*` → gets `testing` label
- Touch `pyproject.toml` → gets `dependencies` label

When you're ready to release, there's already a draft release waiting with all the notes written.

---

## Stale Bot (`stale.yml`)

Keeps the repo clean. Marks stuff stale after:
- Issues: 60 days inactive
- PRs: 45 days inactive

Then closes after 14 more days if still no activity.

Won't touch anything labeled `pinned`, `security`, `bug`, or `enhancement`.

---

## Dependabot

Not a workflow but worth mentioning. Lives in `.github/dependabot.yml`.

Creates PRs weekly for:
- GitHub Actions updates
- Python dependency updates (grouped by dev/docs/etc)

---

## Files

```
.github/
├── workflows/
│   ├── ci.yml              # main CI
│   ├── publish.yml         # PyPI
│   ├── docs-deploy.yml     # deploy docs
│   ├── docs-validate.yml   # validate docs
│   ├── codeql.yml          # security
│   ├── release-drafter.yml # release notes
│   └── stale.yml           # cleanup
├── release-drafter.yml     # config for release-drafter
└── dependabot.yml          # dependency updates
```

---

## Secrets

| Secret | What for | Required? |
|--------|----------|-----------|
| `CODECOV_TOKEN` | Coverage reports | Optional |
| `GITHUB_TOKEN` | Everything else | Auto-provided |

PyPI publishing uses OIDC (trusted publishing) - no tokens stored.

---

## Branch Protection

If you want to require checks before merging to main:

**Definitely require:**
- `Lint Code`
- `Test Python 3.12`

**Maybe require:**
- `Type Check` (once types are stable)
- `Build Package`

Set this up in Settings → Branches → Add rule.

---

## When Things Break

**Tests pass locally but fail in CI?**
- Check Python version
- CI is a clean environment, no leftover state
- Make sure deps are in pyproject.toml

**Codecov not updating?**
- Check CODECOV_TOKEN is set in repo secrets
- Make sure coverage.xml is being generated

**PyPI publish fails?**
- Trusted publisher set up correctly?
- Version number already exists? (PyPI rejects dupes)
- Environment names match exactly?

**Docs deploy fails?**
- GitHub Pages enabled?
- Try building locally: `sphinx-build -b html docs/source docs/_build/html`
