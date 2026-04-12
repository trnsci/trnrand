# Release procedure

trnrand publishes to PyPI via the GitHub Actions `publish.yml` workflow,
using PyPI [trusted publishers](https://docs.pypi.org/trusted-publishers/)
(OIDC) — no API tokens stored anywhere. Pre-releases (tags containing `-`,
e.g. `v0.2.0-rc1`) go to TestPyPI; stable tags go to PyPI.

## One-time setup

1. **Reserve the PyPI name.** Register `trnrand` on
   [PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/) by
   creating a placeholder release manually, or by configuring the trusted
   publisher first and letting the first OIDC publish create the project.
2. **Configure trusted publisher** — for both PyPI and TestPyPI:
   - Go to `Account → Publishing → Add a new pending publisher`.
   - Owner: `scttfrdmn`, repository: `trnrand`, workflow: `publish.yml`,
     environment: `pypi` (PyPI) or `testpypi` (TestPyPI).
3. **Confirm the GitHub environments exist.** In the repo settings under
   `Environments`, ensure `pypi` and `testpypi` are present (the workflow
   references them via `environment.name`).

## Cutting a release

1. **Bump version** in `pyproject.toml` and add a CHANGELOG section under
   `## [Unreleased]` → move it to `## [X.Y.Z] - YYYY-MM-DD`.
2. **Commit** the bump on `main` (or via PR).
3. **Tag and create a GitHub release**:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   gh release create vX.Y.Z --generate-notes
   ```
   For a release candidate, use `vX.Y.Z-rc1` — the workflow routes that to
   TestPyPI.
4. **Verify** the workflow run on the Actions tab. PyPI/TestPyPI shows the
   new version within a minute or two.

## Pre-release smoke test

Before cutting a stable release, do a TestPyPI dry-run:

```bash
git tag v0.1.0-rc1
git push origin v0.1.0-rc1
gh release create v0.1.0-rc1 --prerelease --generate-notes

# After the workflow lands, verify the install works from TestPyPI:
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            trnrand==0.1.0rc1
python -c "import trnrand; print(trnrand.__version__)"
```

## Rollback

PyPI does not allow re-uploading the same version. If a release is broken,
yank it (`pip` will refuse to install yanked versions but they remain
installable when explicitly pinned for reproducibility) and cut a new
patch version:

```bash
# from your PyPI account UI: yank the broken version
git tag vX.Y.(Z+1)
git push origin vX.Y.(Z+1)
gh release create vX.Y.(Z+1) --generate-notes
```
