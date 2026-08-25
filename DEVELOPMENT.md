# Development and verification

Create an isolated Python 3.12 environment, then install the checkout:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

Run the deterministic examples and visual gallery when plotting code changes:

```bash
python examples/continuous_demo.py
python examples/categorical_demo.py
python examples/color_schemes_demo.py
python scripts/render_visual_gallery.py --output visual_checks
```

The optional Palmer Penguins upstream builder additionally requires external
data obtained by the user and the reproduction dependencies:

```bash
python -m pip install -e ".[reproduction]"
python examples/palmer_penguins_global_categories.py --help
```

Never place an empirical input or output under the repository root. Follow
`ARTICLE_DATA_ACQUISITION.md` and point the builder at an external workspace.

Build and validate distributions only from a clean checkout:

```bash
python -m build
python -m twine check --strict dist/*
```

Install each wheel and source distribution into a fresh environment and run a
PNG/PDF smoke render before publishing.

## Release discipline

- Do not commit `dist/`, build outputs, caches, credentials, or files unrelated
  to the software repository.
- Do not commit raw, processed, derived, prediction, or explanation data.
- Update `_version.py`, `pyproject.toml`, `CITATION.cff`, `CHANGELOG.md`,
  tests, and documentation together.
- Run both pinned and latest-permitted Python dependency jobs.
- Regenerate hashes and release assets after every source or documentation
  change; never overwrite an already published version.
