# Changelog

## 0.4.4 — 2026-08-25

### Changed

- Removed every bundled empirical source, processed table, explanation matrix,
  empirical run manifest, and empirical generated artifact. The release now
  contains only code-defined synthetic inputs and outputs.
- Added `ARTICLE_DATA_ACQUISITION.md` with official source locations, external
  renderer schemas, and a strict keep-outside-the-repository rule for
  empirical inputs and outputs.
- Converted the retained Palmer Penguins builder to explicit external input
  and output paths; it no longer assumes repository-local data.
- Set the canonical Palmer Penguins / Figure 5 renderer to top-aligned
  `right_stacked` category keys.
- Strengthened public-release validation and ignore rules against dataset-like
  files and empirical artifact directories.
- For categorical features above eight observed levels, retain seven raw
  levels under a deterministic frequency-and-tie rule and pool the remainder
  into a final `Others` entry while retaining every row; a literal raw
  `Others` level joins the pooled entry to keep labels unique.
- Disambiguated a literal raw `Missing` category from actual missing cells,
  including when the raw level is pooled into `Others`; the actual missing
  cells are labeled `Missing (NA)` whenever both occur together.

### Validation

- Added regression coverage for standalone, subgroup-global, and focal
  categorical paths, deterministic frequency ties, missing values, a literal
  raw `Others` or `Missing` level, all key placements, Figure 5 key placement,
  and sampling invariance.

All notable public-source changes are documented here.

## 0.4.3 — 2026-08-18

### Fixed

- Wrapped long focal-category legend titles to preserve required layout
  clearance with Matplotlib 3.11 while retaining Matplotlib 3.10 behavior.

### Repository

- Added a root-level Python continuous-integration workflow.
- Added contribution, security, development, and licensing documentation.
- Included the CC0 Palmer Penguins example and its provenance record.
- Added deterministic source checksums and repository-content validation.
- Kept the public source distribution focused on the verified Python
  implementation.

## 0.4.2 — 2026-07-30

- Initial publication-verified implementation.
