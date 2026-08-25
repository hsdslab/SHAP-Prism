# Article data acquisition and external-input contract

This repository redistributes no empirical dataset, processed table,
precomputed explanation matrix, or row-level derivative. The software and its
tests run without those materials. The quick-start example is generated
entirely in memory from deterministic synthetic values.

The MethodsX article uses four public empirical sources. Readers who want to
repeat an application must obtain the source data from the data owner and
create the aligned feature and SHAP matrices in their own workspace. Source
licenses and access conditions remain with the source provider.

## Official source locations

| Application | Obtain the source from |
|---|---|
| Minnesota barley | R `lattice::barley` documentation: <https://stat.ethz.ch/R-manual/R-devel/library/lattice/html/barley.html> |
| Airfoil Self-Noise | UCI Machine Learning Repository record and DOI: <https://doi.org/10.24432/C5VW2C> |
| Wave soldering | R `rpart::solder.balance` documentation: <https://stat.ethz.ch/R-manual/R-devel/library/rpart/html/solder.balance.html> |
| Palmer Penguins | Tagged `palmerpenguins` source table: <https://github.com/allisonhorst/palmerpenguins/blob/v0.1.0/inst/extdata/penguins.csv>; archived package DOI: <https://doi.org/10.5281/zenodo.3960218> |

No download is performed by SHAP Prism. Acquisition, license review, data
cleaning, model fitting, and explanation generation are deliberate upstream
steps outside the plotting package.

## Renderer input contract

`plot_summary` requires:

- a two-dimensional feature table;
- a finite two-dimensional SHAP matrix with exactly the same rows and columns;
- explicit feature names for array inputs; and
- categorical dtypes for numeric category codes.

`plot_prism` additionally requires a nonmissing subgroup vector aligned to the
same rows and the name or index of one focal feature. A user-prepared article
workspace may use the following application-specific schemas. These names are
documented so that the retained renderer scripts can be connected to external
inputs; no table with these fields is bundled.

| Application | External renderer fields |
|---|---|
| Barley | `seed`, `fold`, `variety`, `site`, `year`, `yield`, `prediction`, `baseline_site_only_prediction`, `shap__site`, `shap__year` |
| Airfoil validation table | `specification`, `seed`, `frequency`, `Attack angle`, `Free-stream velocity`, `log10 frequency`, `Displacement thickness`, `Airfoil chord`, `Sound pressure (dB)`, `prediction`, `frequency_shap`, `fold_velocity` |
| Airfoil renderer NPZ | `shap_values`, `feature_values`, `feature_names`, `groups`, `chord_value`, `frequency`, `velocity`, `target`, `prediction` |
| Solder | `seed`, `Opening`, `Solder`, `Mask`, `PadType`, `Panel`, `skips`, `prediction_visible_skips`, `shap_Opening`, `shap_Solder`, `shap_Mask`, `shap_PadType`, `shap_Panel` |
| Penguins explanations | `source_row`, `fold`, observed/predicted/baseline fields, and paired `feature__<name>` / `shap__<name>` columns for each displayed feature |

Before plotting, verify row and column alignment, finite SHAP values, the model
output scale, and local additivity for the selected explainer. Encoded columns
may be summed to a source-feature contribution only when that mapping and its
additivity check are recorded.

## External input and output rules

- Keep acquired and derived empirical files outside the cloned repository and
  outside any release or supplementary archive.
- Do not commit, package, cache, or copy raw, cleaned, filtered, transformed,
  imputed, matched, explanation, prediction, or row-identifier tables.
- Point retained article renderer scripts at an external working directory.
- Write empirical renders and run records to an external output directory.
  Publication figures belong in the journal submission package, not in the
  software repository or software supplementary archive.
- Record source versions, hashes, model and explainer settings, folds, output
  scale, preprocessing, and complete plotting calls in a private analysis
  record. A checksum is evidence of file identity, not permission to
  redistribute the file.
- Review the source provider's current terms before sharing any independently
  acquired data or derivative.

The repository's synthetic notebook is the executable, distributable example
of the plotting workflow. It is not a substitute for the article analyses and
does not claim to reproduce their empirical results.
