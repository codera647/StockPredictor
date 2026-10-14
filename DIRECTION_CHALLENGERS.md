# Daily direction challenger comparison

The initial test accuracy was 41.49% for logistic regression, 40.73% for LightGBM and 40.28% for a prior-only classifier. These are three-class UP/FLAT/DOWN outcomes; the relevant comparison is the measured naive baseline, not an assumed binary 50% chance rate. A model-family change is an experiment rather than a promise of higher accuracy.

`stock-predictor-challengers` compares free CPU XGBoost and CatBoost classifiers with the existing logistic/LightGBM models on the same 34-feature daily dataset. It does not load the existing final test or the calibration tables. Return-quantile models and the original training artifacts remain unchanged.

```powershell
uv run --frozen stock-predictor-challengers
```

To run from a fresh runner, specify the private prepared and original training manifests:

```powershell
uv run --frozen stock-predictor-challengers --manifest manifests/dataset/20261014T171511441819Z.json --baseline-manifest manifests/training/20261014T195422207291Z.json
```

The same command is available in the manual `direction-challengers.yml` workflow. It uses existing private archive credentials and introduces no new API key or environment variable. `xgboost-cpu` avoids the larger GPU installation. CatBoost uses numeric inputs, native missing-value handling and no categorical ticker/issuer signal.

The frozen config `configs/challengers.daily.json` declares two settings per challenger, up to 400 rounds, early stopping patience 40, seed 647 and four CPU threads. Final challenger settings/iterations are selected on the existing earlier validation block by log loss. Report ordinary accuracy, balanced accuracy, Brier score, class-specific precision/recall, confusion matrices and reliability bins as well.

Two earlier nested checks evaluate the first halves of 2021 and 2022. For each fold, tune settings/stopping on the earlier inner validation, refit with those settings on the outer training period, and evaluate the subsequent fold period. The logistic comparison also selects its C only inside that earlier inner validation. Purge five-session targets at all boundaries using the existing split helpers.

The comparison recommends a research switch only if the strongest challenger by validation log loss beats logistic by at least 1% in validation and weighted earlier-fold log loss, with no loss of validation accuracy or balanced accuracy and no loss of weighted fold balanced accuracy. Otherwise retain logistic as the research direction comparison. This is not production approval: a new future holdout is needed to establish a replacement's final-test improvement. The reused validation/fold periods are selection evidence, not an untouched new accuracy claim.

Selected native UBJ/CBM artifacts and earlier-fold weights are stored locally and in the private archive. Their reloaded probabilities must match the fitted objects. Reports and row-level validation predictions retain hashes, feature/class order, data/core/challenger code fingerprints and dependency versions. Identical completed invocations reuse checked outputs without training again.

The challenger code has a separate package namespace, preserving the original bundle's code fingerprint and holdout binding. Tests enforce exact feature order, native serialization/probability normalization, refusal of test-enabled configs, no reads of reserved data, and cached-run reuse.

Read [DIRECTION_CHALLENGERS_REPORT.md](DIRECTION_CHALLENGERS_REPORT.md) for measured results. These remain subject to the original dataset's survivorship and reconstructed historical availability limits. If model changes provide little improvement, further research should test useful available-at-cutoff feature families and target/horizon choices on earlier periods, rather than optimizing against the already viewed final test.

Primary contracts: [XGBoost CPU installation](https://xgboost.readthedocs.io/en/stable/install.html), [XGBoost objectives/parameters](https://xgboost.readthedocs.io/en/stable/parameter.html), and [CatBoost multiclass objectives](https://catboost.ai/docs/en/concepts/loss-functions-multiclassification).
