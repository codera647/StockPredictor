# Direction challengers: measured comparison

Completed on 2026-10-15, Asia/Karachi. Compared XGBoost and CatBoost against the existing logistic/LightGBM direction classifiers using the same 34 features and labels.

**Neither challenger passed the replacement criteria. Logistic remains the recommended research direction comparison. No production model was switched.**

The existing final test and calibration blocks were not opened. Results below use the earlier validation block or nested earlier fold evaluations, which inform model selection; they are not new final-test accuracy.

## Earlier validation: 12,059 examples

| Model | Accuracy | Balanced accuracy | Log loss | Brier score |
|---|---:|---:|---:|---:|
| catboost | 42.03% | 36.53% | 1.064941 | 0.643443 |
| logistic | 41.94% | 36.53% | 1.064299 | 0.642649 |
| lightgbm | 41.74% | 36.38% | 1.066836 | 0.644349 |
| xgboost | 41.63% | 36.33% | 1.067130 | 0.644558 |
| prior | 39.21% | 33.33% | 1.081434 | 0.655785 |

CatBoost has the highest ordinary validation accuracy, but its gain over logistic is only 0.09 percentage points. Its log loss, Brier score and balanced accuracy are weaker. XGBoost also fails to improve on logistic. Ordinary accuracy alone would conceal those differences.

## Weighted earlier walk-forward comparison

| Model | Accuracy | Balanced accuracy | Log loss |
|---|---:|---:|---:|
| catboost | 38.74% | 36.17% | 1.113902 |
| logistic | 38.92% | 37.03% | 1.105583 |
| prior | 37.74% | 33.33% | 1.105180 |
| xgboost | 38.35% | 35.53% | 1.120386 |

These are evaluation-row-weighted averages over the first halves of 2021 and 2022. Each model tunes only on the earlier inner-validation period and refits on its purged outer training period. Logistic settings are chosen inside the same earlier fold protocol. Both challengers have weaker log loss and balanced accuracy than logistic across these folds.

## Selected settings

| Family | Setting | Boosting rounds |
|---|---|---:|
| catboost | compact | 273 |
| xgboost | moderate | 115 |

Two settings per family and a 400-round ceiling were declared before comparison. Model selection uses log loss, with accuracy/balanced-accuracy checks and a minimum 1% improvement requirement in validation and weighted fold log loss. No thresholds or labels were changed to inflate accuracy.

## Artifacts and verification

- Trained with 84,363 original training examples; validated on 12,059 examples.
- 7 private artifacts: four fold weights, two selected native model files and a row-level validation prediction table.
- Native XGBoost UBJ and CatBoost CBM models reload to matching probabilities; exact feature and DOWN/FLAT/UP class ordering are retained.
- 62 tests pass and Ruff passes. Tests forbid all calibration/final-test source reads and verify native reloads and cached-run reuse.
- All artifact checksums and remote metadata were verified; validation metrics were independently recomputed from saved predictions.
- The original trained bundle code fingerprint, artifacts and existing holdout binding remain unchanged. The core libraries were kept at their locked versions.

Comparison recipe: `0d3e3253e4c7dd09bdf19bfd363f6495ff94acaf97d4c4c80e88ed2128e80479`.
Private manifest: `manifests/challengers/20261014T201806415349Z.json`.
Manifest SHA-256: `f66c0842a4d0f382ce6ffddb0a12953321c3bd5e0724740f23b1cfbc9a8dadca`.
Private report: `reports/challengers/20261014T201806415349Z.json`.

Local reports: `artifacts/challengers/summary.json`, `report.json` and `verification.json`. Model files are under `data/archive/models/challengers/`. The shared command and manual GitHub Actions entry point are documented in [DIRECTION_CHALLENGERS.md](DIRECTION_CHALLENGERS.md). No new API key or environment variable is required.

## What this means for switching

A model-family change did not produce a meaningful measured improvement here. Continue using logistic as the research direction benchmark and the existing LightGBM quantile models for return ranges. Refinement should examine genuinely usable feature information and target/horizon experiments on earlier data. A new future holdout is needed for any later claim of improved final-test accuracy.

The original 41.49% logistic result was measured on the previous final test; the validation percentages above concern a different period and must not be compared as if they were a new test improvement. All models remain subject to the existing fixed-universe and reconstructed historical availability limitations.
