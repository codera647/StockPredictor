# Measured SEC and historical-news experiment

The four feature variants were built and run with unchanged five-session direction labels. The existing final test remained closed and the existing bundle was not replaced. These are chronological research/model-selection results; they do not establish 90% production accuracy.

Highest observed overall validation accuracy: **42.71%** (`price_news_logistic`), compared with **42.03%** for the strongest price-only classifier by accuracy. The difference is 0.68 percentage points. None of the added-feature variants passed the predefined two-fold consistency gate. This small validation gain is not a demonstrated stable improvement or evidence of 90% accuracy.

Only 2,112 validation examples have an active historical news window. Overall accuracy changes, which include missing-news dates, do not by themselves establish that available news improves prediction. The matched-subset comparison below is essential to interpreting the result.

## Data actually collected

The pinned public Parquet conversion contains 15,549,299 metadata rows across 48 shards. Symbol statistics allowed reading 462,000 rows while downloading 83,302,059 bytes of column/metadata ranges. After filtering and cross-shard deduplication, 304,549 headlines remained for 49 of the original 50 stocks. Invalid matching rows: 5; cross-shard duplicate rows removed: 1,449.

Retained publication timestamps span 2009-06-22T00:00:00+00:00 to 2024-01-09T00:00:00+00:00. Model inputs use the separately frozen 2016-01-04 through 2023-12-15 research window and a 24-hour delay. Retained records outside that window do not extend the model's coverage. Per-stock story counts range from 399 to 11,708; availability is uneven. Stocks without matching news under the fixed symbol map: `HON`. Archive-window activity does not prove complete daily coverage.

SEC inputs reuse 964,419 cleaned facts, 9,836 filings and the existing 50-company issuer mapping. Annual flows, instant balance-sheet data and filing events obey the declared availability/period-matching rules. News and SEC missingness remain explicit.

The news source is a third-party conversion, pinned by revision/file metadata. Downloaded ranges and the retained private dataset have verified hashes. Full original CSV/file-LFS hashes are not claimed verified by column projection. See the [source and feature specification](FUNDAMENTAL_NEWS_EXPERIMENT.md).

## Existing validation block

Training rows: 84,363; validation rows: 12,059. This validation block selects hyperparameters/stopping/family, so its metrics are selection evidence rather than a new final test.

| Variant | Classifier | Inputs | Accuracy | Balanced accuracy | Log loss |
|---|---|---:|---:|---:|---:|
| price_only | logistic | 34 | 41.94% | 36.53% | 1.064299 |
| price_only | catboost | 34 | 42.03% | 36.53% | 1.064941 |
| price_sec | logistic | 62 | 41.91% | 36.76% | 1.064973 |
| price_sec | catboost | 62 | 42.61% | 37.07% | 1.064073 |
| price_news | logistic | 52 | 42.71% | 37.32% | 1.065116 |
| price_news | catboost | 52 | 42.28% | 36.76% | 1.064611 |
| price_sec_news | logistic | 80 | 42.33% | 37.09% | 1.065804 |
| price_sec_news | catboost | 80 | 42.46% | 36.74% | 1.065980 |

Original saved price-only logistic: 41.94% accuracy, 36.53% balanced accuracy and 1.064299 log loss on the same validation examples.

## Earlier chronological evaluations

Family choice is frozen using each fold's earlier inner-validation log loss. The later evaluation labels do not choose that family or its settings.

| Later evaluation period | Feature variant | Selected family | Accuracy | Balanced accuracy | Log loss |
|---|---|---|---:|---:|---:|
| 2021-01-04 to 2021-06-23 | price_only | logistic | 41.31% | 35.57% | 1.082591 |
| 2021-01-04 to 2021-06-23 | price_sec | logistic | 41.07% | 36.10% | 1.096359 |
| 2021-01-04 to 2021-06-23 | price_news | catboost | 40.03% | 33.77% | 1.085818 |
| 2021-01-04 to 2021-06-23 | price_sec_news | catboost | 40.70% | 34.31% | 1.082849 |
| 2022-01-03 to 2022-06-23 | price_only | logistic | 36.34% | 37.89% | 1.125539 |
| 2022-01-03 to 2022-06-23 | price_sec | catboost | 36.45% | 37.16% | 1.142283 |
| 2022-01-03 to 2022-06-23 | price_news | logistic | 36.18% | 37.84% | 1.136594 |
| 2022-01-03 to 2022-06-23 | price_sec_news | catboost | 36.94% | 37.61% | 1.141527 |

Earlier-fold baseline gates (lower log loss and noninferior accuracy in both folds): `{"price_sec": false, "price_news": false, "price_sec_news": false}`. These gates compare against a price-only family selected under the same procedure. They do not approve production deployment.

## Matched news-window validation subset

Within the existing validation block, 2,112 examples (17.51%) fall in an active source window. The following comparisons use exactly those same rows. Outcomes outside the window remain in the overall table above; their news inputs are missing.

| Variant/classifier | Accuracy | Balanced accuracy | Log loss |
|---|---:|---:|---:|
| price_only_logistic | 40.25% | 35.71% | 1.066737 |
| price_only_catboost | 40.72% | 36.05% | 1.065249 |
| price_sec_logistic | 40.86% | 36.08% | 1.067871 |
| price_sec_catboost | 40.81% | 36.08% | 1.064751 |
| price_news_logistic | 40.29% | 36.08% | 1.072298 |
| price_news_catboost | 40.06% | 35.60% | 1.065217 |
| price_sec_news_logistic | 40.67% | 36.32% | 1.071965 |
| price_sec_news_catboost | 40.06% | 35.46% | 1.063275 |

## Verification and reproducibility

All 75 Python tests passed; Ruff and the locked dependency check passed. Verified 29 model/data/prediction artifacts in private storage, plus the source/report/experiment manifests. Recomputed 24 complete metric sets and matched-subset metrics; all eight final JSON/native model reloads reproduced saved probabilities. Future-availability violations: 0. The original core-code fingerprint is preserved.

Experiment manifest: `manifests/fundamental-news/20261015T061830393043Z.json`.

Manifest SHA-256: `5b0b925846dc525779c86eb3f498f1e1f0165506c5f1c2ac56c9d65dac2beaa6`.

Source manifest: `manifests/fundamental-news/source-c63781010e4f9caf89bc6b9b935b6adaa11a29936c5273093c0ae2e8556fd8f2.json`.

Recipe SHA-256: `f91c0ebee00823bc0dc51e11e5ad2f77c1c0b733f3d9e37997d1aa4f83adff25`.

The same command and frozen inputs are available in the manual GitHub Actions workflow. This run completed against local archived inputs and its artifacts/checkpoints were subsequently uploaded and verified. Running hosted Actions requires separately configured repository secrets. Current-CIK mapping, survivorship, reconstructed receive times, third-party conversion and uneven/finite news coverage remain material research limits.
