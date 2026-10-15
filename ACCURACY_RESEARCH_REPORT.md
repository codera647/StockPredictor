# Measured accuracy research results

**The requested 90% direction accuracy has not been established.** These are research/model-selection results, not a new independent final-test result. The original final test was not read and the production bundle was not replaced.

The experiment keeps the original five-session UP/FLAT/DOWN target. It adds released macro vintages to make 58 inputs and trains logistic, CatBoost and a 64/32-hidden-unit neural network. Earlier validation selects a bounded probability mixture by log loss.

Training examples: 84,363; validation examples: 12,059.

| Classifier | Validation accuracy | Balanced accuracy | Log loss |
|---|---:|---:|---:|
| Macro logistic | 40.51% | 34.79% | 1.069433 |
| Macro CatBoost | 39.56% | 33.78% | 1.074564 |
| Macro neural network | 39.67% | 35.02% | 1.072674 |
| Selected macro ensemble | 40.38% | 34.64% | 1.064914 |
| Original price-only logistic | 41.94% | 36.53% | 1.064299 |

Selected component weights, in logistic/CatBoost/neural order: `[0.5, 0.25, 0.25]`. Model choice uses probability loss, so the mixture need not have the highest argmax accuracy.

## Earlier chronological checks

Each fold tunes only on its earlier inner validation and evaluates frozen weights on later dates. These periods remain research evidence.

| Evaluation period | Examples | Macro ensemble accuracy | Price-only logistic accuracy | Ensemble log loss | Baseline log loss |
|---|---:|---:|---:|---:|---:|
| 2021-01-04 to 2021-07-01 (exclusive) | 5,831 | 40.22% | 41.26% | 1.092649 | 1.082321 |
| 2022-01-03 to 2022-07-01 (exclusive) | 5,950 | 34.69% | 36.34% | 1.120235 | 1.125539 |

## Confidence and coverage

Temperature was fitted on 5,833 earlier calibration rows. The following diagnostic uses 6,050 later calibration rows, separated by a target-horizon purge. It selects a rule; it is not an independent assessment of that rule.

| Minimum confidence | Accepted examples | Dates | Coverage | Measured accuracy | Approximate 95% lower date-block bound | Baseline on same rows |
|---|---:|---:|---:|---:|---:|---:|
| 0.00 | 6,050 | 121 | 100.00% | 43.32% | 38.86% | 45.92% |
| 0.40 | 1,035 | 120 | 17.11% | 60.77% | 54.00% | 60.77% |
| 0.50 | 1 | 1 | 0.02% | 100.00% | unknown | 100.00% |
| 0.60 | 0 | 0 | 0.00% | unknown | unknown | unknown |
| 0.70 | 0 | 0 | 0.00% | unknown | unknown | unknown |
| 0.80 | 0 | 0 | 0.00% | unknown | unknown | unknown |
| 0.85 | 0 | 0 | 0.00% | unknown | unknown | unknown |
| 0.90 | 0 | 0 | 0.00% | unknown | unknown | unknown |
| 0.95 | 0 | 0 | 0.00% | unknown | unknown | unknown |
| 0.99 | 0 | 0 | 0.00% | unknown | unknown | unknown |

Rule status: `target_not_supported`. The rule remains disabled. Qualification requires at least 200 examples across 30 dates and a 90% approximate lower bound; a fresh future holdout is still required. The 100% row above contains only one prediction and does not qualify. Empty predictions have unknown accuracy. The block estimate is not a distribution-shift guarantee.

## Reproducibility and verification

Private artifact manifest: `manifests/accuracy-research/20261015T052315230825Z.json`.

Manifest SHA-256: `7cca8b4f7bb12a01f77a3104d8e4e0521ef6e205e94ca7dbac19b528fab3e11f`.

Recipe SHA-256: `8879e4f91d8e80b5104d0259dfa5d3f59c64a2b2f6108165bcce2f433528a3e1`.

Verified 15 private artifacts locally and remotely, including model reload probabilities and independently recomputed validation metrics. Macro release-cutoff violations: 0. Original core-code fingerprint and test binding are preserved. The full Python suite passed 68 tests and Ruff passed.

A transient private-storage read failure interrupted the first invocation. The same frozen experiment then completed against locally archived inputs; all completed objects were subsequently uploaded and checked against their SHA-256 values. This is the same pipeline used by the manual GitHub Actions workflow. Hosted Actions requires its separately configured secrets.

Historical macro release times are conservative ALFRED vintage reconstructions. Existing universe/survivorship and source-availability limitations still apply. These results do not authorize a production accuracy claim. See [research sources and implementation](ACCURACY_RESEARCH.md).
