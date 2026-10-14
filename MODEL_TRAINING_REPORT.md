# Daily models: measured training run

Completed on **2026-10-15, Asia/Karachi**. Archive run IDs use UTC and therefore begin `20261014`. The shared pipeline consumed the existing prepared dataset; no historical backfill or paid API was required.

**Models trained and artifacts verified. The candidate failed the direction performance gates and is not approved for production.**

## Training and artifacts

- Training: 84,363 examples; validation: 12,059; calibration: 12,133; final test: 11,989.
- Four main models: LightGBM q10/q50/q90 regressors and a three-class direction classifier, with the shared 34-feature daily schema.
- Comparisons: Ridge, multinomial logistic regression, per-stock ARIMA for all 50 stocks, and an unchanged-price/prior-frequency reference.
- The selected statistical model is ARIMA(0,0,0) on one-session log returns; ARIMA(1,0,0) was also fitted and compared using validation. No weekly seasonality was assumed.
- 20 immutable artifacts: eight daily model/reference files, eight walk-forward weight files, three evaluation prediction tables and one latest archived-cutoff research prediction table.
- 48 latest eligible archived-cutoff research predictions were saved. These are not refreshed live forecasts.

## Selected settings

| Model | Validation-selected setting | Iterations |
|---|---|---:|
| LightGBM q10 | compact | 44 |
| LightGBM q50 | moderate | 1 |
| LightGBM q90 | compact | 52 |
| LightGBM direction | moderate | 89 |
| Ridge | alpha=100.0 | n/a |
| Logistic | C=0.1 | n/a |

The q50 model stopped after one boosting round. This is consistent with the limited median-return improvement below. Hyperparameters and stopping iterations were selected before calibration and final-test access.

## Final test: return forecasts

The test contains prediction sessions 2025-10-14 through 2026-10-06, with targets through 2026-10-13. Errors and widths below are in log-return units, not percentages of correct price predictions. All intervals shown have the predeclared calibration applied.

| Model | MAE | RMSE | Mean pinball loss | Interval coverage | Mean interval width |
|---|---:|---:|---:|---:|---:|
| arima | 0.034831 | 0.049025 | 0.011522 | 81.68% | 0.111884 |
| lightgbm | 0.034790 | 0.048984 | 0.011316 | 79.90% | 0.106165 |
| naive | 0.034793 | 0.049035 | 0.011655 | 77.60% | 0.098668 |
| ridge | 0.034945 | 0.049172 | 0.011701 | 77.57% | 0.098618 |

LightGBM's mean pinball loss improved 2.91% against the unchanged-price reference and 1.79% against the strongest declared return comparison. Its median MAE is almost identical to the unchanged-price reference; this does not establish a useful point-forecast or trading edge.
Raw LightGBM quantile crossing rate: 0.00%. Predictions use the fixed rearrangement before interval widening.

## Final test: direction probabilities

| Model | Log loss (lower is better) | Multiclass Brier (lower is better) | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|
| lightgbm_direction | 1.081767 | 0.654679 | 40.73% | 35.50% |
| logistic | 1.075540 | 0.650408 | 41.49% | 36.96% |
| naive_direction | 1.087237 | 0.658965 | 40.28% | 33.33% |

The logistic comparison outperformed the LightGBM direction classifier on final-test log loss, Brier score and accuracy. LightGBM also failed the required validation improvement against logistic. Its modest improvement over prior-only probabilities is insufficient for the combined model gate. The test results did not trigger more tuning, a changed gate threshold or a switch to a test-selected model.

## Walk-forward checks

| Evaluation period | Training examples | Evaluation examples | LightGBM MAE | Reference MAE | LightGBM coverage |
|---|---:|---:|---:|---:|---:|
| 2021-01-04 to before 2021-07-01 | 49,785 | 5,831 | 0.028413 | 0.028582 | 79.37% |
| 2022-01-03 to before 2022-07-01 | 62,159 | 5,950 | 0.043280 | 0.040703 | 69.51% |

These earlier checks use a predeclared compact candidate and earlier inner-validation stopping, followed by outer-training refitting. Their fold evaluation outcomes are not used for stopping. They are uncalibrated; the weaker 2022 interval coverage demonstrates regime sensitivity and prevents treating final-test coverage as a universal guarantee.

## Date-block uncertainty

Moving blocks of 20 consecutive session dates retain whole stock panels. Two hundred bootstrap repetitions estimate approximate 95% intervals for LightGBM minus unchanged-price/prior-reference losses. Negative values favor LightGBM.

| Loss difference | Point estimate | Approximate lower 95% | Approximate upper 95% |
|---|---:|---:|---:|
| log_loss | -0.00547016 | -0.01038710 | 0.00002187 |
| mae | -0.00000242 | -0.00027842 | 0.00038304 |
| mean_pinball | -0.00033885 | -0.00050036 | -0.00015881 |

The median-return MAE and direction log-loss intervals include zero. The quantile-loss difference versus the naive interval is more consistent within this research sample. Bootstrap results do not eliminate survivorship, reconstructed availability or other dataset biases.

## Gates and verification

- Research model gate passed: `false`.
- Failed gates: validation and test direction log-loss improvement against the strongest direction comparison.
- Dataset production certification and production model approval remain false.
- 58 tests pass; Ruff passes. Holdout access order, purging, train-only preprocessing, probability/quantile rules, ARIMA forecasts and serialization are covered.
- Every artifact has its local SHA-256 checked and remote size/hash metadata verified. The manifest/report bytes were read back remotely with checksum verification.
- Saved final-test prediction metrics were independently recomputed, and the loaded native-text/JSON bundle reproduced the latest archived-cutoff predictions.
- A repeated identical training command reused the complete cache without fitting or re-opening test input tables.

## Reproducibility and next work

Dataset version: `a7abeaa5d7595e572153491047de74b3ab28605fa2d3273e8f59aa721fcf9331`.
Training recipe: `25ea7018c15696d481c97fd29487771289babdddcbbd6cab9a3c13906204d5fb`.
Training manifest: `manifests/training/20261014T195422207291Z.json`.
Manifest SHA-256: `002c60b3db08be7432f305ed4731a453da63932a256834f6fa333f794b95f021`.
Training report: `reports/training/20261014T195422207291Z.json`.

Local reports: `artifacts/training/summary.json`, `report.json`, and `verification.json`. Weights are under `data/archive/models/`; row-level evaluation and research prediction tables are under `data/archive/staging/training/`. All are also stored in the private archive and excluded from GitHub.

The immutable manifest pins model references, feature order, class order, training config, data/code hashes, dependency versions, calibration and trial records. Native text and JSON artifacts avoid pickle loading; the same ResearchBundle loader is available for future serving.

GitHub Actions now has extraction -> preparation -> training and a training-only entry point that reuses the archive. This measured run used the shared command with workstation `.env`; hosted Actions still needs repository secrets. No new environment variables were introduced.

Any further model experiments must select using earlier walk-forward/validation data and reserve a new future holdout for a later claim. The current test block is bound to this frozen recipe. Model refinement, first-receive live data certification, application tables, API/dashboard and deployment remain future work. No profitability or live production reliability is claimed.
