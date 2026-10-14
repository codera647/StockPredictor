# Daily five-session training

The shared `stock-predictor train` command consumes an immutable prepared dataset, trains the declared candidates, calibrates the selected models, evaluates a frozen final holdout and saves checksum-verified artifacts. Training libraries are free and open source. No new API subscription, key or environment variable is required.

```powershell
uv run --frozen stock-predictor train --manifest-from-summary artifacts/dataset/summary.json
```

On another runner, reuse the prepared archive directly:

```powershell
uv run --frozen stock-predictor train --manifest manifests/dataset/20261014T171511441819Z.json
```

The normal command uses `.env` for private S3 access and the shared configuration loader. It does not recollect market history. `--no-upload` explicitly selects disk-only verification. GitHub Actions uses repository secrets rather than the workstation file. The `model-training.yml` entry point reuses an existing prepared manifest; extraction and preparation workflows can continue through training with `train_models=true`.

## Fitted models

| Family | Models | Target/output |
|---|---|---|
| Main return model | Three pooled LightGBM quantile regressors, alpha 0.1/0.5/0.9 | `y_price_5d`: lower, median and upper return endpoints |
| Main direction model | Pooled LightGBM multiclass classifier | `y_direction_5d`, probabilities in DOWN/FLAT/UP order |
| Linear comparisons | Ridge regression and multinomial logistic regression | Return and direction |
| Statistical comparison | Per-stock ARIMA(0,0,0) or ARIMA(1,0,0) on one-session log returns | Conditional five-session aggregate return and Gaussian interval |
| Naive reference | Zero median return; training-derived empirical tails and class frequencies | Unchanged-price return and prior direction probabilities |

All pooled models use the same ordered 34-column feature schema; ticker strings and future outcomes are excluded from X. ARIMA fits the most recent sufficiently long contiguous eligible training segment for each stock. Coefficients stay fixed in later blocks; predictions condition on the currently observed one-session return. Fitting percentage log-return units improves numerical optimization; saved parameters are converted back to log-return units. This is an ARIMA comparison, without an assumed weekly seasonal cycle.

The executable-return companion, text models and intraday models remain later architecture stages. This command fits the agreed initial daily forecast bundle and comparisons.

## Selection and calibration

`configs/training.daily.json` freezes the seed, CPU threads, tuning budgets, walk-forward dates, quantiles, calibration rules, bootstrap settings and model gates before test access.

- Fit on 84,363 training examples. Tuning/early stopping uses only the 12,059-example validation block.
- Two LightGBM candidates are logged for each quantile and the classifier. Each quantile selects validation pinball loss; direction selects multiclass log loss. Ridge checks three alphas using MAE; logistic checks two C values using log loss. Two ARIMA orders are compared using validation return MAE. Failed statistical fits are recorded rather than silently substituted.
- Fit all linear imputation medians and standardization parameters on training only. LightGBM uses the declared missing-value convention directly.
- Repair quantile crossing with a predeclared row-wise rearrangement. Measure original crossing frequency; evaluate repaired outputs.
- Use the dedicated 12,133-example calibration block for global widening of the q10-q90 interval and bounded multiclass temperature calibration. Median predictions are unchanged by interval padding. Apply the same calibration approach to comparisons.
- Calibration-block metrics are fitting diagnostics. Assess realized coverage and probability quality on the later held-out block; dependence between stocks and overlapping labels prevents an iid coverage guarantee.

Before final model selection, two walk-forward checks evaluate the first half of 2021 and 2022 using only earlier data. A fixed compact candidate stops on an earlier inner validation period, then refits on the outer training period with the frozen iteration count. Neither fold evaluation period is used for early stopping. Labels are purged at inner and outer boundaries. These fold checks report uncalibrated performance against the earlier-training naive reference.

## Frozen holdout and reporting

After model selection/calibration, save a selection checkpoint and bind the test archive hashes to that exact recipe. Only then load the 11,989-example final test block. New recipes cannot reuse that same holdout for another selection. Identical completed runs reuse checked artifacts and metrics without fitting or re-opening the test dataset. Interrupted runs can resume their frozen selection checkpoint.

The report includes validation/calibration/test return MAE/RMSE, per-quantile and mean pinball losses, interval coverage/width, multiclass log loss/Brier score, accuracy/balanced accuracy, confusion matrices, per-class metrics, reliability bins and per-stock/year diagnostics. Moving contiguous date-block bootstrap comparisons retain whole stock panels and five-session target dependence; report approximate 95% intervals rather than iid row uncertainty.

Predeclared gates require at least 1% mean-pinball improvement against the strongest declared comparison, direction-log-loss improvement against the strongest direction comparison, median MAE no worse than unchanged price, Brier no worse than prior probabilities, and test interval coverage within five percentage points of 80%. Model gates and dataset certification are distinct. A trained candidate remains research-only while the dataset's observed historical availability/security identities are reconstructed; no model is automatically promoted or deployed.

`artifacts/training/summary.json` and `report.json` contain actual metrics, gate results and private manifest keys. All weights, preprocessing/calibration parameters, fold weights, row-level evaluation predictions and latest archived-cutoff research predictions are stored locally and remotely. A completed report pins the dataset manifest, training config, code hash, dependency versions and ordered schema. Recent research predictions are not a refreshed live forecast ledger.

## Loading and verification

`training/bundle.py::ResearchBundle` loads the same immutable bundle for research inference and future serving. It verifies checksums and the exact feature order. LightGBM uses native text artifacts; linear coefficients, preprocessing, ARIMA and calibration use JSON. No pickle deserialization is used. Reloaded predictions are compared against fitted objects before test access.

```powershell
uv run --frozen ruff check backend
uv run --frozen pytest -q
```

Training tests cover quantile loss/crossing, calibration/class order, training-only linear preprocessing and JSON equivalence, conditional ARIMA forecasting, actual boundary purging, schema rejection, holdout access order, end-to-end artifact reload and cached-run reuse. Earlier extraction/preparation tests remain part of the same checks.

Primary implementation contracts: [LightGBM training](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html), [scikit-learn logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [statsmodels ARIMA fitting](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.fit.html), and [SciPy bounded scalar optimization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html).
