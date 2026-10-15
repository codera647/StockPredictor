# Researching the 90% direction-accuracy target

The requested target is over 90% correct five-session UP/FLAT/DOWN predictions on genuinely unseen outcomes. The research pipeline keeps those labels, horizon and temporal boundaries unchanged. Its measured results are in [ACCURACY_RESEARCH_REPORT.md](ACCURACY_RESEARCH_REPORT.md).

## What the web research supports

[Gu, Kelly and Xiu, Empirical Asset Pricing via Machine Learning](https://dachxiu.chicagobooth.edu/download/ML_BKP.pdf) studies return prediction with richer characteristics, regularization, nonlinear models and neural networks. Its out-of-sample return-prediction improvements do not establish 90% daily direction accuracy. This motivates testing richer information rather than repeatedly replacing a tree library.

[FusionLSTM-CNF](https://www.nature.com/articles/s41598-026-43381-3) reports 61.2% on its binary stock-movement task and 71.2% on a confidence-filtered subset covering 40% of predictions. Its task and data differ from this project's five-session, three-class labels. Its historical datasets are available from the authors on request, so it is not an immediately reproducible free-data replacement. We test transferable ideas with existing free macro vintages, a small feedforward neural classifier and probability mixtures.

[Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500) formalizes trading prediction coverage for lower error. The paper's guarantees and image-classification experiments do not transfer automatically to dependent stock returns. Our optional confidence diagnostic reports accepted examples, dates, coverage, measured accuracy and an approximate date-block lower bound.

[Selective Prediction and the Persistence Illusion](https://www.mdpi.com/1911-8074/19/8/609) audits 90–94% financial classification results and attributes much of the result to persistent regime labels. It is not a demonstration of 90% five-session stock direction prediction. [Kapoor and Narayanan](https://reproducible.cs.princeton.edu/) document leakage-driven overoptimism across ML applications; the implementation retains the existing untouched-test restriction.

These sources motivate a bounded experiment. They do not supply a verified method that guarantees this project's requested 90% result.

## Implemented experiment

```powershell
uv run --frozen stock-predictor-research
```

On a fresh runner:

```powershell
uv run --frozen stock-predictor-research --manifest manifests/dataset/20261014T171511441819Z.json --baseline-manifest manifests/training/20261014T195422207291Z.json
```

The same code runs in `accuracy-research.yml`. It uses the existing private archive credentials and already collected free ALFRED/FRED observations; no additional API key or paid source is required. `--no-upload` explicitly runs against local archived inputs, useful while storage is unavailable. Normal operation uploads artifacts and verifies bytes. Local completion alone does not establish that remote uploads completed.

The input bundle contains 58 columns: the original 34, twelve macro numeric/change/age features and twelve missing indicators. Macro inputs include effective/target interest rates, two-/ten-year yields, curve slope, CPI year-over-year change, unemployment and three-month change, 20-session rate/yield changes, and source ages.

For every cutoff, replay only revisions with reconstructed release time no later than that cutoff. Preserve previously released versions until their replacement becomes usable. Date-only ALFRED vintages use the existing conservative next-day policy. Carry the latest known observation only within the configured age window; unknown/stale values retain missing flags. Historical observations downloaded later remain research reconstructions rather than verified live receive-time records.

Train macro-augmented logistic and CatBoost classifiers, plus a ReLU feedforward network with hidden layers 64/32. Fit preprocessing only on the training block. Neural training uses explicit chronological validation and epoch stopping; it does not create a random internal validation split. Export coefficients and transforms as JSON and the tree in native format, and verify reloaded probabilities.

Choose a bounded probability mixture using earlier validation log loss. Two earlier folds train on their inner-training periods, tune on inner validation, and evaluate later dates with frozen weights. The original price-only logistic comparison is retained. These reused historical periods inform research/model selection and do not become a fresh final test.

Within the original calibration block, fit probability temperature before 2025-04-14, purging labels that reach that boundary. Use the later portion for optional confidence-rule selection. A rule targeting 90% requires minimum accepted samples/dates and an approximate date-block lower accuracy bound of 90%. The diagnostic is not a replacement for overall accuracy or a time-series guarantee. Zero accepted predictions have unknown accuracy, not 100% accuracy. Any selected rule still requires a fresh future holdout and remains disabled for production.

The original final-test data is not loaded. Accuracy over all validation examples, per-class/balanced metrics, earlier-fold results and confidence-filtered coverage are reported separately. The original return models and their bundle/holdout binding are preserved. The requested accuracy target is not marked achieved without new independent evidence.

`configs/accuracy-research.daily.json` freezes budgets, dates, schemas, mixture candidates, confidence thresholds and evidence minimums. Immutable reports pin model/data/code references, native/JSON weights, augmented input matrices, macro provenance and row-level predictions. Completed identical invocations reuse verified outputs.

Tests cover future revision invariance, release timing, explicit macro missingness, neural JSON equivalence through the end-to-end run, probability/selector temporal separation, zero-coverage behavior, misleading confidence, unchanged targets, exclusion of existing test data, and cached-run reuse.
