# Plain LSTM comparison with DeepLOB

The referenced [stock-prediction repository](https://github.com/huseinzol05/Stock-Prediction-Models)
reports a price-error-based LSTM score. Its [notebook](https://github.com/huseinzol05/Stock-Prediction-Models/blob/master/deep-learning/1.lstm.ipynb)
does not calculate the fraction of correct DOWN/FLAT/UP predictions. A price score
above 82.94% cannot establish an improvement over our DeepLOB classification result.

Two separate experiments are implemented in `stock_predictor_lstm_compare`. The
original DeepLOB/TransLOB package, model weights and frozen study are unchanged.

## Completed price experiment

A controlled, independent PyTorch port uses the source notebook's single-layer,
128-unit LSTM, output keep probability 0.8, five-step truncated sequences and Adam
learning rate 0.01. It uses the same public `GOOG-year.csv` and final 30 test dates,
but reserves 30 validation dates and fits its scaler on training observations only.
The source notebook fits its min-max scaler before its train/test split.

The port trains on 192 dates, validates on the next 30, and tests on the final 30.
Selection uses validation RMSE for a 30-day recursive forecast. At the test origin,
the already observed validation history can prime the recurrent state; no future
test prices enter the forecast. Dropout is disabled at inference. Initialization,
validation selection, framework and omitted post-forecast smoothing differ from
the original notebook. This is not an exact reproduction of its reported score.

Sources are pinned to upstream commit `33266732b0b16188b565e0aeb6b24efa71161f6a`.
The original notebook SHA-256 is
`e3f96e0b85991bc3c49e512c327993ba336efe74164613455323978b716583bd`;
the CSV SHA-256 is
`3667ef08ae4dac9e8048d3942c703319bda86298aed9060c19d9bb3d0259cfe6`.
The upstream repository is Apache-2.0 licensed; this implementation does not execute
its downloaded legacy TensorFlow code.

The price score is:

```text
100 × (1 − sqrt(mean(((actual − predicted) / (actual + 1))²)))
```

It is not a calibrated probability or a classification rate and can be negative.
A constant last-price forecast can score highly. Both models therefore receive
MAE, RMSE and this same price score. A separately named forecast-path sign agreement
compares consecutive daily changes along the two 30-day paths; that target differs
from FI-2010's smoothed three-class labels. See [measured results](LSTM_PRICE_CHECK_REPORT.md).

## Matched FI-2010 classifier

For a direct classification comparison, a second model adapts the plain LSTM to
40 order-book inputs, 100-step histories, three-class logits and cross-entropy at
horizon 10. It is an architecture adaptation rather than the original price model.
It uses one 128-unit LSTM layer and 0.2 output dropout. Adam 0.001, batch 128,
seed 20261025, maximum 150 epochs, patience 15 and a 90-minute epoch-level time
budget match the main DeepLOB training settings where applicable. Selection uses
validation log loss. No class reweighting or test-driven threshold search is used.

The dataset, decimal normalization, purged chronological split and sample windows
match the verified DeepLOB horizon-10 run:

| Split | Windows |
| --- | ---: |
| Train | 203,601 |
| Validation | 50,751 |
| Test | 138,990 |

The final test checks the ordered file IDs, window endpoints and native labels
against SHA-256 `d27dc766018de39c7bb28f717535845284e3317acbd409e98b0f2acaa78d1b36`.
This prevents comparing a filtered or relabeled test subset with DeepLOB's score.
It reports accuracy, balanced accuracy, macro F1, confusion matrices and per-file
metrics, and compares them with DeepLOB's verified 82.9412%, 68.0917% and 71.7199%.

The FI-2010 test was already inspected before this experiment was proposed. This
is explicitly a retrospective comparison, not a fresh blind holdout. Any stronger
generalization claim needs new out-of-sample data. The existing daily holdout stays
closed and the daily forecasting model is not promoted or replaced.

## Run on free Colab GPU

Open [StockPredictor_LSTM_Comparison.ipynb](notebooks/StockPredictor_LSTM_Comparison.ipynb)
in Colab, select a GPU runtime, mount Drive and run the cells in order. It embeds
its source and tests and downloads both public datasets with checksums. No API
keys, `.env`, provider credentials or manual uploads are required. Free GPU access
and session duration depend on Colab's allocation.

It writes only to the new `MyDrive/StockPredictor_LSTM_Comparison` directory. Keep
the original `StockPredictor_FI2010` directory intact. The large classifier saves
model/optimizer/RNG checkpoints after every epoch and can resume with the same
code, configuration, Torch version and directory. Interrupted epochs repeat.
The small price check restarts if interrupted. Configuration/source changes after
the comparison's final-test freeze are rejected.

The notebook first runs the price check and shows its forecast and last-price
baseline. It then trains/selects the classifier without test evaluation. The
final-comparison cell freezes the weights, checks exact test-row alignment and
prints whether classification accuracy exceeds DeepLOB's 82.94%. There is no
guarantee of an improvement. The last cell exports weights, predictions, reports
and checkpoints as `StockPredictor_LSTM_Comparison_results.zip` in Drive.

The full GPU classifier has **not** been trained in this workspace. It has been
checked on synthetic data through fitting, freezing, serialization, cached metric
recomputation and interrupted-checkpoint continuation. The notebook's code cells
and embedded source are checked separately; no full Colab GPU execution is claimed.

## Shared commands

```powershell
.venv\Scripts\uv.exe sync --frozen --extra lob
.venv\Scripts\uv.exe run --frozen --extra lob stock-predictor-lstm-compare price
# On a CUDA-enabled environment with the source package available:
python -m stock_predictor_lstm_compare.pipeline classify
python -m stock_predictor_lstm_compare.pipeline classify --evaluate
# Rebuild the self-contained notebook:
.venv\Scripts\python.exe scripts/build_lstm_comparison_notebook.py
```

Colab preserves its installed CUDA PyTorch and does not install this project's
CPU-only optional Torch extra. Both routes use the same model/data/training code.
