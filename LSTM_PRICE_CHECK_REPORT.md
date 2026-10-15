# Measured LSTM price check

A controlled PyTorch port was trained and evaluated on the referenced repository's
pinned `GOOG-year.csv`, using its final 30 dates: 2017-09-21 through 2017-11-01.
This is a price-regression experiment, separate from FI-2010 classification.

| Model | Repository-style price score | Price RMSE (USD) | Price MAE (USD) |
| --- | ---: | ---: | ---: |
| Plain LSTM price forecaster | 95.2013% | 47.9247 | 41.2523 |
| Repeat the last known price | 94.9999% | 49.9103 | 43.1420 |

The LSTM gains about **0.20 points of price score** over a constant-price forecast.
The high score by itself does not establish strong direction prediction or a win
over DeepLOB's 82.94% three-class accuracy. Daily forecast-path sign agreement is
63.33% (19 of 30), a different target and horizon setup from FI-2010.

Training stopped after 50 epochs by validation patience, with epoch 20 selected
using the lowest validation recursive-forecast RMSE. Min-max scaling used only the
192 training dates. It did not use the validation/test dates to fit normalization.
One fixed seed was used; no best-of-random-runs or test-driven tuning was performed.

The implementation, split and inference handling differ from the original legacy
TensorFlow notebook, as described in [LSTM_COMPARISON.md](LSTM_COMPARISON.md). These
numbers are our controlled-port results, not an exact reproduction of its claim.

Recipe: `adee83ac2070146b837afcd079cc4da983d7172faa834b3569e9857c209a85d9`.
Selected weight SHA-256:
`00780fa5109b27bcb4db3b985e352b235c186c1eeabd512234890c933873549e`.
Prediction artifact SHA-256:
`ebfac26723a8399eff0c66dd0c65da1976235a59cebaf14569264776fbbf6ca1`.

Local evidence: `artifacts/lstm-comparison/price_report.json`, per-recipe plans,
weights, full epoch history and predictions. The verification helper recomputes
both models' metrics from predictions and replays the selected model. Its plot is
`artifacts/lstm-comparison/price_forecast.png`.

The directly comparable plain-LSTM FI-2010 classifier is ready in the separate
Colab notebook. Its full GPU training and matched test score are pending. No claim
that it beats DeepLOB is supported yet; no production model has been replaced.
