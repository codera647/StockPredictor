# FI-2010 DeepLOB and TransLOB benchmark

The [Colab notebook and complete dataset handoff](COLAB_TRAINING.md) provide the
GPU training path requested for this experiment. This is a separate intraday
research track. It does not replace the five-session
daily model or support a claim of 80–90% accuracy for that model.

## Sources and published results

- [DeepLOB paper](https://arxiv.org/abs/1808.03668) and
  [author repository](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books).
  The paper's seven-day/three-day setup reports 84.47% accuracy at horizon 10.
- [TransLOB paper](https://arxiv.org/abs/2003.00130) and
  [author repository](https://github.com/jwallbridge/translob).
  Its table 3 reports 87.66% accuracy at horizon 10 and 91.62% at horizon 100.

Those are published benchmark results, not this project's measured results. The
architectures here are independent PyTorch implementations; they do not execute
downloaded notebooks or deserialize third-party model checkpoints.

The public FI-2010 ZIP is pinned to DeepLOB repository commit
`ff14d7c2fd38bdfc143389786993d0f0236d4eb8`. Its SHA-256 is
`7a5564290a504bdc104dccaf44aa6c0b3857bf1db6b0517b9b88577c2def3e5f`
(56,278,154 bytes). Parsing reads ZIP members directly, checks the 149-row schema,
keeps the first 40 order-book features, skips 104 engineered features, and reads
the five supplied label rows. Download, parsed-cache and archive checksums are
verified. No paid API or additional data key is needed.

FI-2010 has ten historical trading days from five Finnish stocks. It is an
event-based, downsampled order-book benchmark, not a daily-price dataset. Public
availability does not establish commercial-use or redistribution rights; resolve
the upstream data terms before commercial use.

## Frozen protocol

The complete machine-readable recipe is [configs/lob.fi2010.json](configs/lob.fi2010.json).
That file is the bounded CPU profile. The Colab GPU profile is
[configs/lob.fi2010.colab.json](configs/lob.fi2010.colab.json): stride 1 for training
and validation, CUDA device, up to 150 epochs, patience 15 and 90 minutes per model.
All remaining split, feature, target and test-evaluation rules below are shared.
The plan hashes that recipe, package source, Torch/Python versions and OS before
training. The same CLI is used locally and in GitHub Actions.

- Use the provided **decimal-precision normalization**. No scaler is fitted on
  validation or test observations. This differs from the z-score normalization
  used for the papers' reported headline comparisons.
- The official first-seven-day training file is split at 80% of its observations.
  Training ends 100 observations before that boundary. Validation inputs start
  at the boundary and require a complete 100-observation history. Features and
  training target horizons therefore do not cross the split.
- Training and validation use every twentieth eligible rolling window. Each
  input still contains 100 consecutive observations and all 40 raw features.
  The three official test files are evaluated separately, using **every** eligible
  rolling window. The final 100 endpoints of validation and each test file are
  discarded because the supplied labels' terminal-horizon treatment is unclear.
- Files have no instrument IDs or internal session-boundary metadata. Windows
  cannot cross file or split boundaries, but internal stock/day boundaries inside
  a supplied file cannot be audited. No instrument IDs are inferred from prices.
- Supplied native labels are explicitly mapped: 1 UP, 2 STATIONARY, 3 DOWN become
  internal indices 2 UP, 1 FLAT, 0 DOWN. Targets are the labels at the input window's
  final observation, for horizons 10 and 100 benchmark steps.
- Train both architectures at both horizons, using seed `20261015 + horizon`.
  Adam learning rates are 0.001 for DeepLOB and 0.0001 for TransLOB. Cross-entropy
  consumes logits. TransLOB's hidden projection has an L2 penalty of 0.0001.
  The batch size is 128 with two CPU threads and gradient clipping at norm 5.
- Each model has a maximum of 20 epochs, patience 5 and a 480-second training
  budget checked between complete epochs. A final epoch/validation can exceed
  that budget. Selection uses minimum validation log loss. Time limits depend on
  hardware, so the frozen checkpoint hashes, not just a seed, identify the run.
- All four selected checkpoints are frozen before any test member is parsed.
  An immutable holdout-policy record rejects changed recipes/model sets once the
  test is opened. Interrupted evaluation resumes those exact checkpoints.
  `--train-only` stops after fitting/selection and leaves the test closed.
- Report pooled and per-file accuracy, balanced accuracy, macro F1, confusion
  matrices, log loss, and a baseline using the training class frequencies.
  Saved predictions retain test-file IDs and window endpoints. Reloaded models
  must reproduce probe predictions before results can be published.

This bounded CPU comparison changes normalization, fitting stride, epoch budget,
optimizer details and implementation framework. It is **not an exact reproduction**
of either published score. Overlapping event windows are correlated, and three
test days cannot establish live profitability or broad market generalization.

## Architecture

DeepLOB uses three convolution blocks that reduce the 40 price/volume features
to a single width, three inception branches, a 64-unit LSTM and three-class
logits. TransLOB uses causal dilated convolutions, 14 learned channels plus one
position channel, three-head masked attention, two applications of a shared
transformer block, and a 64-unit classification head. No architecture is promoted
to production based on this experiment.

## Run and verify

```powershell
.venv\Scripts\uv.exe sync --frozen --extra lob
.venv\Scripts\uv.exe run --frozen --extra lob pytest -q
.venv\Scripts\uv.exe run --frozen --extra lob stock-predictor-lob
.venv\Scripts\uv.exe run --frozen --extra lob stock-predictor-lob --reuse-completed
```

The first benchmark run trains and evaluates. An ordinary identical rerun reuses
its report. `--reuse-completed` verifies the published frozen report, model and
prediction hashes and recomputes reported test metrics, without refitting; this
also works across operating systems. If no completed report exists, it runs the
benchmark. The workflow uses that option so later runs do not tune against an
already exposed test set.

Use existing `S3_ENDPOINT_URL`, `S3_REGION`, `S3_BUCKET_NAME`, `S3_ACCESS_KEY_ID`
and `S3_SECRET_ACCESS_KEY` for private artifact storage. No Alpaca credential is
required for this public benchmark. `.env` is never uploaded. Optional Torch CPU
dependencies are locked under the `lob` extra; daily workflows can omit that extra.

Outputs are `artifacts/lob/report.json`, `summary.json`, and (after verification)
`verification.json`; epoch checkpoints are under the recipe subdirectory.
Content-addressed source, plan, models, training histories, predictions and report
are stored under `lob/` in the existing private artifact archive. The holdout
policy and completed-report pointer are under `checkpoints/lob/`.
Colab uses the same package with `--no-upload` and a Google Drive archive; it needs
no S3 secrets. CUDA weights, optimizer state and CPU/CUDA generator state are
checkpointed for epoch-level continuation.

GitHub Actions uses [.github/workflows/lob-benchmark.yml](.github/workflows/lob-benchmark.yml)
with the two existing S3 credential secrets. Vercel is not a training host and
this benchmark adds no forecasting endpoint. Measured results are recorded in
[LOB_BENCHMARK_REPORT.md](LOB_BENCHMARK_REPORT.md) after training.
