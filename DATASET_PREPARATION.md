# Daily dataset preparation

The shared `stock-predictor prepare` command performs canonical cleaning, evidence cleaning, trailing feature generation, five-session labeling and purged chronological splitting. It consumes the archived extraction manifest, checks input SHA-256 hashes and writes content-addressed Parquet tables plus an immutable private dataset manifest. The completed measured run is documented in [DATASET_PREPARATION_REPORT.md](DATASET_PREPARATION_REPORT.md).

Run the same stage used by GitHub Actions:

```powershell
uv run --frozen stock-predictor prepare --manifest-from-summary artifacts/extraction/summary.json
```

To reuse an archived extraction on a fresh machine, without recollecting evidence:

```powershell
uv run --frozen stock-predictor prepare --manifest manifests/full-extraction/20261014T163648065170Z.json
```

The private `.env` supplies Alpaca and S3 settings. Actions instead uses repository secrets. A first preparation fetches split-adjusted reference bars from the same free historical Alpaca feed for reconciliation; subsequent runs reuse verified reference checkpoints. `--no-upload` explicitly selects disk-only verification. The normal command uploads and checks remote bytes.

## Cleaning and availability

- Validate raw positive finite OHLC, integer nonnegative volume, OHLC inequalities, revisions and unique exchange-calendar sessions. Reindex missing sessions without filling market observations.
- Reconcile raw/reference price ratios against the archived forward/reverse split events. Preserve raw prices; derive compatible initial-share units using the cumulative factor of splits effective by each row. This is equivalent to converting every trailing window onto its cutoff's share basis for these relative features.
- Quarantine unresolved split ratios, conflicting bar revisions, unsupported spin-offs/identity changes and unexplained jumps. Restart indicator state and require 252 contiguous valid sessions after a boundary. Merger records involving a stock as acquirer do not imply its own share units changed.
- Retain observed receive timestamps separately from estimated historical availability. Close +30 minutes is the disclosed price reconstruction cutoff; full provider-day volume is lagged one completed session because it includes extended-hours prints.
- Clean auxiliary filings/facts, vintage macro observations and recent announcement/news records with validity flags. Preserve accession-specific facts, macro revisions and open-ended `9999-12-31` vintage intervals. Missing/outage evidence is not a fabricated zero news count. These tables are excluded from the first 34-feature model.

This run is a **research reconstruction**, with `production_ready=false`. Same-provider reconciliation does not independently certify complete corporate-action history or historical security identities. The fixed current universe has survivorship bias. Later-downloaded revisions and retrospective action review must be disclosed in model evaluation; first-receive live snapshots are still needed for production certification.

## Features and labels

The ordered input schema is defined once in `datasets/features.py`: 31 numeric features and three fixed liquidity missing flags. It covers returns/lags, trend, RSI, ranges/volatility, lagged liquidity and SPY context. Every rolling calculation is trailing and instrument-specific. RSI uses Wilder initialization and resets after invalid history. No target, symbol, timestamp or quality field enters `X`.

Liquidity conventions are frozen in `configs/dataset.daily.json`:

| Feature | Definition at cutoff t |
|---|---|
| `relative_volume_20_lag1` | V[t-1] / mean(V[t-21:t-1]), excluding V[t-1] from the denominator |
| `log_dollar_volume_20_lag1` | log(1 + mean(raw close * raw volume over the 20 sessions ending t-1)) |
| `volume_trend_5_20_lag1` | mean(V over 5 sessions ending t-1) / mean(V over 20 sessions ending t-1) - 1 |

Split-consistent price/volume units apply to volume ratios. The raw-dollar product is split invariant. The eligibility floor is USD 10 million of lagged 20-session mean close-times-volume; this is a liquidity proxy, not exact traded dollar volume. The initial model bundle needs 252 contiguous valid sessions and finite core/benchmark inputs.

`y_price_5d` is log(C[t+5] / C[t]) on compatible split units. `y_direction_5d` is UP/DOWN/FLAT using threshold max(0.005, 0.5 * trailing std20 * sqrt(5)). `y_exec_gross_5d` is a separate research companion from next-session open to target close. These are price-only outcomes excluding cash dividends, not total returns. The horizon follows five actual exchange sessions; invalid future paths and immature outcomes cannot enter fitting. Recent eligible unlabeled features are retained separately for inference.

## Chronological blocks

| Block | Assigned prediction sessions | Role |
|---|---|---|
| Train | Before 2023-10-16 | Fit model and any learned transformations |
| Validation | 2023-10-16 through 2024-10-13 | Tune parameters and select experiments |
| Calibration | 2024-10-14 through 2025-10-13 | Later probability/interval calibration |
| Test | From 2025-10-14 | Untouched final evaluation |

All stocks on one date receive the same block. Labels whose target reaches the next block are purged. No random row split is used. No imputation, scaling, clipping, feature selection, training or calibration is fitted during preparation; any learned processing must be fit only on its permitted earlier block.

Outputs include clean bars/actions and auxiliary evidence, full feature and label snapshots, split assignment/exclusion records, `X_*`, `y_*`, `index_*`, and recent `inference_features`. Metadata retains reconstructed and observed availability, row IDs, versions and source references. Each X/y/index triplet shares row order; the index maps each row to its stock, cutoff and target. `artifacts/dataset/report.json` lists every local/archive key and checksum; `summary.json` gives the manifest and block counts. The private archive also holds those reports.

## GitHub Actions

The extraction workflow now calls preparation on its completed manifest. Extraction exit 2 records optional-source gaps and still permits core dataset checks; fatal extraction errors stop the workflow. Preparation independently rejects unusable core data and empty fitting blocks.

`data-preparation.yml` allows rerunning preparation on an existing private manifest. It needs `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, `S3_ACCESS_KEY_ID` and `S3_SECRET_ACCESS_KEY` in repository Actions secrets; fixed S3 settings are in the workflow. The all-source extraction workflow additionally needs `SEC_USER_AGENT` and `FRED_API_KEY`. No dataset or credentials are committed to GitHub. Hosted execution requires those repository secrets; workstation `.env` values are not available to Actions.

Tests cover future-prefix invariance, today's-volume exclusion, session gaps, boundary purging across stocks, split-consistent labels, missing split verification, conflicting revisions, RSI edge cases, vintage revisions, observed news availability, conflicting SEC facts and an end-to-end checksummed artifact run.
