# Prepared daily dataset: measured run

Completed on 2026-10-14 using the checksummed full-source extraction and the shared preparation command. All generated tables are saved locally under `data/archive/staging/datasets/` and in the private S3-compatible archive. Secrets and market datasets are excluded from GitHub.

**Research training dataset prepared; production certification remains false. No model has been trained or deployed.**

## Canonical cleaning

- Input: 138,210 daily bars, 50 stocks plus SPY, 2016-01-04 through 2026-10-13.
- Valid canonical rows: 138,202; missing sessions: 0; conflicting session revisions: 0.
- Raw-versus-split-reference reconciliation failures: 0.
- Seven unsupported action/identity boundaries and the unexplained AMD jump on 2016-04-22 are quarantined. Their future-label intervals are excluded and feature history restarts. Canonical records retain raw observations and quality reasons.
- Feature snapshots: 135,500; research-ready features: 121,554; contiguous-history exclusions: 13,938.

## Prepared model inputs and labels

- 34 ordered numeric input columns: price/return/trend/volatility, previous completed-day liquidity and SPY context.
- Five actual exchange-session, split-consistent price-return labels; UP/DOWN/FLAT thresholds are frozen in the dataset config. Cash distributions are excluded.
- 120,544 usable labeled examples across all four blocks.
- 240 eligible recent unlabeled feature rows retained for inference. HON and XOM currently fail the 252-contiguous-session requirement after recent action/identity boundaries.
- X, y and row audit indices are stored separately; each triplet has matching order. No fitted preprocessing or model artifacts were created.

| Block | Examples | First forecast session | Last forecast session | Last target session | Purged boundary rows |
|---|---:|---|---|---|---:|
| train | 84,363 | 2016-12-30 | 2023-10-06 | 2023-10-13 | 245 |
| validation | 12,059 | 2023-10-16 | 2024-10-04 | 2024-10-11 | 245 |
| calibration | 12,133 | 2024-10-14 | 2025-10-06 | 2025-10-13 | 250 |
| test | 11,989 | 2025-10-14 | 2026-10-06 | 2026-10-13 | 0 |

The gap before each following block removes prior predictions whose five-session target reaches that block. All stocks on a date remain together. Validation is for tuning; calibration is reserved for later probability/interval fitting; test remains the final evaluation block.

## Cleaned auxiliary evidence

| Table | Cleaned records (including retained quarantine) | Invalid records | Semantic duplicates removed |
|---|---:|---:|---:|
| sec_filings | 9,836 | 0 | 0 |
| sec_facts | 964,419 | 12 | 0 |
| macro_observations | 31,328 | 0 | 0 |
| macro_vintages | 12,638 | 0 | 0 |
| macro_series | 6 | 0 | 0 |
| sec_companies | 50 | 0 | 0 |
| company_announcements | 359 | 0 | 27 |
| news_gkg | 268 | 0 | 0 |
| news_articles | 0 | 0 | 0 |

These evidence tables are cleaned but excluded from the initial model inputs. The 12 invalid SEC facts remain quarantined. Filing accessions and macro revisions are preserved; announcement duplicates are collapsed by stock and canonical URL. Empty GDELT DOC staging and the extraction source failures remain explicitly reported. Recent news/IR observations are not backdated into a ten-year training history.

## Verification and reproducibility

- 48 tests pass and Ruff passes. Tests include future-prefix invariance, lagged volume, actual calendar gaps, action reconciliation, boundary purging, evidence revisions and a complete preparation run.
- 27 dataset Parquet objects: local SHA-256/row counts and remote object metadata verified. The dataset report and manifest were also read back remotely with checksum checks.
- Independently verified the ordered 34-column schema, finite core inputs/targets, direction rules, matching X/y/index rows, unique examples, date separation, label maturity and boundary purging.
- Reconstructed availability is checked against the declared cutoff. Actual observed receive times are retained; these do not establish historical first-publication availability.

Dataset version: `a7abeaa5d7595e572153491047de74b3ab28605fa2d3273e8f59aa721fcf9331`.
Dataset manifest: `manifests/dataset/20261014T171511441819Z.json`.
Manifest SHA-256: `1dd3a7165edea22dd6798846837661e70f0ac9dc48e831d7a42619bca029320a`.
Dataset report: `reports/dataset/20261014T171511441819Z.json`.

Local reports: `artifacts/dataset/summary.json`, `report.json` and `verification.json`. The report contains exact output keys, hashes, feature ordering, source manifest, config/code fingerprint, per-stock/year coverage, readiness/label exclusions and class counts. Raw archive objects and earlier manifests are retained.

## Limits and next stage

Historical bar/action availability is reconstructed from later provider revisions, and security identities use a fixed-symbol research mapping with action boundaries. Same-provider split reconciliation is not independent certification of complete action history. The current 50-stock universe is subject to survivorship bias. These conditions must be disclosed when evaluating models.

The next stage is training and walk-forward validation using these fixed artifacts, followed by calibration and final holdout evaluation. The preparation stage is included after extraction in GitHub Actions, with a second manual entry point to reuse an existing archive. This measured run used the shared command on the workstation and uploaded artifacts; hosted Actions has not been run because repository secrets were not confirmed configured.
