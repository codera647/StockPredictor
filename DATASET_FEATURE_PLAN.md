# Daily dataset cleaning and feature plan

Status: design only. No dataset has been downloaded or cleaned and no model has been trained.

Scope: free-source US equity daily MVP, five-trading-session forecast, using the expanded architecture and `BUILD_PLAN.md`. This document specifies the dataset stage of the same build/train/test/deploy pipeline; it does not introduce a separate prototype.

## 1. Row meaning and storage layers

One feature row represents one stable instrument at one after-close forecast cutoff. Compute each instrument independently, then combine rows into the cross-stock training panel. All historical calculations use information usable at that row's cutoff.

| Layer | Contents | Rule |
|---|---|---|
| Raw archive | Original provider responses, request parameters, pagination/checkpoints, timestamps and content hashes | Immutable; retain failed/invalid payloads for investigation |
| Canonical bars/actions | Typed, validated records, identifiers, revisions and quality reasons | Never silently overwrite older revisions |
| Feature snapshots | Instrument/cutoff/version plus model inputs | No future targets or later outcomes |
| Label snapshots | Separately generated future returns, direction and label intervals | Available to supervised fitting only after maturity |
| Dataset manifest | Exact snapshot IDs, feature list/code version, universe, adjustment policy and split dates | Reproduces a run without fetching the whole history again |

## 2. Required canonical columns

| Group | Columns |
|---|---|
| Identity | `instrument_id`, `symbol`, `exchange`, `currency`, separate `issuer_cik` where resolved |
| Session/time | `session_date`, `bar_end_at`, `available_at`, `retrieved_at`, `availability_method` |
| Raw observations | `open_raw`, `high_raw`, `low_raw`, `close_raw`, `volume_raw` |
| Source/version | `source`, `feed`, `record_version`, `response_hash`, `adjustment_basis`, `adjustment_version` |
| Validation | `quality_status`, `quality_reason_codes`, `action_verification_status` |
| Actions | Separate action records with event ID/type, split ratio or cash amount, announcement/effective/available timestamps, availability method and revision |

Derived OHLCV uses a verified split factor and a named adjustment version. Missing action verification must not be represented by an automatically assumed identity split factor.

Feature-row metadata includes `instrument_id`, `session_date`, `cutoff_at`, `feature_version`, `dataset_version`, `input_snapshot_id`, `max_input_available_at` and `readiness_status`. IDs, ticker strings, hashes, timestamps and quality status are audit fields, not default predictive inputs. Raw price/volume levels are retained for charts and checks; the initial model primarily uses relative quantities.

## 3. Cleaning sequence and failure rules

1. **Freeze collection scope.** Select the disclosed engineering universe, stock/ETF mapping, USD currency, regular sessions and explicit Alpaca SIP feed. Define a historical eligibility rule using only then-known history/liquidity. Require at least 252 completed validated sessions of history; freeze the configured liquidity floor before model selection. Record the fixed-universe survivorship limitation.
2. **Parse and type.** Parse prices as finite positive numbers, volume as a nonnegative integer, and timestamps as timezone-aware UTC. Derive session dates through the exchange calendar, not by taking a UTC date. Reject invalid values rather than coercing them to zero. Validate provider daily-bar timestamp semantics before assigning `bar_end_at`; use the actual regular-session close, including early closes.
3. **Resolve identities.** Map each bar to a stable security/share-class ID using effective ticker intervals. Do not merge two share classes because they share an issuer CIK. Quarantine ambiguous ticker reuse or unresolved identity instead of joining histories by today's ticker alone.
4. **Handle duplicates and revisions.** Key bars by instrument, session, source/feed and bar type. Collapse identical duplicate payloads in canonical views while retaining ingestion provenance. Conflicting values become separate revisions. Select only the revision usable at the simulated cutoff; live first-pass snapshots remain unchanged after corrections.
5. **Align to the actual exchange calendar.** Create expected session keys so every gap is visible. Weekends/holidays are not missing records. Mark early closes, halts, suspensions, listing changes and delistings explicitly. Never compute a five-session target by shifting five surviving downloaded rows across hidden gaps.
6. **Validate OHLCV.** Require `high >= max(open, close)`, `low <= min(open, close)`, `high >= low`, positive prices and nonnegative integer volume. Zero volume is not automatically a missing record; investigate consistency with session/trading status. Check that required stock and SPY bars are complete at cutoff and share the chosen feed.
7. **Recover actual missing bars.** Refetch missing expected sessions with bounded retries and saved checkpoints. Leave unrecovered values missing; do not interpolate prices/volume, backward-fill, or copy a prior close into a missing daily bar. An unverified gap inside a required feature window blocks that instrument's new forecast. A stale latest bar also blocks new inference.
8. **Reconcile jumps and corporate actions.** Flag suspicious discontinuities for action/source checks. Keep verified crashes, earnings gaps and large volume spikes: these are useful market observations. Quarantine unresolved possible unit/split errors rather than deleting all extreme returns.
9. **Build split-consistent values.** Preserve raw OHLCV and derive split-aware units as of each cutoff. Never combine dividend-adjusted close with raw high/low. If a split introduces `r` new shares per old share, convert a pre-split historical price to the post-split basis by dividing by `r`, and historical volume by multiplying by `r`. Apply only actions effective by the cutoff and known under the documented availability policy. Example: a 2-for-1 split transforms an earlier price 100 and volume 1,000 into 50 and 2,000; cash dividends remain separate events.
10. **Enforce availability.** Every input must satisfy `available_at <= cutoff_at`, and the bar must be finalized under the readiness policy. Live collection uses measured receive/processing times. Historical archives use trustworthy source timestamps or a disclosed reconstruction assumption; mark estimated availability. A revised archive downloaded today is not proof of first-pass historical inputs. Keep that replay limitation in the evaluation report.
11. **Calculate trailing inputs and readiness.** Sort and group by instrument and actual sessions. Use trailing windows ending at session `t`. At the documented after-close cutoff, today's finalized bar is allowed. Do not apply a blanket one-day shift to all features. Require contiguous verified windows and indicator warm-up state; windows must never bridge another stock or an unverified gap.
12. **Create labels separately, then split time.** Compute future targets only in the label stage. Fit transformations, select features and calibrate on the appropriate earlier blocks. Preserve mature labels for training and unlabeled recent rows for inference; do not delete recent rows from the live feature table.

Daily data finalization at close +30 minutes is an initial operational assumption, not a guarantee that no later correction occurs. Readiness checks can delay or abstain from a run.

## 4. Missing values and numerical treatment

| Condition | Treatment |
|---|---|
| Required stock/SPY price absent, stale or invalid | Refetch; block affected feature rows/inference until verified |
| Too little history or a gap in a required window | `insufficient_history` / `missing_required_input`; do not shorten the rolling window silently |
| Optional sector mapping/data unavailable | Choose the declared price+SPY bundle; do not attach today's sector history to past rows |
| Verified zero volume | Retain zero; zero denominator in a volume ratio produces missing value with an explicit indicator |
| Optional fundamental/macro value not yet released | Missing plus source-age/missing fields; never fill from a future release |
| Valid extreme market observation | Retain raw and label values; flag for reporting |
| NaN from undefined ratio | Store missing and its reason; never pass infinity to a model |

The 31-feature core bundle permits the three liquidity features to be undefined after valid observations with zero denominators; give each a fixed `_missing` column. Those flags are zero when the corresponding quantity is defined and one when it is mathematically undefined. They do not authorize missing-bar imputation. Core price/market readiness failures still block the row.

LightGBM can receive these explicitly permitted missing numeric values. For the linear comparison, fit median imputation and standardization on each fold's training portion only and reuse those fitted transformations in later blocks. If a permitted feature is entirely missing in training, reject that schema/run rather than inventing a median. LightGBM does not require standardization. Default to no clipping; any later clipping thresholds are learned on training only, versioned, and evaluated. Never clip future labels to improve the reported error.

## 5. Initial feature dictionary: 31 numeric features

Notation: `t` is the latest verified completed session. `O`, `H`, `L`, `C`, `V` are OHLCV on a consistent split basis valid at cutoff. `r_t = ln(C_t / C_(t-1))`. Rolling windows contain actual sessions and end at `t` unless specified otherwise. Stock/SPY returns use price changes excluding cash dividends. Sample standard deviation/covariance uses `ddof=1`; daily volatility remains in daily log-return units.

### A. Return history: 11 features

| Columns | Count | Definition |
|---|---:|---|
| `ret_1d`, `ret_5d`, `ret_20d`, `ret_60d` | 4 | `ln(C_t / C_(t-n))`, for `n = 1, 5, 20, 60` |
| `ret_1d_lag_1` through `ret_1d_lag_5` | 5 | `r_(t-j)`, for `j = 1..5`; each is a prior one-session return |
| `overnight_gap` | 1 | `ln(O_t / C_(t-1))` on compatible share units |
| `intraday_return` | 1 | `ln(C_t / O_t)` |

### B. Trend and price position: 6 features

| Columns | Count | Definition |
|---|---:|---|
| `close_to_sma_10`, `close_to_sma_20`, `close_to_sma_60` | 3 | `C_t / mean(C over last n sessions) - 1` |
| `drawdown_60` | 1 | `C_t / max(C over last 60 sessions) - 1`; distance below the trailing closing-price peak |
| `distance_to_high_60` | 1 | `C_t / max(H over last 60 sessions) - 1`; uses daily highs rather than closing peaks |
| `rsi_14` | 1 | Wilder RSI on 14-session gains/losses: seed with 14 simple averages, then recursive smoothing with alpha `1/14`. Output 100 if losses are zero and gains positive, 0 for the reverse, and 50 if both are zero |

RSI smoothing must use identical initialization and action handling in training and inference. On a change of split basis, replay needed history on that basis or correctly transform saved state. After a gap, restart and satisfy the versioned warm-up policy; never reuse an unexplained pre-gap state. Require at least 60 contiguous validated sessions for RSI warm-up and 61 for features needing 60 daily returns, independently of the 252-session eligibility minimum.

### C. Volatility and daily range: 5 features

| Columns | Count | Definition |
|---|---:|---|
| `rolling_std_5`, `rolling_std_20`, `rolling_std_60` | 3 | Standard deviation of the last `n` one-session log returns; requires `n+1` valid closes |
| `normalized_atr_14` | 1 | Simple mean of the last 14 true ranges divided by `C_t`, where `TR_s = max(H_s-L_s, abs(H_s-C_(s-1)), abs(L_s-C_(s-1)))`. This simple-average ATR convention is explicit and versioned |
| `range_pct` | 1 | `(H_t - L_t) / C_t` |

### D. Liquidity and participation: 3 features

| Columns | Count | Definition |
|---|---:|---|
| `relative_volume_20` | 1 | `V_t / mean(V over preceding 20 sessions, excluding t)`; requires 21 valid volume observations |
| `log_dollar_volume_20` | 1 | `ln(1 + mean(C_raw * V_raw over last 20 sessions))`; USD units and SIP feed recorded |
| `volume_trend_5_20` | 1 | `mean(V over last 5 sessions) / mean(V over last 20 sessions) - 1`, both ending at `t` |

Retain the raw arithmetic `dollar_volume_20` as a derived eligibility/audit value; only its named logarithmic transformation is a core model input. Close times volume is a daily dollar-volume proxy, not exact traded notional.

### E. Market context: 6 features

| Columns | Count | Definition |
|---|---:|---|
| `spy_ret_1d`, `spy_ret_5d`, `spy_ret_20d` | 3 | SPY log return over matching sessions |
| `spy_volatility_20` | 1 | Standard deviation of SPY's last 20 one-session log returns |
| `beta_spy_60` | 1 | `cov(stock r, SPY r over matching last 60 sessions) / var(SPY r)` |
| `residual_momentum_20` | 1 | `ret_20d - beta_spy_60 * spy_ret_20d`; defined here as a beta-adjusted cumulative-return proxy |

If the benchmark variance is zero, beta is undefined and the core bundle abstains rather than inventing a value. SPY inputs must be available by the same cutoff. Join on actual session dates and source availability, not dataframe row positions.

Total core schema: **31 numeric features + 3 fixed liquidity missing indicators = 34 model-input columns**. This fits the architecture's initial 30-60-column budget. Adding a feature requires a chronological ablation under matching labels, costs and universe; the count is a design choice, not a claim of predictive value.

## 6. Conditional and later feature families

### Optional sector bundle

Add `sector_ret_1d`, `sector_ret_5d`, `sector_ret_20d` and `stock_minus_sector_ret_20d = ret_20d - sector_ret_20d` when historical mapping and sector-ETF coverage are verified at every cutoff. This bundle has 38 model-input columns including the three existing liquidity flags. Current free company classifications do not establish historical sectors. Until reliable effective-date mappings exist, the 34-column price+SPY schema is the declared MVP bundle. A sector fallback is a separately trained/evaluated schema, not ad hoc column removal at serving time.

### Evidence features added only after coverage and ablation checks

| Family | Planned columns | Required cleaning/availability policy |
|---|---|---|
| SEC fundamentals | `revenue_growth_yoy`, `operating_margin`, `debt_ratio`, `financial_data_age_days`, `financial_data_missing` | Match accession, taxonomy, units, share class where relevant, fiscal period and original/amended dissemination. Compare compatible periods. Convert year-to-date cash-flow values into quarters only using facts already available at cutoff |
| Macro | `policy_rate`, `yield_10y`, `yield_2y`, `curve_slope`, `cpi_yoy`, `unemployment_rate`, per-source age/missing fields | Vintage/release-aware joins. Preserve units; convert percentages consistently; do not use today's revised history |
| Events | `days_to_known_earnings`, `days_since_earnings`, verified split/dividend event flags, event age/missing fields | Require an archived schedule known at cutoff. An unknown schedule stays missing. Direction/return labels are not event features |
| News | `news_count_24h`, `relevant_source_count_24h`, `sentiment_24h`, `sentiment_delta`, `news_feed_failed`, source-age fields | Entity/ticker relevance, canonical links and duplicate stories; publication and first-seen times; frozen/reviewed extractor. Feed healthy with no relevant news gives count 0; a failed feed gives missing count and failure flag |

Each extension defines exact formulas, staleness thresholds and extractor versions before use. Fundamentals may be carried forward only after release until the configured age limit. News/macro outages must not be converted into invented zeros. FinBERT sentiment is a text score, not a stock-direction probability. Later-trained text models may contaminate historical experiments; report and control that limitation.

Defer `turnover` until point-in-time shares exist, VWAP distance until suitable intraday or validated provider VWAP data exists, and earnings surprise until pre-release consensus history exists. Do not synthesize VWAP as a simple average of daily OHLC or calculate turnover from today's share count for past years. Adding these features must preserve the feature budget through evidence-based selection, rather than appending every candidate.

## 7. Labels and training-only columns

| Column | Meaning |
|---|---|
| `y_price_5d` | `ln(C*_(t+5) / C*_t)`, using verified compatible share units across the future action interval |
| `direction_threshold_5d` | `max(epsilon_5, k * rolling_std_20[t] * sqrt(5))`; configuration chosen using validation and frozen before holdout |
| `y_direction_5d` | UP above the threshold, DOWN below its negative, otherwise FLAT |
| `y_exec_gross_5d` | Later trading companion: `C*_(t+5) / O*_(t+1) - 1`, next-session open to target close |
| Label metadata | `target_session`, `horizon_sessions`, `label_start_at`, `label_end_at`, `label_available_at`, `target_version`, `label_quality_status` |

`t+5` means five actual exchange sessions, not five calendar days or five remaining downloaded rows. Future corporate-action outcomes may be used to calculate the future label; they must never enter the earlier feature snapshot. Preserve the original prediction cutoff when attaching outcomes. Label maturity requires target finalization and verified action processing, not merely reaching midnight on the target date.

For the initial dataset, unresolved target/path actions or unexplained gaps during the five-session label interval mark the label unusable until resolved. Report dropped-label reasons by period/instrument; do not silently omit delisted or halted securities to improve evaluation. Missing terminal outcomes prevent broad-universe claims.

The last five sessions normally lack mature five-session labels. Keep their features for inference; exclude their immature labels from supervised fitting. Predicted quantiles, calibrated probabilities and model outputs are produced after training and are never preprocessing inputs to the same model.

## 8. Chronological preparation and reproducibility

- Split **unique session dates** into training, tuning, calibration and evaluation blocks, then expand dates to the stock panel. No random row split; all companies for a date remain together.
- Check label intervals and maturity directly. Fitting labels must be available by the fit cutoff and must not overlap the next block. A five-row gap on a flattened stock panel is not a five-session gap.
- Use earlier years for walk-forward experiments. Reserve the latest fully labeled 6-12 months as the untouched final holdout if coverage permits. Record exact configurable dates and gaps in the manifest.
- Fit imputation/scaling, feature selection and optional clipping on the training block only. Early stopping/parameter selection uses tuning; probability/interval calibration uses the dedicated subsequent block. Purge boundaries between those blocks as required by label intervals.
- Store per-row cutoff and input provenance. Freeze source revisions, feed, calendar/action policies, liquidity/universe rule, feature formulas and ordering, label thresholds/versions and split dates.
- Use one shared feature engine for historical training and deployed inference. Refetch only needed partitions; recompute affected versioned outputs after corrections without rewriting issued forecasts.

## 9. Dataset-stage acceptance report

Before training, write a machine-readable and human-readable report containing coverage by stock/year/feed, duplicates/revisions, missing sessions, quarantine reasons, action coverage, reconstructed-availability share, feature warm-up losses, optional missingness, row/label counts by split and label-drop reasons. Report class balance without optimizing the untouched holdout.

Required checks: no nonfinite model values other than declared missing inputs; no feature source timestamp after cutoff; no silent session compression; no cross-stock rolling windows; no inconsistent split basis; no immature fitting labels; no overlapping label intervals across fitting/evaluation boundaries; immutable raw/feature versions; and identical feature definitions in training/inference. Unsupported coverage or schema blocks the pipeline with explicit reasons.

Artifacts: immutable raw snapshots, versioned canonical bars/actions, feature and label Parquet files, the dataset manifest, the quality report, and later the fold-specific fitted preprocessing artifacts bundled with their models. Deploy from those same versioned artifacts.

