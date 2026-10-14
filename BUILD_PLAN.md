# Stock Predictor build plan

Status: the shared extraction pipeline implements all planned free-source families: Alpaca, Nasdaq, SEC facts/filings, FRED/ALFRED, GDELT and official company announcements, with explicit per-source history windows and limits. Historical daily and evidence collection have run, with external news/IR gaps retained in reports. Locked dependencies, private archives/checkpoints, manifests/reports and the manual GitHub Actions workflow use the same code. See [DATA_EXTRACTION.md](DATA_EXTRACTION.md) and [measured collection results](DATA_COLLECTION_REPORT.md). The shared cleaning/feature/label pipeline and purged chronological research datasets are now implemented and run; see [dataset preparation](DATASET_PREPARATION.md). The daily LightGBM and baseline models are now trained and evaluated; the direction performance gate failed, so the bundle remains a research candidate. See [training results](MODEL_TRAINING_REPORT.md). Database schema, application and deployment remain subsequent stages.

User decisions: daily forecasting MVP first, live intraday forecasts later; no paid data APIs. Build, train, test and deploy through one reproducible pipeline using the same code and artifacts. Prepare the plan before implementing.

Deployment decision: Vercel production through GitHub Actions. New private repository: `codera647/StockPredictor`, initial branch `master`. Vercel projects created through authenticated CLI 64.0.2 in `codera647s-projects`: `stock-predictor` for the dashboard and `stock-predictor-api` for FastAPI. Both project owners and IDs were verified. Root directories, framework/build configuration and application deployment will be set with the runnable application; no deployment exists yet. Variable templates with fixed defaults and the complete configuration contract are in [.env.example](.env.example) and [ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md); the private `.env` is gitignored.

## Source documents

- `stock_prediction_architecture.md`: original architecture blueprint.
- `stock_prediction_architecture (1).md`: expanded blueprint, used as the working specification.

The expanded document retains the original design and adds section 13A: a multi-company live dashboard, server-side price streaming, separate intraday forecasting models, reconnect recovery, and live-specific acceptance checks. Its introduction and architecture diagram also incorporate intraday inference.

The blueprints describe requirements and proposed experiments. They do not supply market data, trained artifacts, measured accuracy, or evidence of profitable trading.

## Proposed first release

Start with the documented five-trading-session daily forecasting MVP. Keep interfaces compatible with the later intraday extension. The first pipeline must use actual fetched market data and produce a measured evaluation, persisted forecasts and a working dashboard.

The first usable release should let a user select a supported company, inspect its observed prices, view the stored median and estimated 80% forecast interval at an explicit target session, inspect direction probabilities and historical evaluation, and browse forecasts exactly as originally issued.

Use a small, explicitly documented engineering universe initially. Current surviving company examples must not be represented as a historically unbiased universe. Expand toward the blueprint's 50–100 securities only when data quality and coverage permit.

## Proposed stack

Follow the source documents: Python/FastAPI for the backend; pandas, Parquet and DuckDB for research; LightGBM with scikit-learn and statsmodels for experiments; PostgreSQL for application state and the forecast ledger; Next.js/React for the dashboard. Deploy dashboard and API as separate Vercel projects from the same repository. Run collection, training, evaluation and daily batch inference in GitHub Actions; persist archives in external S3-compatible storage and the forecast ledger in external PostgreSQL. Keep Docker Compose as an optional local verification tool; it is not the Vercel production deployment format. Add Redis when implementing shared live-price subscriptions. GitHub Actions is the initial scheduler; do not add a second mandatory Prefect/Celery service.

Python, Node.js, npm, Git and Docker executables are present on this workstation. Their versions, service availability and compatibility still need checking before installation. Provider capabilities and package APIs must be checked against current primary documentation during implementation.

## Free data sources and collection plan

Sources below were checked against primary documentation on 14 October 2026. Free API access and an open-source downloader are different from an unrestricted data redistribution license. The plan requires no paid data subscription, but public display rights must be established for the chosen feed before exposing it to other users.

### Required for the daily MVP

| Data | Chosen source | Fetch method | Cost/access | Important limit |
|---|---|---|---|---|
| Daily stock OHLCV | Alpaca Basic, explicitly selected historical SIP feed | REST `GET https://data.alpaca.markets/v2/stocks/bars`, `timeframe=1Day`; explicit start/end, adjustment and pagination | Free account and server-side API keys | Historical SIP requests must end at least 15 minutes ago; documented history since 2016, subject to symbol coverage; 200 historical calls/minute |
| Market/sector features | SPY and configured sector ETF daily bars from the same Alpaca SIP feed | Same bars endpoint and collection partitions | Same free access | Match stock feed, adjustment convention and cutoff; ETF coverage/history still require validation |
| Splits, dividends and other supported actions | Alpaca corporate actions; relevant issuer filings/announcements for reconciliation | `GET https://data.alpaca.markets/v1/corporate-actions`; paginate and archive original records | Corporate actions listed in Alpaca's free plan | Availability may lag announcement; complete historical coverage is not guaranteed. The original announcement API described history from April 2020, so do not assume actions match the bars' 2016 start |
| Current company directory | Nasdaq Trader symbol files | Download `nasdaqlisted.txt` and `otherlisted.txt` from its symbol directory; save daily snapshots | Public downloadable files | Current listings, not complete historical universe membership or delisting returns |
| Current ticker-to-issuer mapping | SEC | `https://www.sec.gov/files/company_tickers_exchange.json` | Free, no API key | CIK identifies the issuer, not each traded share class; retain separate local instrument IDs |
| Trading sessions, holidays and early closes | Open-source `exchange_calendars` | Generate the exchange schedule locally; persist calendar version | Apache-2.0 software; no paid API | Check exceptional closures against exchange notices; use exchange-local close times |

Price-feed choice: historical SIP is the proposed canonical daily feed. The Alpaca FAQ explicitly permits unsubscribed historical SIP queries whose `end` is at least 15 minutes old. A run 30 minutes after regular-session close fits that restriction, provided data passes readiness checks. Always request the feed explicitly; do not accept an account-dependent default or silently substitute IEX volume.

Raw OHLCV and action records remain immutable. Derive consistent split-aware features using the action policy valid for the snapshot, preserving forecast-date share basis. Never mix dividend-adjusted close with raw high/low. Retrieve-time historical actions and adjusted archives do not automatically prove original point-in-time availability.

The usable training window is the common verified coverage of prices, identities and relevant actions. Target the blueprint's 8-10 years where supported; shorten and disclose the window when action or identity history cannot be verified. Unexplained discontinuities block affected forecasts.

Primary references: [Alpaca plans](https://docs.alpaca.markets/us/docs/about-market-data-api), [historical SIP access](https://docs.alpaca.markets/us/docs/market-data-faq), [bars](https://docs.alpaca.markets/us/reference/stockbars), [free corporate-action listing](https://alpaca.markets/data), [corporate-action contract](https://docs.alpaca.markets/us/reference/corporateactions-1), [historical action announcement](https://alpaca.markets/blog/introducing-corporate-actions-api-announcements/), [Nasdaq directory definitions](https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs), [SEC mapping file](https://www.sec.gov/files/company_tickers_exchange.json), [exchange_calendars](https://github.com/gerrymanoim/exchange_calendars).

### Free sources for later evidence features

| Data | Source and fetch method | What we use | Constraint |
|---|---|---|---|
| Financial statements and filings | SEC EDGAR submissions, Company Facts JSON and original filing documents | Revenue, earnings, margins, debt, cash flow, accession and filing metadata | No API key; declared User-Agent; remain below 10 requests/second. Resolve original filings/amendments and usable timestamps before historical feature joins |
| Macro data | FRED/ALFRED API `fred/series/observations` with historical real-time/vintage parameters | Rates, Treasury yields, CPI and unemployment | Free API-key registration; series-specific use terms. Use vintages actually known at cutoff rather than today's revisions |
| News metadata and events | GDELT downloadable GKG/event files and documented query services | Article links, organizations, timestamps, themes and candidate event signals | Free/open GDELT data; updates every 15 minutes. Filter company relevance, deduplicate and record first-seen time. Historical retrieval is not a guarantee of complete finance-news coverage |
| Company announcements and future earnings schedules | Official company investor-relations releases, RSS and event pages where automated access is permitted | Announced event dates and source-linked releases | Company-specific parsers; begin snapshotting now. Today's calendar cannot reconstruct a historical schedule |

Use metadata and links for news evidence. GDELT access does not grant unrestricted rights to reproduce publishers' full articles. Download only relevant archive partitions; avoid relying on large paid BigQuery scans. Provider-specific current fundamentals, consensus earnings estimates and analyst transcripts are not required inputs to the first model.

References: [SEC APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), [SEC access rules](https://www.sec.gov/about/webmaster-frequently-asked-questions), [FRED API keys](https://fred.stlouisfed.org/docs/api/api_key.html), [ALFRED vintages](https://fred.stlouisfed.org/docs/api/fred/alfred.html), [observations endpoint](https://fred.stlouisfed.org/docs/api/fred/series_observations.html), [FRED series terms](https://fred.stlouisfed.org/docs/api/terms_of_use.html), [GDELT downloads](https://gdeltproject.org/data.html).

### Alternatives and architecture changes under the free-only rule

- Yahoo Finance through open-source `yfinance` is an optional personal-research alternative if Alpaca access is unavailable. It offers market-data fetching without a paid subscription, but is unofficial, subject to upstream changes/rate limits, and its README identifies Yahoo data access as intended for personal use. Its software license does not license public redistribution of Yahoo data. Treat it as a separate source/dataset; never silently splice its bars into Alpaca history. [Project documentation](https://github.com/ranaroussi/yfinance).
- Remove paid Databento dependencies. Maintain local instrument IDs, issuer mappings and prospectively archived symbol changes. Free current directories do not replace a complete point-in-time security master or valid delisting/merger outcomes. First evaluation is a disclosed fixed-universe research result; broad-universe strategy claims remain unavailable until those outcomes are covered.
- Remove Alpha Vantage as a mandatory collector. Its full daily history and daily-adjusted endpoint have premium restrictions, making them unsuitable foundations for the required free historical pipeline. [Endpoint documentation](https://www.alphavantage.co/documentation/).
- Do not require options, order books, historical consensus estimates, short-interest datasets or transcript vendors for the daily MVP.
- Later live streaming can use Alpaca's free IEX feed for up to 30 subscribed symbols. It is venue-only, not consolidated live SIP. Train/replay an IEX intraday bundle on matching inputs, or use a consistently delayed SIP design with delayed forecasts; neither option should be labeled consolidated real-time predictions. [Plan and feed limits](https://docs.alpaca.markets/us/docs/about-market-data-api).

## One reproducible build, train, test and deployment pipeline

```text
Pin dependencies and test the checked-out application commit
  -> Fetch real data and freeze raw snapshots
  -> Validate sessions, feeds, actions and availability
  -> Create versioned features, labels and dataset manifest
  -> Train five-session baseline and candidate bundles
  -> Evaluate chronological folds and untouched holdout
  -> Record model gate result and approved artifact
  -> Generate and persist forecasts from finalized inputs
  -> Build Vercel dashboard/API deployment outputs
  -> Test complete local data-to-dashboard flow
  -> Stage Vercel production builds, smoke-test, then promote
```

One pipeline means the same implementation and artifacts progress through these gates. It does not mean deploying an unvalidated model when evaluation fails. A failed candidate leaves the system with an honestly labeled baseline/research state or explicit forecast unavailability. Model promotion remains an authorized, audited action rather than an assistant bypass.

Make each step restartable from its manifest/checkpoint. Reuse cached immutable raw snapshots and existing dataset/model artifacts when inputs and configuration are unchanged. Avoid repeated backfills, unnecessary retraining and duplicate environment implementations. Persist logs and evaluation reports so failures can be diagnosed without rerunning every stage. App-only commits reuse the approved model; they do not trigger an unnecessary historical backfill and training cycle.

Local tests and Actions use the same shared Python feature engine and versioned datasets/model artifacts. Initial runs require a historical backfill; subsequent after-close runs incrementally fetch data and infer using approved weights. Retraining is an offline candidate cycle, not an operation performed for every daily forecast or page refresh. Vercel serves persisted forecasts and metrics; heavy training/backfills do not run inside a browser request or Vercel build.

Vercel is the chosen application host. The StockPredictor PostgreSQL project is provisioned through Neon in the connected Free organization: `hidden-butterfly-59546787`, production branch `br-lingering-shape-b8lz1937`, database `stock_predictor`, AWS US East 1, PostgreSQL 17. Runtime, pipeline and migration connections are saved and verified in the private local `.env`; restricted SQL roles are created, with specific table grants pending application schema implementation. Archives use the private Neon Object Storage bucket stock-predictor-artifacts on the production branch, with storage-only credentials and a 5 GB Free-plan allowance. All five S3 variables are saved in the private local .env. Free data does not guarantee unlimited free infrastructure or public redistribution. Vercel and Actions ephemeral disks are not durable archives. Check repository-plan Actions quotas before enabling recurring training.

### Planned GitHub Actions workflows

- Pull requests: dependency/type checks, meaningful data/feature tests, application builds and isolated integration checks; no production credentials for unreviewed/fork code.
- Initial data/model run: manually triggered backfill -> validation -> dataset -> training -> chronological evaluation -> authorized model approval -> persisted forecasts. Upload only non-secret artifacts and freeze their manifest/hashes.
- Production release: reviewed `master` commit, exact pinned actions/CLI, protected production environment. Pull project configuration, build each Vercel project, deploy staged production output, verify API/forecast/history/auth behavior, then promote. Configure one deployment owner so native Git deploys do not race the Actions release.
- Daily update: schedule candidate runs around both daylight-saving UTC offsets, then let the exchange calendar decide whether close +30 minutes has passed, including early closes. An idempotent readiness check skips holidays and already-completed cutoffs. Scheduled Actions can start late; show actual freshness and never promise execution at an exact clock time.
- Model refresh: explicit candidate job with mature labels and the same chronological gates; a failed candidate preserves the approved model. Never train on every deploy or price refresh.

Keep writer/migration/storage credentials only in the job steps that need them; Vercel runtime gets only its restricted database/session configuration. Configure production branch restrictions and release review checks when workflows are implemented. Application and model releases are linked through manifests, while daily forecast updates do not require redeploying the dashboard. Exact workflow files will be created with the runnable application, rather than adding active jobs that reference nonexistent scripts now.

## Implementation milestones

Dataset implementation contract: [DATASET_FEATURE_PLAN.md](DATASET_FEATURE_PLAN.md) defines required columns, cleaning/quarantine rules, the 31-feature core dictionary with three missing indicators, conditional sector features, labels, chronological splits and dataset acceptance checks.

| Milestone | Concrete work | Acceptance criteria |
|---|---|---|
| 1. Foundation and data | Backend configuration; PostgreSQL migrations; stable instrument identities; exchange calendar; provider adapter; immutable raw snapshots; bar revisions; corporate-action records; validation and readiness reports | No secrets in the browser; source/feed recorded; all pagination consumed; valid session and OHLCV checks; missing data and unverified actions block inference |
| 2. Reproducible dataset | Trailing feature engine; separate future-label generation; availability reconstruction policy; versioned Parquet and manifests; folds on unique session dates | No feature input after cutoff; no missing-session interpolation; label intervals purged; all fitting labels mature; all stocks for a session remain in one fold |
| 3. Five-session models | Zero-return baseline; linear and statistical comparisons; three LightGBM quantile models; calibrated direction classifier; experiment records; approved artifact loading | Chronological evaluation reports pinball loss, interval coverage/width, return errors and classification metrics; final holdout stays untouched by model selection; crossing repair is documented and evaluated |
| 4. Daily product | Read-only forecast API; company directory; watchlists; stock detail; endpoint/range chart; immutable history; data-health view; authentication and user-scoped state | Every forecast exposes cutoff, reference price, target, model/data versions and status; stale or unsupported inputs produce explicit states; refreshing a page does not retrain or generate forecasts |
| 5. Live extension | Backend market-data connection; shared subscription/cache layer; snapshot and sequence-numbered event gateway; heartbeat and reconnect handling; independent price timestamps | Price changes leave issued daily forecasts unchanged; reconnect restores a consistent snapshot; bounded queues protect slow clients; feed/delay and stale status remain visible |
| 6. Intraday models | Historical and live five-minute bar parity; readiness watermark; time-of-day trailing features; separate 30/60 trading-minute bundles; per-horizon chronological calibration | Only completed available bars enter inference; targets stay within the regular session; no future day totals enter morning features; missing bars block inference until recovered |
| 7. Paper trading | Executable-return companion; next-open long-only backtester; explicit costs/corporate actions; capital allocation; paper order/fill ledger | Forecast error and strategy returns reported separately; overlapping positions share finite capital; exposure limits and costs apply to actual trades; terminal outcomes covered before broad-universe claims |
| 8. Research extensions | Point-in-time filings/news/macro; reviewed extraction; feature-family ablations; evidence retrieval; bounded research assistant; 1/20-session bundles | Source availability and age retained; failed feed differs from no news; assistant reads deterministic numerical results and cannot promote models or execute trades; each horizon validated independently |

Live streaming and intraday modeling are separate deliverables. A daily model cannot become an intraday model by substituting the latest tick for the daily closing price.

## Initial repository layout

```text
backend/
  src/stock_predictor/
    collectors/
    schemas/
    data_quality/
    features/
    labels/
    datasets/
    training/
    evaluation/
    inference/
    api/
    storage/
  tests/
dashboard/
configs/
manifests/
infra/
```

Add `backtesting/` and `research_assistant/` when their milestones begin. Keep datasets, secrets and trained artifacts out of source control.

## First API deliverables

- `GET /health`: service readiness without exposing credentials.
- `GET /v1/instruments`: supported company directory.
- `GET /v1/instruments/{id}/bars`: validated observed price history with provenance.
- `GET /v1/forecasts?instrument_id=&horizon=`: persisted forecasts and explicit availability status.
- `GET /v1/forecasts/{prediction_id}`: the exact immutable issued forecast.
- `GET /v1/forecasts/history?instrument_id=`: original forecasts and separately attached outcomes.
- `GET /v1/models/{version}/metrics`: measured evaluation with split dates and baseline comparison.
- `GET /v1/data-health`: coverage, freshness and blocked/degraded reasons.

Add authenticated snapshot/stream endpoints during the live extension. Long-running ingestion, training and backtesting belong in workers and return tracked job IDs rather than running inside browser requests.

## Tests that protect the design

1. Reject late inputs, naive timestamps and incomplete required exchange-session windows.
2. Preserve raw bars and revision history; quarantine invalid OHLCV and unexplained actions.
3. Keep future labels out of feature snapshots; prevent label overlap across fit/evaluation cutoffs.
4. Make forecast insertion idempotent and immutable; attach realized outcomes separately.
5. Keep quote updates separate from forecast events and retain original cutoff/reference price.
6. Reject unsupported model/schema combinations and serve explicit stale/abstain states.
7. Verify regular-session holiday/early-close handling and suppress intraday targets beyond close.
8. Verify next-open entry, finite capital, splits/dividends and actual trade costs before enabling paper-policy evaluation.
9. Enforce user ownership of watchlists/portfolios and role checks on model promotion.

## Decisions and external inputs still needed

- First milestone preference: decided - daily MVP, followed by live intraday forecasts.
- Market scope: the documents assume US equities in USD. A different exchange requires a different calendar and suitable provider coverage.
- Market-data access: free Alpaca Basic account and keys, confirmed historical SIP access, action coverage and permitted display/redistribution. FRED free key is needed only when macro features begin. Credentials should be configured on the backend, not pasted into chat.
- Initial engineering universe and benchmark/sector mapping.
- Historical security and delisting coverage: required before claiming broad-universe results.
- Neon database/storage provisioning and local credentials: complete. Application schema/table grants, scoped Vercel/Actions secret configuration, archive pipeline implementation and authentication remain to be implemented. Deployment target is Vercel via GitHub Actions.

Use synthetic fixtures only for engineering tests or an explicitly labeled demonstration. Publish no real forecast, live quote, probability or accuracy score until its underlying data/model/evaluation exists.

## Completion definition

The daily MVP is complete when real permitted historical data flows through validation, a reproducible chronological evaluation and an approved model into an immutable forecast ledger, and the authenticated dashboard displays forecasts, uncertainty, versions, history and failure states correctly. The full blueprint additionally requires the live/intraday, paper-trading and evidence milestones above.
