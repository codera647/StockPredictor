# Environment variables

Status: planned configuration contract; application code does not exist yet. `.env` and `.env.example` contain the same 34 names with blank values. Populate `.env` privately; only `.env.example` may be committed. Names are deliberate project conventions except where noted as provider-standard. Additional provider-specific variables will be added only if the selected implementation requires them.

## Where values belong

- Local development/testing: private root `.env`. Python jobs must explicitly load it; the future dashboard/API launchers must pass the appropriate subset into their processes. A root file is not automatically loaded by a Next.js application located in `dashboard/`.
- GitHub Actions: protected environment/repository Secrets for credentials; Variables for non-secret URLs and IDs. Filling a local file does not configure Actions.
- Vercel: each project's Production/Preview environment settings for its runtime subset. Filling a local file does not configure deployed functions. Preview uses separate data and credentials.
- All values remain blank in the supplied files, including known endpoint URLs and project metadata, as requested. No real token is stored or uploaded.

## Collection variables

| Name | Purpose/value to supply | Required when | Destination |
|---|---|---|---|
| `APCA_API_KEY_ID` | Alpaca free-account API key ID; provider-standard name | Daily MVP | Actions collection/inference jobs |
| `APCA_API_SECRET_KEY` | Corresponding Alpaca secret; provider-standard name | Daily MVP | Actions Secrets |
| `ALPACA_DATA_BASE_URL` | Historical data API base: `https://data.alpaca.markets` | Daily MVP | Actions Variables |
| `ALPACA_DATA_FEED` | Explicit daily feed: `sip`, subject to verified account access and delayed historical queries | Daily MVP | Actions Variables |
| `ALPACA_STREAM_URL` | Later free IEX stream: `wss://stream.data.alpaca.markets/v2/iex` | Live extension only | Future streaming worker |
| `SEC_API_BASE_URL` | SEC JSON API host: `https://data.sec.gov` | Initial identity checks / later filings | Actions Variables |
| `SEC_ARCHIVE_BASE_URL` | Original filing archive: `https://www.sec.gov/Archives/edgar/data` | Filings extension | Actions Variables |
| `SEC_TICKER_MAPPING_URL` | `https://www.sec.gov/files/company_tickers_exchange.json` | Identity mapping | Actions Variables |
| `SEC_USER_AGENT` | Identifying application name and a valid contact email in the SEC-required User-Agent format | Any SEC collection | Actions configuration; contact value kept private |
| `NASDAQ_LISTED_SYMBOLS_URL` | Nasdaq-listed symbol directory download address for `nasdaqlisted.txt` | Directory ingestion | Actions Variables |
| `NASDAQ_OTHER_LISTED_SYMBOLS_URL` | Other-exchange symbol directory download address for `otherlisted.txt` | Directory ingestion | Actions Variables |
| `FRED_API_KEY` | Free registered FRED API key | Macro extension only | Actions Secrets |
| `FRED_API_BASE_URL` | `https://api.stlouisfed.org/fred` | Macro extension only | Actions Variables |
| `GDELT_DOC_API_URL` | Document-search service URL from GDELT's current documentation | News extension only | Actions Variables |
| `GDELT_GKG_BASE_URL` | Download base for GDELT GKG historical/incremental files | News extension only | Actions Variables |

Nasdaq source definition: https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs . Preserve the full download URL, not a symbol-search webpage. Resolve the currently working documented transport during collector implementation. SEC, Nasdaq and GDELT do not require paid API keys for the planned public collection methods. The calendar library and public FinBERT artifact do not require a paid API key.

## Persistence variables

Choose external PostgreSQL and S3-compatible object storage with suitable free allowances before implementation. No provider is provisioned by this plan. Vercel function disk and Actions runner disk are not the durable data archive. Free allowance, account eligibility and data volume must be checked for the selected provider.

| Name | Purpose/value to supply | Required when | Destination |
|---|---|---|---|
| `DATABASE_URL` | TLS PostgreSQL connection URL for the API's restricted application role; pooled when appropriate | Deployed daily MVP | Vercel API Secret; dashboard only if auth requires DB access |
| `DATABASE_WRITE_URL` | TLS PostgreSQL connection URL for the ingestion/forecast writer role | Daily pipeline | Actions Secrets only |
| `DATABASE_MIGRATION_URL` | Direct PostgreSQL connection URL for controlled schema migrations | Initial setup / migrations | Protected Actions Secrets only |
| `S3_ENDPOINT_URL` | Selected object store's S3-compatible API endpoint | Persistent raw/dataset/model archives | Actions Variables |
| `S3_REGION` | Region identifier required by the selected provider's S3 API | Archive integration | Actions Variables |
| `S3_BUCKET_NAME` | Private archive bucket name | Archive integration | Actions Variables |
| `S3_ACCESS_KEY_ID` | Scoped archive access-key ID | Archive integration | Actions Secrets |
| `S3_SECRET_ACCESS_KEY` | Corresponding scoped archive secret | Archive integration | Actions Secrets |

The daily Vercel API reads persisted forecasts and metrics from PostgreSQL. It does not need archive-write credentials or the training dependencies. Serving file exports later may require a separate read-only storage credential; add that contract then rather than granting the live API archive-write access now.

## Application variables

| Name | Purpose/value to supply | Required when | Destination |
|---|---|---|---|
| `NEXT_PUBLIC_APP_URL` | Dashboard's canonical HTTPS URL; public, never a credential | Deployed daily MVP | Vercel dashboard Config |
| `API_BASE_URL` | FastAPI deployment's HTTPS origin; used by server-side dashboard requests | Deployed daily MVP | Vercel dashboard Config |
| `AUTH_SECRET` | Strong randomly generated application session-signing secret; planned shared verification contract between dashboard and API | Authenticated daily MVP | Vercel dashboard/API Secrets |
| `ALLOWED_ORIGINS` | Explicit allowed dashboard/local origins, comma-separated; never an unrestricted wildcard with credentialed requests | API configuration | Vercel API Config |

Use the canonical app URL for redirects/session origin; do not introduce a second redundant auth URL until the chosen auth library requires it. Provider OAuth clients, email delivery and external identity-provider keys are not mandatory in the current plan. If selected, document their additional variables before integration. No paid LLM/API key is required for the forecasting MVP.

## GitHub Actions to Vercel deployment variables

| Name | Purpose/value to supply | Required when | Destination |
|---|---|---|---|
| `VERCEL_TOKEN` | Vercel deployment token with access to the intended team/projects; configure it yourself in Actions | Deployment | Protected GitHub Actions Secret only |
| `VERCEL_ORG_ID` | Vercel account/team ID owning both application projects | Deployment | Actions Variable |
| `VERCEL_PROJECT_ID` | Dashboard Vercel project ID | Dashboard deployment | Actions Variable |
| `VERCEL_API_PROJECT_ID` | FastAPI Vercel project ID; workflow maps it to `VERCEL_PROJECT_ID` for the API step | API deployment | Actions Variable |
| `VERCEL_CLI_VERSION` | Reviewed exact CLI version pinned by the workflow | CI deployment tooling | Actions Variable |
| `VERCEL_AUTOMATION_BYPASS_SECRET` | Automation bypass secret for smoke tests against protected staged deployments, if protection is enabled | Conditional deployment verification | Protected Actions Secret only |

These CI authentication values are not needed in the deployed application. The connected Codex plugins do not inject their credentials into GitHub Actions. Read IDs from project settings; never paste tokens into chat. `GITHUB_TOKEN`, `GITHUB_REPOSITORY`, `GITHUB_SHA` and runner context are supplied by Actions, so no GitHub personal access token is required by this pipeline and they are intentionally omitted from `.env`. Vercel-supplied runtime/system variables are also not manual configuration entries.

## Deferred live variable

| Name | Purpose | Required when |
|---|---|---|
| `REDIS_URL` | Private Redis connection URL for shared latest-price caching/pub-sub | Live extension, after its worker/cache host is selected |

Leave later/conditional entries blank until enabled. The future config loader must validate the variables required for the selected command/service and reject empty required values; it must not demand FRED, GDELT, Redis or stream credentials for the daily price-only MVP. Provider URLs should have documented defaults, with these names acting as explicit overrides once implemented.

## Current connector setup

- GitHub repository created: `https://github.com/codera647/StockPredictor`, private, initial branch `master`.
- Requested new Vercel projects: `stock-predictor` (dashboard) and `stock-predictor-api` (FastAPI), in `codera647s-projects`.
- Vercel creation attempted through the connector; it returned HTTP 403 for insufficient access to that scope. Neither new Vercel project has been created or deployed.
- No application, training run, model or live forecast has been deployed. This turn prepares configuration and project planning only.

References: [Vercel environments](https://vercel.com/docs/environment-variables), [Vercel CLI deployment](https://vercel.com/docs/cli/deploy), [Vercel FastAPI support](https://vercel.com/docs/frameworks/backend/fastapi), [GitHub workflow triggers](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow).

