# Environment variables

Status: configuration contract; application code does not exist yet. `.env` and `.env.example` contain the same 34 names, with 18 fixed URLs/settings/project IDs prefilled. The template's 16 credential/contact/service entries remain blank; the private `.env` now also contains the three verified Neon database connections. Other user-entered values are preserved. Only `.env.example` may be committed. Names are deliberate project conventions except where noted as provider-standard. Additional provider-specific variables will be added only if the selected implementation requires them.

## Where values belong

- Local development/testing: private root `.env`. Python jobs must explicitly load it; the future dashboard/API launchers must pass the appropriate subset into their processes. A root file is not automatically loaded by a Next.js application located in `dashboard/`.
- GitHub Actions: protected environment/repository Secrets for credentials; Variables for non-secret URLs and IDs. Filling a local file does not configure Actions.
- Vercel: each project's Production/Preview environment settings for its runtime subset. Filling a local file does not configure deployed functions. Preview uses separate data and credentials.
- Fixed provider endpoints, the explicit SIP daily feed, project IDs, CLI version and assigned Vercel domains are prefilled. Credentials remain blank in the supplied template. Never upload the private `.env` after entering real values.

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
| `NASDAQ_LISTED_SYMBOLS_URL` | `ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt` | Directory ingestion | Actions Variables |
| `NASDAQ_OTHER_LISTED_SYMBOLS_URL` | `ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt` | Directory ingestion | Actions Variables |
| `FRED_API_KEY` | Free registered FRED API key | Macro extension only | Actions Secrets |
| `FRED_API_BASE_URL` | `https://api.stlouisfed.org/fred` | Macro extension only | Actions Variables |
| `GDELT_DOC_API_URL` | `https://api.gdeltproject.org/api/v2/doc/doc` | News extension only | Actions Variables |
| `GDELT_GKG_BASE_URL` | `https://data.gdeltproject.org/gdeltv2` | News extension only | Actions Variables |

Nasdaq source definition: https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs . Its documented public FTP file addresses are prefilled; the collector must support that transport. HTTPS requests to the FTP host timed out during setup, so they are not assumed to work. The GDELT archive HTTPS last-update endpoint responded successfully. SEC, Nasdaq and GDELT do not require paid API keys for the planned public collection methods. The calendar library and public FinBERT artifact do not require a paid API key. GDELT document-search reference: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/ .

## Persistence variables

Database provisioned through the Neon connector in the connected Free organization: StockPredictor (`hidden-butterfly-59546787`), AWS US East 1, PostgreSQL 17, branch `production` (`br-lingering-shape-b8lz1937`), database `stock_predictor`. Compute is capped at 0.25 CU with the Free plan's default suspension policy. One database has separate roles for runtime, pipeline writes and schema migrations. The private local `.env` now contains pooled runtime/writer connections and a direct migration connection, preserving Neon's TLS options. All three authenticated successfully against the expected role/database. Object-storage provider selection remains separate. Vercel function disk and Actions runner disk are not durable archives.

Created SQL roles: `app_runtime` and `pipeline_writer`, with LOGIN but without superuser, CREATEDB, CREATEROLE, replication, BYPASSRLS or neon_superuser membership. They have database CONNECT and public-schema USAGE only. Their exact table grants will be added with the application schema: runtime forecasting/metric reads and permitted application-state operations, and pipeline ingestion/forecast writes. `schema_migrator` is the controlled administrative owner created with the project. The non-secret metadata and bootstrap SQL are in `infra/database/`; passwords are stored only in the private `.env`.

For subsequent setup, reuse this project and branch and the existing private database URLs. The migration URL was retrieved through the connector; restricted-role passwords were securely generated, the roles were created with SQL, and their pooled URLs were built from the same verified endpoint. These three names are our project's conventions. Keep credentials in the private `.env` and the appropriate scoped Vercel/Actions Secrets; this local setup has not populated remote deployment secrets.

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
| `NEXT_PUBLIC_APP_URL` | Assigned dashboard origin: `https://stock-predictor-beryl.vercel.app` | Deployed daily MVP | Vercel dashboard Config |
| `API_BASE_URL` | Assigned API origin: `https://stock-predictor-api-virid.vercel.app`; used by server-side dashboard requests | Deployed daily MVP | Vercel dashboard Config |
| `AUTH_SECRET` | Strong randomly generated application session-signing secret; planned shared verification contract between dashboard and API | Authenticated daily MVP | Vercel dashboard/API Secrets |
| `ALLOWED_ORIGINS` | Prefilled dashboard origin: `https://stock-predictor-beryl.vercel.app`; comma-separated if adding explicitly authorized local/preview origins | API configuration | Vercel API Config |

Use the canonical app URL for redirects/session origin; do not introduce a second redundant auth URL until the chosen auth library requires it. Provider OAuth clients, email delivery and external identity-provider keys are not mandatory in the current plan. If selected, document their additional variables before integration. No paid LLM/API key is required for the forecasting MVP.

## GitHub Actions to Vercel deployment variables

| Name | Purpose/value to supply | Required when | Destination |
|---|---|---|---|
| `VERCEL_TOKEN` | Vercel deployment token with access to the intended team/projects; configure it yourself in Actions | Deployment | Protected GitHub Actions Secret only |
| `VERCEL_ORG_ID` | Vercel account/team ID owning both application projects | Deployment | Actions Variable |
| `VERCEL_PROJECT_ID` | Dashboard Vercel project ID | Dashboard deployment | Actions Variable |
| `VERCEL_API_PROJECT_ID` | FastAPI Vercel project ID; workflow maps it to `VERCEL_PROJECT_ID` for the API step | API deployment | Actions Variable |
| `VERCEL_CLI_VERSION` | Exact CLI version pinned by the workflow; setup used `64.0.2` | CI deployment tooling | Actions Variable |
| `VERCEL_AUTOMATION_BYPASS_SECRET` | Automation bypass secret for smoke tests against protected staged deployments, if protection is enabled | Conditional deployment verification | Protected Actions Secret only |

These CI authentication values are not needed in the deployed application. The connected Codex plugins do not inject their credentials into GitHub Actions. Read IDs from project settings; never paste tokens into chat. `GITHUB_TOKEN`, `GITHUB_REPOSITORY`, `GITHUB_SHA` and runner context are supplied by Actions, so no GitHub personal access token is required by this pipeline and they are intentionally omitted from `.env`. Vercel-supplied runtime/system variables are also not manual configuration entries.

## Deferred live variable

| Name | Purpose | Required when |
|---|---|---|
| `REDIS_URL` | Private Redis connection URL for shared latest-price caching/pub-sub | Live extension, after its worker/cache host is selected |

Leave later/conditional entries blank until enabled. The future config loader must validate the variables required for the selected command/service and reject empty required values; it must not demand FRED, GDELT, Redis or stream credentials for the daily price-only MVP. Provider URLs should have documented defaults, with these names acting as explicit overrides once implemented.

## Current connector setup

- GitHub repository created: `https://github.com/codera647/StockPredictor`, private, initial branch `master`.
- New Vercel projects created and verified through CLI 64.0.2: `stock-predictor` (dashboard) and `stock-predictor-api` (FastAPI), in `codera647s-projects`.
- The connector's creation attempt returned HTTP 403; successful CLI login provided the authorized creation path. CLI identity was verified as `codera647` in the intended Hobby team. This does not change the connector's separate credentials.
- Projects currently have no root-directory/framework configuration or deployments. Set `dashboard/` + Next.js and `backend/` + FastAPI when the runnable application is prepared.
- No application, training run, model or live forecast has been deployed. This turn prepares configuration and project planning only.

Verified non-secret values now prefilled in `.env` and `.env.example` (copy the applicable values into GitHub Actions Variables):

| Name | Value |
|---|---|
| `VERCEL_ORG_ID` | `team_75Ikyytg6OGvW0iRj2fKq0Gg` |
| `VERCEL_PROJECT_ID` | `prj_lxU7mZ6zcq4GWdfLpU3CsPJOOkhc` |
| `VERCEL_API_PROJECT_ID` | `prj_ZiGiuZCVNKkCAyRzLjF5aCWwhSac` |
| `VERCEL_CLI_VERSION` | `64.0.2` |

CLI login authenticates this workstation. GitHub Actions still needs its own `VERCEL_TOKEN` stored directly in protected Actions Secrets; never copy or publish the CLI's credential file.

Both Vercel domain records are verified and belong to the intended project IDs. These are assigned project domains, not evidence of a live application: no deployment exists yet. Change application origin values together if a custom domain is configured later.

## User-specific values

- Daily collection: `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, `SEC_USER_AGENT`.
- Database: `DATABASE_URL`, `DATABASE_WRITE_URL`, `DATABASE_MIGRATION_URL` are configured and verified locally; keep these entries blank in the committed template.
- Object storage: `S3_ENDPOINT_URL`, `S3_REGION`, `S3_BUCKET_NAME`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`.
- Application signing: `AUTH_SECRET`.
- Actions deployment: `VERCEL_TOKEN`.
- Later/conditional: `FRED_API_KEY`, `VERCEL_AUTOMATION_BYPASS_SECRET`, `REDIS_URL`.

References: [Vercel environments](https://vercel.com/docs/environment-variables), [Vercel CLI deployment](https://vercel.com/docs/cli/deploy), [Vercel FastAPI support](https://vercel.com/docs/frameworks/backend/fastapi), [GitHub workflow triggers](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow).

