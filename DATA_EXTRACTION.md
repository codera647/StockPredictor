# Data extraction pipeline

This is the first implemented stage of the shared build/train/deploy pipeline. It
collects actual source data into the private Neon S3 archive and creates versioned
Parquet staging files, manifests and coverage reports. It does not train or deploy
a model.

## Collection scope

`configs/universe.daily.json` freezes 50 stocks and SPY, with deterministic instrument
IDs and a disclosed current-survivor selection. The IDs are provisional until
historical symbol/security verification. Current SEC CIK mappings identify issuers,
not share classes or historical symbol intervals.

- Alpaca historical daily bars: explicit SIP, USD, unadjusted `1Day`, ascending order,
  and `asof=-` to disable implicit symbol renaming. All pagination is consumed.
- Alpaca corporate actions: all supported action types, including early/incomplete
  records using `data_quality=all`; query intervals are **process dates**, not ex-dates.
- Current Nasdaq listed/other-listed directory snapshots and SEC ticker/issuer mapping.
- Versioned XNYS session calendar, including holidays, daylight saving and early closes.

The default historical request starts on 2016-01-04 and ends at the latest session
whose regular close was at least 30 minutes ago. Every SIP request has an explicit
end at least 16 minutes before retrieval. The requested history is not a promise of
provider coverage.

Legacy Nasdaq FTP defaults are fetched from the same provider's verified HTTPS
copies under `https://www.nasdaqtrader.com/dynamic/SymDir/`. Both configured and
actual fetched URLs are recorded. This avoids blocked FTP ports on hosted runners.

## Run the same implementation locally or in Actions

Python 3.13 and uv 0.13.1 are used. Dependency versions and platform wheel hashes are
frozen in `uv.lock`.

```powershell
python -m pip install uv==0.13.1
uv sync --frozen
uv run --frozen stock-predictor extract --start 2016-01-04
```

Default execution requires these private `.env` values: `APCA_API_KEY_ID`,
`APCA_API_SECRET_KEY`, `S3_ENDPOINT_URL`, `S3_REGION`, `S3_BUCKET_NAME`,
`S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`. SEC access requires a real contact in
`SEC_USER_AGENT`. Database, Redis, FRED and Vercel tokens are not required by this stage.

An explicit smaller interval/subset uses the same code and storage contract:

```powershell
uv run --frozen stock-predictor extract --start 2024-01-01 --end 2024-12-31 --symbols AAPL,SPY
```

In GitHub, open **Actions â†’ Data extraction â†’ Run workflow** on `master`. Set these
five repository Actions secrets first:

- `APCA_API_KEY_ID`
- `APCA_API_SECRET_KEY`
- `SEC_USER_AGENT`
- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`

Public endpoint, region and bucket defaults are already in the workflow. The local
`.env` is never uploaded or committed. Inputs pass through environment variables and
quoted arguments rather than being interpolated into shell code. Credentials are
provided only to the extraction step, after checks pass. There is no active schedule
yet; manual backfill is the first stage of the eventual daily pipeline.

## Checkpoints, revisions and private storage

Requests are partitioned by year. Each fingerprint includes source endpoint,
collector version and complete request parameters, including symbols/feed/adjustment.
Every original response is stored as exact base64 bytes in a compressed envelope
with retrieval time and a checksum. Request authentication headers and SEC contact
headers are excluded from envelopes and logs.

Each page is confirmed in S3 before its checkpoint advances. Retrying an interrupted
request starts from the last saved pagination cursor. Completed requests reuse their
archived pages, verifying checksums; `--refresh` explicitly captures new revisions
without modifying old snapshots. The global Actions concurrency group prevents two
remote extraction jobs from changing checkpoints simultaneously. Do not run a local
writer concurrently with the remote job.

Private archive layout:

```text
raw/{source}/{hash}.json.gz
checkpoints/extraction/v1/{request_hash}.json
staging/{bars,actions,sessions}/{hash}.parquet
manifests/universes/{hash}.json
manifests/extraction/{run_id}.json
reports/extraction/{run_id}.json
```

Local mirrors live in ignored `data/archive/`; concise reports are also written to
ignored `artifacts/extraction/`. Actions attaches only reports. Raw payloads, datasets,
keys and `.env` stay out of source control. S3 objects use path-style addressing and
SigV4. Extraction does not modify PostgreSQL.

## Report and exit status

The report includes coverage per stock/year, missing exchange sessions, invalid
OHLCV, identical duplicates, conflicting revisions, action counts/types and directory
failures. Valid zero volume stays zero. Invalid observations are quarantined;
unrecovered sessions are never interpolated. Every observation retains its raw
snapshot/hash and source timestamp.

- Exit `0`: extraction complete without mechanical coverage/directory issues.
- Exit `2`: extraction complete with explicit issues; reports and archives remain available.
- Exit `1`: authentication/network/schema/storage/configuration failure; earlier page
  checkpoints remain available for retry.

`training_ready` is always **false** in this extraction milestone. Action verification,
historical identities, availability reconstruction and feature generation are later
dataset stages. No empty action response establishes that an instrument never split.

## Provider timing finding that affects the feature stage

Alpaca daily-bar volume can be updated by extended-hours trades. Consequently the
retrospective provider day aggregate is not a regular-session-only volume observation
available at close +30 minutes. Staging records explicitly carry
`volume_scope=provider_daily_includes_extended_hours` and use observed retrieval time
for availability; the extractor does not invent earlier availability timestamps.

Before fitting the planned volume features, the dataset stage must implement and
evaluate a cutoff-consistent convention: verified regular-session aggregation, a
conservative lagged full-day volume policy, or a later forecast cutoff. Historical
provider corrections remain a separate replay limitation. This choice must also be
used by the daily inference collector.

Primary contracts: [bars](https://docs.alpaca.markets/us/reference/stockbars),
[corporate actions](https://docs.alpaca.markets/us/reference/corporateactions-1),
[SIP access and aggregation rules](https://docs.alpaca.markets/us/docs/market-data-faq),
[exchange calendar](https://github.com/gerrymanoim/exchange_calendars).

## Verification

```powershell
uv run --frozen ruff check backend
uv run --frozen pytest -q
```

Tests cover pagination across symbols, empty pages with cursors, interrupted recovery,
looping cursors, retry/error archives, credential-safe failures, checkpoint ordering,
checksum corruption, immutable collisions, missing sessions/revisions, early closes,
free-SIP timing, invalid OHLCV, zero volume and a complete manifest-to-Parquet flow.

Initial verification on 2026-10-14: all 20 tests and Ruff checks passed. The live
directory extraction archived 5,629 Nasdaq-listed records, 7,658 other-listed records
and 10,447 SEC mapping records. Private S3 uploads and read-back checks succeeded.
The first actual historical bars request returned HTTP 401 for the configured Alpaca
key pair. No price history or training dataset is claimed until that credential issue
is corrected and the historical extraction completes.
