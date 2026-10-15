# SEC fundamentals and historical news experiment

This implements the next experiment from the [Unidata dataset review](https://unidata.pro/blog/stock-market-datasets-ml/): add company fundamentals and dated historical news to the original price inputs, then measure their separate contributions. It keeps the five-session UP/FLAT/DOWN labels and chronological boundaries. It does not establish a 90% accuracy claim or replace the existing bundle.

The completed run and its measured comparisons are documented in [FUNDAMENTAL_NEWS_REPORT.md](FUNDAMENTAL_NEWS_REPORT.md).

## Run the shared pipeline

```powershell
uv run --frozen stock-predictor-fundamental-news
```

On a fresh GitHub Actions runner, the manual `fundamental-news.yml` workflow supplies the archived dataset and baseline manifests. It uses the existing APCA/S3 credentials; no new paid API, token or environment variable is needed. SEC facts/filings come from the existing checksummed daily collection. Public Hugging Face downloads require no account. `--collect-only` prepares news independently; `--no-upload` explicitly uses local archives and does not claim cloud completion. Completed source/model checkpoints are reused with checksum verification.

## News provenance and collection

The [authors' FNSPID repository](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) links the original large CSV. The default experiment uses the [public Parquet conversion](https://huggingface.co/datasets/sabareesh88/FNSPID_nasdaq), pinned to `7551d1738dd42480f8b7284e29b94ca361394f46`. `configs/fnspid-parquet.files.json` records all 48 shard paths, sizes and provider LFS hashes.

The original 23-GB CSV download was too slow in this environment. Parquet metadata allows skipping row groups whose symbol ranges cannot match the existing universe, including unlabeled groups. Adjacent headline/date/symbol/URL/publisher columns are fetched together without downloading article bodies. Each HTTP range validates its offset, length and content hash; verified ranges and filtered shard checkpoints support resumption. Reports record metadata rows, actual read rows, downloaded bytes, per-symbol retained counts and all range hashes. Immutable provider revision and range hashes provide provenance; the full original CSV and full Parquet-file LFS digests are **not** claimed verified by this projection. The third-party conversion has not been independently proven identical to the authors' CSV.

Rows need a known stock symbol, valid date, headline and canonical URL. Deduplicate stories within each stock by URL, then by timestamp/headline; retain legitimate cross-stock links. No malformed source rows are silently skipped by the original CSV fallback. The current default uses typed Parquet columns. Article bodies, summaries and upstream scraper code are not copied into our feature bundle.

The authors' July 2025 notice releases research/commercial rights, but older restrictive wording remains. This experiment stays research-only and does not resolve third-party content rights or approve redistribution of raw news.

## Features and availability

| Variant | Price inputs | SEC additions | News additions | Total |
|---|---:|---:|---:|---:|
| price_only | 34 | 0 | 0 | 34 |
| price_sec | 34 | 28 | 0 | 62 |
| price_news | 34 | 0 | 18 | 52 |
| price_sec_news | 34 | 28 | 18 | 80 |

SEC additions are fourteen numeric inputs with fourteen missing flags: liabilities/assets, cash/assets, current ratio, equity/assets, annual net margin, annual ROA, annual operating cash/assets, annual revenue growth, scaled annual income change, financial-release age, five-session filing/8-K/earnings-8-K counts, and age since the last 8-K.

Replay only facts whose reconstructed availability is at or before the forecast cutoff. Within a reporting period, retain only revisions available then, using fixed concept preferences. Instant balance-sheet facts and annual 300–400-day flow facts have separate rules. Ratios require matching period ends; flow/flow ratios also require matching starts. Annual growth compares periods 300–430 days apart using versions available at cutoff. Stale periods over 550 days old and unsupported ratios remain missing. Filing events use acceptance times and exchange-session windows. Recent-submission snapshots may truncate older history, so pre-first-observed filing counts remain unknown. Existing current-CIK mapping and survivorship assumptions remain research limitations.

News additions are nine numeric inputs and nine missing flags: one-/five-session counts, one-/five-session lexical sentiment means, five-session sentiment dispersion, positive/negative/earnings headline fractions, and headline age capped at 365 days. The vocabulary and three-token negation rule are fixed before evaluation, require no labels and are a simple lexical proxy, not FinBERT or a learned financial sentiment model.

Reported news publication times are delayed by a full 24 hours to reduce date/time-zone ambiguity. Only headlines usable by the forecast cutoff enter trailing windows. The declared research window is 2016-01-04 through 2023-12-15; before a stock's first usable headline and after the window, news inputs remain missing. Active-window flags are evaluation metadata, not extra model inputs, and do not prove complete coverage. Missing post-2023 news is never represented as a verified absence of events. Contemporary GDELT/IR snapshots are not backdated into historical training.

## Training and comparison

Each feature variant fits regularized logistic candidates (`C=0.1,1`) and a bounded CatBoost classifier. Training-only imputation preserves entirely missing columns; scaling uses training statistics. Earlier chronological validation selects logistic regularization, tree stopping and the family by probability log loss. Two later historical folds evaluate frozen earlier-trained weights. The existing validation block also provides a reused research comparison, not a fresh test.

Report both families, balanced accuracy, per-class/confusion/probability metrics, and matched-subset results for news-window dates. Compare each selected feature variant with a price-only family selected on the same earlier dates. The baseline gate requires lower log loss and noninferior accuracy in both earlier folds. A favorable gate alone does not establish production readiness or 90% accuracy. The original logistic bundle is also evaluated on the same validation rows for continuity. The old final-test block is never read.

Native/JSON model reload probabilities must agree before outputs are accepted. Immutable model weights, feature tables, row-level predictions, code/dependency/config fingerprints and source manifests make the experiment reviewable. Tests cover future revisions, matching financial periods, after-cutoff filings, microsecond/nanosecond timestamp handling, news delay/missing windows, cross-stock deduplication, resumable bytes, multi-line CSV chunks, Parquet filtering, model reloads, unchanged labels, test exclusion and cache reuse.
