# Data collection report

Collected on 14 October 2026. Status: complete with explicit external-source gaps. No model has been trained; these are extraction observations, not training examples.

| Data | Archived records | Coverage |
|---|---:|---|
| Stock/SPY daily bars | 138,210 | 50 stocks + SPY; 2016-01-04 through 2026-10-13; 2,710 sessions |
| Corporate actions | 1,995 | All returned action types; process-date window; verification pending |
| SEC Company Facts | 964,419 | 50 current issuer CIKs; original accession/date/unit preserved |
| SEC filing metadata | 9,836 | Matching financial/reporting forms in the requested historical window |
| Original SEC documents | 299 | Latest up to six primary documents per company |
| FRED/ALFRED observation/revision rows | 31,328 | Six series; historical window plus prior-year warm-up; bounded revision requests |
| FRED/ALFRED vintage-date records | 12,638 | Date-level vintage precision retained |
| GDELT GKG company matches | 268 | Four explicitly selected recent 15-minute archive partitions |
| Company announcement/event candidates | 386 | Official IR pages/RSS snapshots; links and dates need later event validation |
| nasdaq_listed directory records | 5,629 | Current snapshot, not historical membership |
| nasdaq_other directory records | 7,658 | Current snapshot, not historical membership |
| sec_mapping directory records | 10,447 | Current snapshot, not historical membership |

All six FRED series completed: DFEDTARU, DFF, DGS10, DGS2, CPIAUCSL and UNRATE. All 50 current SEC issuer collectors completed after accepting valid numeric-string CIK responses.

GDELT article search returned HTTP 429; three failed queries opened the circuit breaker. The remaining company queries were explicitly skipped. The independent GKG archive completed. Nineteen company IR URLs returned HTTP 403 and one timed out after retries; Apple and NVIDIA newsroom RSS feeds remained usable even though their IR pages were blocked. These failures are retained as missing coverage, never converted into zero-news features.

News is recent/prospective, not a complete ten-year news archive. Original filing downloads are an explicit subset. Current issuer CIKs and directories do not resolve historical reorganizations, ticker reuse or delisted companies. Historical symbol/issuer verification, corporate-action reconciliation and cutoff-consistent volume handling remain required before training. Provider daily volume includes extended-hours prints.

Private bucket: `stock-predictor-artifacts`. Stored objects: 1,528; total stored bytes: 182,963,763. Raw snapshots, Parquet data and manifests are in private object storage; local mirrors are gitignored.

Final full-extraction manifest:

```text
manifests/full-extraction/20261014T163648065170Z.json
SHA256: 495ef8a3f97b23d9ac834ff42a41a4c06de0f52854239bc6ec8950f73cfc2616
```

Local detailed reports: `artifacts/extraction/report.json`, `summary.json`, and `verification.json`. Corrections published superseding manifests and retained the original archives and reports.

All 12 Parquet tables matched row counts, local SHA256 and remote S3 metadata; 36 FRED snapshots passed secret-redaction audit; 33 tests and Ruff passed.

The same implementation is in the manual GitHub Actions workflow. This collection executed on the workstation against private remote storage. GitHub Actions still requires the six documented repository extraction secrets before a hosted run can start.
