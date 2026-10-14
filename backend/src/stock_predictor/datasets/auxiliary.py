"""Clean evidence without pretending sparse current snapshots cover past forecasts."""

from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd


def canonical_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, parsed.query, ""))


def clean_evidence(
    name: str, source: pd.DataFrame, filings: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict]:
    data = source.copy()
    report = {"input_rows": len(data), "used_in_core_features": False}
    if data.empty:
        return data, {**report, "output_rows": 0, "status": "empty_source", "invalid_rows": 0}
    before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    report["identical_duplicates"] = before - len(data)
    valid = pd.Series(True, index=data.index)
    if name == "sec_filings":
        data["acceptance_at"] = pd.to_datetime(data.acceptance_at, utc=True, errors="coerce")
        data["available_at_reconstructed"] = data.acceptance_at
        valid &= (
            data.issuer_cik.notna() & data.accession_number.notna() & data.acceptance_at.notna()
        )
        data["availability_policy"] = "edgar_acceptance_metadata_not_verified_first_receive"
        key = ["issuer_cik", "accession_number"]
    elif name == "sec_facts":
        data["value"] = pd.to_numeric(data.value, errors="coerce")
        end = pd.to_datetime(data.period_end, errors="coerce")
        filed = pd.to_datetime(data.filed_date, errors="coerce", utc=True)
        valid &= (
            np.isfinite(data.value)
            & end.notna()
            & filed.notna()
            & data.unit.notna()
            & data.accession_number.notna()
        )
        # Date-only filed facts are conservatively usable the following UTC day.
        availability = filed + pd.Timedelta(days=1)
        if filings is not None and not filings.empty:
            accepted = (
                filings[filings.quality_valid][["issuer_cik", "accession_number", "acceptance_at"]]
                .drop_duplicates(["issuer_cik", "accession_number"])
                .set_index(["issuer_cik", "accession_number"])
                .acceptance_at
            )
            keys = pd.MultiIndex.from_frame(data[["issuer_cik", "accession_number"]])
            exact = pd.Series(accepted.reindex(keys).to_numpy(), index=data.index)
            availability = pd.to_datetime(exact, utc=True).fillna(availability)
        data["available_at_reconstructed"] = availability
        data["availability_policy"] = "acceptance_if_matched_otherwise_filed_date_plus_one_day"
        valid &= availability.ge(pd.to_datetime(data.period_end, utc=True, errors="coerce"))
        key = [
            "issuer_cik",
            "taxonomy",
            "concept",
            "unit",
            "period_start",
            "period_end",
            "accession_number",
        ]
        # A frame annotation is not a new economic fact. Conflicting values remain quarantined.
        variants = data.groupby(key, dropna=False).value.transform("nunique")
        valid &= variants.le(1)
        report["conflicting_value_rows"] = int(variants.gt(1).sum())
    elif name == "macro_observations":
        data["value"] = pd.to_numeric(data.value, errors="coerce")
        observed = pd.to_datetime(data.observation_date, utc=True, errors="coerce")
        start = pd.to_datetime(data.realtime_start, utc=True, errors="coerce")
        # realtime_end may be 9999-12-31: preserve its ISO value, not a truncated timestamp.
        valid &= observed.notna() & start.notna() & data.realtime_end.ge(data.realtime_start)
        valid &= data.value.isna() | np.isfinite(data.value)
        data["value_missing"] = data.value.isna()
        data["available_at_reconstructed"] = start + pd.Timedelta(days=1)
        data["availability_policy"] = "alfred_date_vintage_plus_one_day_not_intraday_release_time"
        key = ["series_id", "observation_date", "realtime_start", "realtime_end"]
        variants = data.groupby(key, dropna=False).value.transform("nunique")
        valid &= variants.le(1)
        report["conflicting_value_rows"] = int(variants.gt(1).sum())
    elif name == "macro_vintages":
        valid &= pd.to_datetime(data.vintage_date, errors="coerce").notna() & data.series_id.notna()
        key = ["series_id", "vintage_date"]
    elif name in ("company_announcements", "news_gkg", "news_articles"):
        data["canonical_url"] = data.url.map(canonical_url)
        data["available_at_observed"] = pd.to_datetime(
            data.first_observed_at, utc=True, errors="coerce"
        )
        valid &= data.canonical_url.notna() & data.available_at_observed.notna()
        if "publication_at" in data:
            data["publication_at"] = pd.to_datetime(data.publication_at, utc=True, errors="coerce")
            valid &= data.publication_at.isna() | data.publication_at.le(data.available_at_observed)
        data["availability_policy"] = "first_observed_snapshot_only_no_backdated_historical_join"
        key = ["symbol", "canonical_url"] if "symbol" in data else ["canonical_url"]
        data = data.assign(quality_valid=valid).sort_values("available_at_observed")
    else:
        key = ["id"] if "id" in data else ["issuer_cik", "symbol"]
    if "quality_valid" not in data:
        data["quality_valid"] = valid
    data["quality_status"] = np.where(data.quality_valid, "typed_research_evidence", "quarantined")
    report["invalid_rows"] = int((~data.quality_valid).sum())
    before = len(data)
    # Invalid variants are retained, so their disagreements cannot silently disappear.
    accepted = data[data.quality_valid].drop_duplicates(key, keep="first")
    data = pd.concat([accepted, data[~data.quality_valid]], ignore_index=True)
    report.update(
        output_rows=len(data),
        semantic_duplicates=before - len(data),
        status="cleaned_evidence_only",
    )
    return data, report
