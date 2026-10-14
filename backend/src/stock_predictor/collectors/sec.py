import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from urllib.parse import quote

from stock_predictor.config import PipelineError, Settings
from stock_predictor.storage.archive import Archive, digest, json_bytes

from .cache import CachedSource
from .http import SourceClient

LOG = logging.getLogger(__name__)
FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A", "20-F", "20-F/A", "6-K"}


def submission_rows(payload: dict) -> list[dict]:
    data = payload.get("filings", {}).get("recent", payload)
    required = ("accessionNumber", "filingDate", "form", "primaryDocument")
    if not all(isinstance(data.get(name), list) for name in required):
        raise PipelineError("SEC submissions required arrays are missing")
    count = len(data["accessionNumber"])
    if any(len(data[name]) != count for name in required):
        raise PipelineError("SEC submissions arrays have inconsistent lengths")
    return [
        {
            name: values[i]
            for name, values in data.items()
            if isinstance(values, list) and len(values) == count
        }
        for i in range(count)
    ]


def fact_rows(payload: dict, start: date, end: date, ref: dict, envelope: dict) -> list[dict]:
    cik = payload.get("cik")
    if (
        not isinstance(payload.get("facts"), dict)
        or isinstance(cik, bool)
        or not isinstance(cik, (int, str))
        or not str(cik).isdigit()
        or not 1 <= len(str(cik)) <= 10
    ):
        raise PipelineError("SEC Company Facts schema is invalid")
    rows = []
    for taxonomy, concepts in payload["facts"].items():
        for concept, details in concepts.items():
            for unit, facts in details.get("units", {}).items():
                for fact in facts:
                    filed = fact.get("filed")
                    if not filed or not start.isoformat() <= filed <= end.isoformat():
                        continue
                    rows.append(
                        {
                            "issuer_cik": str(payload["cik"]).zfill(10),
                            "taxonomy": taxonomy,
                            "concept": concept,
                            "unit": unit,
                            "value": fact.get("val"),
                            "period_start": fact.get("start"),
                            "period_end": fact.get("end"),
                            "filed_date": filed,
                            "accession_number": fact.get("accn"),
                            "form": fact.get("form"),
                            "fiscal_year": fact.get("fy"),
                            "fiscal_period": fact.get("fp"),
                            "frame": fact.get("frame"),
                            "retrieved_at": envelope["retrieved_at"],
                            "available_at": envelope["retrieved_at"],
                            "availability_method": "observed_retrieval_requires_accession_join",
                            "record_version": digest(json_bytes(fact)),
                            "raw_snapshot_key": ref["key"],
                        }
                    )
    return rows


def filing_url(base: str, cik: str, accession: str, document: str) -> str:
    if not cik.isdigit() or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
        raise PipelineError("SEC filing identity is invalid")
    if not document or "/" in document or "\\" in document or document in (".", ".."):
        raise PipelineError("SEC primary-document filename is unsafe")
    return f"{base}/{int(cik)}/{accession.replace('-', '')}/{quote(document, safe='-_.')}"


def collect_sec(
    settings: Settings,
    archive: Archive,
    instruments: list[dict],
    start: date,
    end: date,
    *,
    scope: str,
    refresh: bool,
    documents_per_company: int,
    workers: int = 1,
) -> dict:
    if "@" not in settings.sec_user_agent or "example.com" in settings.sec_user_agent:
        raise PipelineError("SEC_USER_AGENT must contain a real contact email")
    if workers > 1:
        # Four clients at <=2 requests/second each stay below SEC's aggregate 10/s.
        # Keep share classes from the same issuer together so requests are not duplicated.
        groups = {}
        for instrument in instruments:
            groups.setdefault(instrument.get("issuer_cik", instrument["symbol"]), []).append(
                instrument
            )
        combined = {
            "source": "sec",
            "companies": [],
            "facts": [],
            "filings": [],
            "company_metadata": [],
            "raw": [],
        }

        def collect_group(group):
            return collect_sec(
                settings,
                archive,
                group,
                start,
                end,
                scope=scope,
                refresh=refresh,
                documents_per_company=documents_per_company,
                workers=1,
            )

        with ThreadPoolExecutor(max_workers=min(workers, 4)) as pool:
            for result in pool.map(collect_group, groups.values()):
                for key in ("companies", "facts", "filings", "company_metadata", "raw"):
                    combined[key].extend(result[key])
        return combined
    cache = CachedSource(
        SourceClient(
            "sec",
            archive,
            headers={"User-Agent": settings.sec_user_agent},
            interval=0.5,
        ),
        archive,
        refresh=refresh,
    )
    results, facts, filings, metadata, refs = [], [], [], [], []
    seen_ciks = set()
    try:
        for instrument in instruments:
            if instrument["kind"] != "stock":
                continue
            symbol, cik = instrument["symbol"], instrument.get("issuer_cik")
            if not cik:
                results.append(
                    {"symbol": symbol, "status": "failed", "reason": "unresolved_issuer_cik"}
                )
                continue
            if cik in seen_ciks:
                results.append({"symbol": symbol, "issuer_cik": cik, "status": "shared_issuer"})
                continue
            seen_ciks.add(cik)
            LOG.info("SEC facts and filing history: %s", symbol)
            result = {"symbol": symbol, "issuer_cik": cik, "status": "complete", "issues": []}
            try:
                payload, ref, envelope = cache.json(
                    f"{settings.sec_api_base}/api/xbrl/companyfacts/CIK{cik}.json",
                    revision_scope=scope,
                )
                refs.append(ref)
                observations = fact_rows(payload, start, end, ref, envelope)
                facts.extend(observations)
                result["facts"] = len(observations)
            except PipelineError as exc:
                result["issues"].append(str(exc))
            company_filings = []
            try:
                payload, ref, envelope = cache.json(
                    f"{settings.sec_api_base}/submissions/CIK{cik}.json",
                    revision_scope=scope,
                )
                refs.append(ref)
                metadata.append(
                    {
                        "symbol": symbol,
                        "issuer_cik": cik,
                        "name": payload.get("name"),
                        "website": payload.get("website"),
                        "investor_website": payload.get("investorWebsite"),
                        "sic": payload.get("sic"),
                        "sic_description": payload.get("sicDescription"),
                        "metadata_scope": "current_snapshot_only",
                        "raw_snapshot_key": ref["key"],
                    }
                )
                batches = [(submission_rows(payload), ref, envelope)]
                for shard in payload.get("filings", {}).get("files", []):
                    if (
                        shard["filingTo"] < start.isoformat()
                        or shard["filingFrom"] > end.isoformat()
                    ):
                        continue
                    name = shard["name"]
                    if not re.fullmatch(r"CIK\d+-submissions-\d+\.json", name):
                        raise PipelineError("Unexpected SEC historical submissions filename")
                    older, old_ref, old_envelope = cache.json(
                        f"{settings.sec_api_base}/submissions/{name}",
                        revision_scope=scope,
                    )
                    refs.append(old_ref)
                    batches.append((submission_rows(older), old_ref, old_envelope))
                seen_accessions = set()
                for entries, source_ref, source_envelope in batches:
                    for filing in entries:
                        if (
                            filing["form"] not in FORMS
                            or not start.isoformat() <= filing["filingDate"] <= end.isoformat()
                        ):
                            continue
                        accession = filing["accessionNumber"]
                        if accession in seen_accessions:
                            continue
                        seen_accessions.add(accession)
                        company_filings.append(
                            {
                                "symbol": symbol,
                                "issuer_cik": cik,
                                "accession_number": accession,
                                "filing_date": filing["filingDate"],
                                "acceptance_at": filing.get("acceptanceDateTime"),
                                "report_date": filing.get("reportDate"),
                                "form": filing["form"],
                                "primary_document": filing["primaryDocument"],
                                "items": filing.get("items"),
                                "is_xbrl": filing.get("isXBRL"),
                                "url": filing_url(
                                    settings.sec_archive_base,
                                    cik,
                                    accession,
                                    filing["primaryDocument"],
                                ),
                                "retrieved_at": source_envelope["retrieved_at"],
                                "raw_snapshot_key": source_ref["key"],
                                "document_status": "metadata_only",
                            }
                        )
                company_filings.sort(
                    key=lambda r: (r["filing_date"], r["accession_number"]), reverse=True
                )
                # Explicit, configurable document scope; the full metadata index remains archived.
                selected = company_filings[:documents_per_company]
                for filing in selected:
                    try:
                        doc_ref, _, _ = cache.get(filing["url"])
                        refs.append(doc_ref)
                        filing["document_status"] = "archived"
                        filing["document_snapshot_key"] = doc_ref["key"]
                    except PipelineError as exc:
                        filing["document_status"] = "failed"
                        result["issues"].append(str(exc))
                filings.extend(company_filings)
                result.update(
                    {
                        "filing_metadata": len(company_filings),
                        "selected_documents": len(selected),
                        "archived_documents": sum(
                            f["document_status"] == "archived" for f in selected
                        ),
                        "document_scope": "latest_primary_documents_within_requested_window",
                        "documents_per_company": documents_per_company,
                    }
                )
            except (PipelineError, KeyError) as exc:
                result["issues"].append(
                    str(exc) if isinstance(exc, PipelineError) else "SEC history schema incomplete"
                )
            if result["issues"]:
                result["status"] = (
                    "complete_with_issues" if result.get("facts") or company_filings else "failed"
                )
            results.append(result)
    finally:
        cache.close()
    return {
        "source": "sec",
        "companies": results,
        "facts": facts,
        "filings": filings,
        "company_metadata": metadata,
        "raw": refs,
    }
