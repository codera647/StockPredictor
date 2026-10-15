import re

import numpy as np
import pandas as pd
from stock_predictor.config import PipelineError

SEC_NUMERIC = [
    "sec_liabilities_assets",
    "sec_cash_assets",
    "sec_current_ratio",
    "sec_equity_assets",
    "sec_annual_net_margin",
    "sec_annual_roa",
    "sec_annual_operating_cash_assets",
    "sec_annual_revenue_growth",
    "sec_annual_income_change_scaled",
    "sec_days_since_financial",
    "sec_filings_5d",
    "sec_8k_5d",
    "sec_earnings_8k_5d",
    "sec_days_since_8k",
]
NEWS_NUMERIC = [
    "news_count_1d",
    "news_count_5d",
    "news_sentiment_1d",
    "news_sentiment_5d",
    "news_sentiment_std_5d",
    "news_positive_fraction_5d",
    "news_negative_fraction_5d",
    "news_earnings_fraction_5d",
    "news_days_since_headline",
]
SEC_COLUMNS = SEC_NUMERIC + [n + "_missing" for n in SEC_NUMERIC]
NEWS_COLUMNS = NEWS_NUMERIC + [n + "_missing" for n in NEWS_NUMERIC]
CONCEPTS = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_cash": ["NetCashProvidedByUsedInOperatingActivities"],
}
ANNUAL = {"revenue", "income", "operating_cash"}
LOOKUP = {
    concept: (name, rank)
    for name, values in CONCEPTS.items()
    for rank, concept in enumerate(values)
}
POSITIVE = set(
    (
        "beat beats exceeded outperform upgrade upgrades raised raises growth gain gains "
        "profit profits record surge strong bullish approve approved"
    ).split()
)
NEGATIVE = set(
    (
        "miss misses missed below underperform downgrade downgrades lowered lowers decline "
        "drop drops loss losses lawsuit bankruptcy weak bearish fraud cut cuts"
    ).split()
)


def sentiment(headline):
    """Frozen lexical headline score; no fitted model or outcome labels are involved."""
    tokens = re.findall(r"[a-z]+", headline.lower())
    score = 0
    for i, token in enumerate(tokens):
        value = int(token in POSITIVE) - int(token in NEGATIVE)
        if any(t in {"not", "no", "never"} for t in tokens[max(0, i - 3) : i]):
            value = -value
        score += value
    return float(np.clip(score / np.sqrt(max(len(tokens), 1)), -1, 1))


def finish(frame, names):
    for name in names:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        frame[name + "_missing"] = frame[name].isna().astype("int8")
    return frame


def fundamentals(facts, filings, companies, index, sessions, config):
    facts = facts[
        facts.quality_valid
        & facts.taxonomy.eq("us-gaap")
        & facts.unit.eq("USD")
        & facts.concept.isin(LOOKUP)
    ].copy()
    for name in ("available_at_reconstructed", "period_end", "period_start"):
        facts[name] = pd.to_datetime(facts[name], utc=True, errors="coerce")
    facts = facts[facts.available_at_reconstructed.notna() & facts.period_end.notna()]
    facts["channel"] = facts.concept.map(lambda c: LOOKUP[c][0])
    duration = (facts.period_end - facts.period_start).dt.days
    facts = facts[
        (facts.channel.isin(ANNUAL) & duration.between(300, 400))
        | (~facts.channel.isin(ANNUAL) & facts.period_start.isna())
    ]
    facts = facts[facts.period_end.le(facts.available_at_reconstructed)]
    filings = filings[filings.quality_valid].copy()
    filings["available_at_reconstructed"] = pd.to_datetime(
        filings.available_at_reconstructed, utc=True
    )
    if companies.symbol.duplicated().any():
        raise PipelineError("Company identity map has duplicate symbols")
    identities = companies.set_index("symbol").issuer_cik.to_dict()
    ordered = sessions.sort_values("session_date")
    earlier_cutoffs = dict(
        zip(
            ordered.session_date,
            (pd.to_datetime(ordered.close, utc=True) + pd.Timedelta(minutes=30)).shift(5),
            strict=True,
        )
    )
    rows = []
    for symbol, block in index.groupby("symbol", sort=True):
        cik = identities.get(symbol)
        if cik is None:
            raise PipelineError("A fitting stock lacks its declared research issuer identity")
        events = list(
            facts[facts.issuer_cik.eq(cik)]
            .sort_values(
                ["available_at_reconstructed", "period_end", "concept", "accession_number"]
            )
            .itertuples(index=False)
        )
        filing_rows = list(
            filings[filings.issuer_cik.eq(cik)]
            .sort_values("available_at_reconstructed")
            .itertuples(index=False)
        )
        state, position, fp, known_filings, latest_release = {}, 0, 0, [], pd.NaT
        for entry in block.sort_values("cutoff_at").itertuples(index=False):
            cutoff = pd.Timestamp(entry.cutoff_at)
            while position < len(events) and events[position].available_at_reconstructed <= cutoff:
                event = events[position]
                channel, rank = LOOKUP[event.concept]
                state.setdefault(channel, {}).setdefault(event.period_end, {})[rank] = event
                latest_release = event.available_at_reconstructed
                position += 1
            while fp < len(filing_rows) and filing_rows[fp].available_at_reconstructed <= cutoff:
                known_filings.append(filing_rows[fp])
                fp += 1

            def current(channel, state=state, cutoff=cutoff):
                periods = state.get(channel, {})
                if not periods:
                    return None
                end = max(periods)
                if (cutoff - end).days > config["maximum_financial_period_age_days"]:
                    return None
                return periods[end][min(periods[end])]

            def ratio(a, b):
                if a is None or b is None or b.value <= 0 or a.period_end != b.period_end:
                    return np.nan
                if (
                    pd.notna(a.period_start)
                    and pd.notna(b.period_start)
                    and a.period_start != b.period_start
                ):
                    return np.nan
                return a.value / b.value

            def growth(channel, scaled=False, state=state, current=current):
                latest = current(channel)
                if latest is None:
                    return np.nan
                prior = [
                    end for end in state[channel] if 300 <= (latest.period_end - end).days <= 430
                ]
                if not prior:
                    return np.nan
                end = min(prior, key=lambda p: abs((latest.period_end - p).days - 365))
                previous = state[channel][end][min(state[channel][end])]
                if (not scaled and previous.value <= 0) or abs(previous.value) < 1e-12:
                    return np.nan
                return (latest.value - previous.value) / abs(previous.value)

            values = {n: current(n) for n in CONCEPTS}
            left = earlier_cutoffs[entry.session_date]
            left = left if pd.notna(left) else cutoff - pd.Timedelta(days=7)
            recent = [f for f in known_filings if f.available_at_reconstructed > left]
            eights = [f for f in known_filings if str(f.form).startswith("8-K")]
            # Recent-submissions snapshots may be truncated: pre-first-filing is unknown.
            counts_known = bool(known_filings)
            row = {
                "row_id": entry.row_id,
                "sec_max_available_at_reconstructed": latest_release,
                "sec_liabilities_assets": ratio(values["liabilities"], values["assets"]),
                "sec_cash_assets": ratio(values["cash"], values["assets"]),
                "sec_current_ratio": ratio(values["current_assets"], values["current_liabilities"]),
                "sec_equity_assets": ratio(values["equity"], values["assets"]),
                "sec_annual_net_margin": ratio(values["income"], values["revenue"]),
                "sec_annual_roa": ratio(values["income"], values["assets"]),
                "sec_annual_operating_cash_assets": ratio(
                    values["operating_cash"], values["assets"]
                ),
                "sec_annual_revenue_growth": growth("revenue"),
                "sec_annual_income_change_scaled": growth("income", scaled=True),
                "sec_days_since_financial": (cutoff - latest_release).total_seconds() / 86400
                if pd.notna(latest_release)
                else np.nan,
                "sec_filings_5d": len(recent) if counts_known else np.nan,
                "sec_8k_5d": sum(str(f.form).startswith("8-K") for f in recent)
                if counts_known
                else np.nan,
                "sec_earnings_8k_5d": sum(
                    str(f.form).startswith("8-K") and "2.02" in str(f.items) for f in recent
                )
                if counts_known
                else np.nan,
                "sec_days_since_8k": (
                    cutoff - eights[-1].available_at_reconstructed
                ).total_seconds()
                / 86400
                if eights
                else np.nan,
            }
            rows.append(row)
    result = finish(pd.DataFrame(rows), SEC_NUMERIC)
    return result


def news_features(news, index, sessions, config):
    news = news.copy()
    news["available_at_reconstructed"] = pd.to_datetime(news.available_at_reconstructed, utc=True)
    news["score"] = news.headline.map(sentiment)
    news["earnings"] = news.headline.str.contains(
        r"\b(?:earnings|revenue|eps|guidance)\b", case=False, regex=True
    ).astype(int)
    calendar = sessions.sort_values("session_date").copy()
    cuts = pd.to_datetime(calendar.close, utc=True) + pd.Timedelta(minutes=30)
    lower = dict(zip(calendar.session_date, cuts.shift(5), strict=True))
    prior = dict(zip(calendar.session_date, cuts.shift(1), strict=True))
    start = pd.Timestamp(config["news_coverage_start"], tz="UTC")
    end = pd.Timestamp(config["news_coverage_end"], tz="UTC") + pd.Timedelta(days=1)
    rows = []
    for symbol, block in index.groupby("symbol", sort=True):
        records = news[news.symbol.eq(symbol)].sort_values("available_at_reconstructed")
        times = pd.DatetimeIndex(records.available_at_reconstructed).as_unit("ns").asi8
        scores, earnings = records.score.to_numpy(), records.earnings.to_numpy()
        for entry in block.itertuples(index=False):
            cutoff = pd.Timestamp(entry.cutoff_at)
            right = np.searchsorted(times, cutoff.value, side="right")
            active = start <= cutoff < end and right > 0
            row = {"row_id": entry.row_id, "news_source_window_active": bool(active)}
            row["news_max_available_at_reconstructed"] = (
                pd.Timestamp(times[right - 1], tz="UTC") if right else pd.NaT
            )
            for name in NEWS_NUMERIC:
                row[name] = np.nan
            if active:
                left5, left1 = lower[entry.session_date], prior[entry.session_date]
                left5 = left5 if pd.notna(left5) else start
                left1 = left1 if pd.notna(left1) else start
                a = np.searchsorted(times, left5.value, side="right")
                b = np.searchsorted(times, left1.value, side="right")
                recent, daily = scores[a:right], scores[b:right]
                row.update(
                    news_count_1d=len(daily),
                    news_count_5d=len(recent),
                    news_sentiment_1d=float(daily.mean()) if len(daily) else 0,
                    news_sentiment_5d=float(recent.mean()) if len(recent) else 0,
                    news_sentiment_std_5d=float(recent.std()) if len(recent) else 0,
                    news_positive_fraction_5d=float((recent > 0).mean()) if len(recent) else 0,
                    news_negative_fraction_5d=float((recent < 0).mean()) if len(recent) else 0,
                    news_earnings_fraction_5d=float(earnings[a:right].mean()) if len(recent) else 0,
                    news_days_since_headline=min((cutoff.value - times[right - 1]) / 86400e9, 365),
                )
            rows.append(row)
    return finish(pd.DataFrame(rows), NEWS_NUMERIC)


def augment(block, features, columns):
    merged = block["index"][["row_id", "cutoff_at"]].merge(
        features, on="row_id", how="left", validate="one_to_one"
    )
    if merged[columns].isna().all(axis=1).any():
        raise PipelineError("Feature augmentation lacks matching fitting rows")
    for column in ("sec_max_available_at_reconstructed", "news_max_available_at_reconstructed"):
        if column in merged and pd.to_datetime(merged[column], utc=True).gt(merged.cutoff_at).any():
            raise PipelineError("Fundamental/news augmentation violated prediction cutoff")
    x = pd.concat([block["X"].reset_index(drop=True), merged[columns]], axis=1)
    return {**block, "X": x}
