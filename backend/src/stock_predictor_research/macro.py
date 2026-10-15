import numpy as np
import pandas as pd
from stock_predictor.config import PipelineError
from stock_predictor.datasets.features import FEATURE_COLUMNS

SERIES = ("DFF", "DFEDTARU", "DGS2", "DGS10", "CPIAUCSL", "UNRATE")
MACRO_NUMERIC = [
    "macro_effective_rate",
    "macro_target_upper",
    "macro_yield_2y",
    "macro_yield_10y",
    "macro_curve_slope",
    "macro_cpi_yoy",
    "macro_unemployment",
    "macro_yield_10y_change_20",
    "macro_effective_rate_change_20",
    "macro_unemployment_change_3m",
    "macro_daily_age_days",
    "macro_monthly_release_age_days",
]
RESEARCH_FEATURES = FEATURE_COLUMNS + MACRO_NUMERIC + [n + "_missing" for n in MACRO_NUMERIC]


def macro_features(observations: pd.DataFrame, sessions: pd.DataFrame, config: dict):
    """Replay each vintage only after its conservative reconstructed release time."""
    records = observations[observations.quality_valid & observations.series_id.isin(SERIES)].copy()
    records["release_at"] = pd.to_datetime(records.available_at_reconstructed, utc=True)
    records["value"] = pd.to_numeric(records.value, errors="coerce")
    records = records.sort_values(["release_at", "series_id", "observation_date", "realtime_start"])
    if set(records.series_id) != set(SERIES):
        raise PipelineError("The versioned macro bundle requires all six archived FRED series")
    events = list(records.itertuples(index=False))
    state, cursor, rows = {name: {} for name in SERIES}, 0, []
    for session in sessions.itertuples(index=False):
        cutoff = pd.Timestamp(session.close) + pd.Timedelta(minutes=30)
        while cursor < len(events) and events[cursor].release_at <= cutoff:
            event = events[cursor]
            # Retain a previously released revision until its replacement becomes
            # usable; date-only vintage expiry does not backdate the new release.
            state[event.series_id][event.observation_date] = event
            cursor += 1
        values, releases, ages = {}, [], {}
        day = pd.Timestamp(session.session_date)
        for name in SERIES:
            known = [
                key
                for key, event in state[name].items()
                if key <= session.session_date and np.isfinite(event.value)
            ]
            if not known:
                values[name] = None
                continue
            event = state[name][max(known)]
            age = int((day - pd.Timestamp(event.observation_date)).days)
            limit = (
                config["maximum_monthly_macro_age_days"]
                if name in ("CPIAUCSL", "UNRATE")
                else config["maximum_daily_macro_age_days"]
            )
            values[name] = event if age <= limit else None
            if values[name] is not None:
                releases.append(event.release_at)
                ages[name] = age

        def rate(name, values=values):
            return values[name].value / 100 if values[name] is not None else np.nan

        def change(name, months, ratio=False, values=values, releases=releases):
            event = values[name]
            if event is None:
                return np.nan
            earlier = (
                (pd.Timestamp(event.observation_date) - pd.DateOffset(months=months))
                .date()
                .isoformat()
            )
            previous = state[name].get(earlier)
            if previous is None or not np.isfinite(previous.value) or previous.value <= 0:
                return np.nan
            releases.append(previous.release_at)
            return (
                event.value / previous.value - 1 if ratio else (event.value - previous.value) / 100
            )

        monthly = [values[n] for n in ("CPIAUCSL", "UNRATE") if values[n] is not None]
        row = {
            "session_date": session.session_date,
            "macro_cutoff_at": cutoff,
            "macro_effective_rate": rate("DFF"),
            "macro_target_upper": rate("DFEDTARU"),
            "macro_yield_2y": rate("DGS2"),
            "macro_yield_10y": rate("DGS10"),
            "macro_curve_slope": rate("DGS10") - rate("DGS2"),
            "macro_cpi_yoy": change("CPIAUCSL", 12, ratio=True),
            "macro_unemployment": rate("UNRATE"),
            "macro_unemployment_change_3m": change("UNRATE", 3),
            "macro_daily_age_days": max([ages[n] for n in SERIES[:4] if n in ages], default=np.nan),
            "macro_monthly_release_age_days": max(
                [(cutoff - e.release_at).total_seconds() / 86400 for e in monthly], default=np.nan
            ),
        }
        row["macro_max_available_at_reconstructed"] = max(releases) if releases else pd.NaT
        rows.append(row)
    result = pd.DataFrame(rows)
    for base in ("macro_yield_10y", "macro_effective_rate"):
        result[base + "_change_20"] = result[base] - result[base].shift(20)
    for name in MACRO_NUMERIC:
        result[name] = result[name].replace([np.inf, -np.inf], np.nan)
        result[name + "_missing"] = result[name].isna().astype("int8")
    if result.macro_max_available_at_reconstructed.gt(result.macro_cutoff_at).any():
        raise PipelineError("Macro replay used a revision before its release cutoff")
    return result


def augment(block, macro):
    merged = block["index"][["session_date", "cutoff_at"]].merge(
        macro, on="session_date", how="left", validate="many_to_one"
    )
    if (
        merged.macro_cutoff_at.isna().any()
        or merged.macro_max_available_at_reconstructed.gt(merged.cutoff_at).any()
    ):
        raise PipelineError(
            "Macro augmentation is missing a session or violates cutoff availability"
        )
    x = pd.concat(
        [
            block["X"].reset_index(drop=True),
            merged[MACRO_NUMERIC + [n + "_missing" for n in MACRO_NUMERIC]],
        ],
        axis=1,
    )
    return {**block, "X": x[RESEARCH_FEATURES]}
