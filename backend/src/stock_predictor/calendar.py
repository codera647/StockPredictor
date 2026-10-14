from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from stock_predictor.config import PipelineError

NEW_YORK = ZoneInfo("America/New_York")


def schedule(start: date, end: date) -> pd.DataFrame:
    if start > end:
        raise PipelineError("Start date must not follow end date")
    calendar = xcals.get_calendar("XNYS", start=start, end=end + timedelta(days=7))
    frame = calendar.schedule.loc[start.isoformat() : end.isoformat(), ["open", "close"]].copy()
    frame.insert(0, "session_date", frame.index.strftime("%Y-%m-%d"))
    return frame.reset_index(drop=True)


def latest_completed_session(now: datetime | None = None) -> date:
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise PipelineError("Collection clock must be timezone-aware")
    today = now.astimezone(NEW_YORK).date()
    frame = schedule(today - timedelta(days=14), today)
    completed = frame[frame["close"] + pd.Timedelta(minutes=30) <= pd.Timestamp(now)]
    if completed.empty:
        raise PipelineError("No completed regular session available")
    return date.fromisoformat(completed.iloc[-1]["session_date"])


def request_bounds(start: date, end: date, now: datetime | None = None) -> tuple[str, str]:
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise PipelineError("Collection clock must be timezone-aware")
    if end > latest_completed_session(now):
        raise PipelineError("Requested end includes a session before close +30 minutes")
    lower = datetime.combine(start, time.min, NEW_YORK).astimezone(UTC)
    # Keep an explicit safety margin beyond Basic's 15-minute SIP restriction.
    upper = min(
        datetime.combine(end + timedelta(days=1), time.min, NEW_YORK).astimezone(UTC)
        - timedelta(microseconds=1),
        now - timedelta(minutes=16),
    )
    if lower > upper:
        raise PipelineError("Requested date interval contains no available history")
    return lower.isoformat(), upper.isoformat()


def year_partitions(start: date, end: date):
    for year in range(start.year, end.year + 1):
        yield max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
