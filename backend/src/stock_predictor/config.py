"""Load only the settings needed by extraction; never expose credential values."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


class PipelineError(Exception):
    """A safe, actionable pipeline error whose text may be logged."""


def require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value or value.startswith("your_") or set(value) == {"*"}:
        raise PipelineError(f"Missing real value for {name}")
    return value


def safe_url(value: str, name: str, schemes: tuple[str, ...] = ("https",)) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in schemes or not parsed.hostname or parsed.username or parsed.password:
        raise PipelineError(f"Invalid {name}; use a provider URL without embedded credentials")
    if parsed.query or parsed.fragment:
        raise PipelineError(f"Invalid {name}; query strings and fragments are not supported")
    return value.rstrip("/")


@dataclass(frozen=True)
class Settings:
    alpaca_base: str
    alpaca_feed: str
    alpaca_key: str = field(repr=False)
    alpaca_secret: str = field(repr=False)
    sec_mapping_url: str = "https://www.sec.gov/files/company_tickers_exchange.json"
    sec_user_agent: str = field(default="", repr=False)
    nasdaq_listed_url: str = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt"
    nasdaq_other_url: str = "ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt"
    s3_endpoint: str = ""
    s3_region: str = ""
    s3_bucket: str = ""
    s3_key: str = field(default="", repr=False)
    s3_secret: str = field(default="", repr=False)

    @classmethod
    def load(cls, env_file: Path, *, upload: bool = True) -> "Settings":
        # Actions environment wins; local .env is never copied into manifests or logs.
        load_dotenv(env_file, override=False)
        feed = os.environ.get("ALPACA_DATA_FEED", "sip").strip()
        if feed != "sip":
            raise PipelineError("Daily extraction requires ALPACA_DATA_FEED=sip; no feed fallback")
        storage = {}
        if upload:
            storage = {
                "s3_endpoint": safe_url(require("S3_ENDPOINT_URL"), "S3_ENDPOINT_URL"),
                "s3_region": require("S3_REGION"),
                "s3_bucket": require("S3_BUCKET_NAME"),
                "s3_key": require("S3_ACCESS_KEY_ID"),
                "s3_secret": require("S3_SECRET_ACCESS_KEY"),
            }
        return cls(
            alpaca_base=safe_url(
                os.environ.get("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets"),
                "ALPACA_DATA_BASE_URL",
            ),
            alpaca_feed=feed,
            alpaca_key=require("APCA_API_KEY_ID"),
            alpaca_secret=require("APCA_API_SECRET_KEY"),
            sec_mapping_url=safe_url(
                os.environ.get("SEC_TICKER_MAPPING_URL", cls.sec_mapping_url),
                "SEC_TICKER_MAPPING_URL",
            ),
            sec_user_agent=os.environ.get("SEC_USER_AGENT", "").strip(),
            nasdaq_listed_url=safe_url(
                os.environ.get("NASDAQ_LISTED_SYMBOLS_URL", cls.nasdaq_listed_url),
                "NASDAQ_LISTED_SYMBOLS_URL",
                ("https", "ftp"),
            ),
            nasdaq_other_url=safe_url(
                os.environ.get("NASDAQ_OTHER_LISTED_SYMBOLS_URL", cls.nasdaq_other_url),
                "NASDAQ_OTHER_LISTED_SYMBOLS_URL",
                ("https", "ftp"),
            ),
            **storage,
        )


def load_universe(path: Path, symbols: list[str] | None = None) -> dict:
    universe = json.loads(path.read_text(encoding="utf-8"))
    entries = universe["instruments"]
    if symbols:
        unknown = set(symbols) - {entry["symbol"] for entry in entries}
        if unknown:
            raise PipelineError(f"Symbols absent from the versioned universe: {sorted(unknown)}")
        entries = [entry for entry in entries if entry["symbol"] in symbols]
    for key in ("symbol", "instrument_id"):
        if len({entry[key] for entry in entries}) != len(entries):
            raise PipelineError(f"Duplicate {key} in universe")
    if not entries:
        raise PipelineError("Universe must contain at least one instrument")
    return {**universe, "instruments": entries}
