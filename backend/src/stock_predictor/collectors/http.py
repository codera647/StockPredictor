import base64
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from stock_predictor.config import PipelineError
from stock_predictor.storage.archive import Archive


class SourceClient:
    def __init__(
        self,
        source: str,
        archive: Archive,
        *,
        headers: dict | None = None,
        interval: float = 0.4,
        client: httpx.Client | None = None,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ):
        self.source = source
        self.archive = archive
        self.client = client or httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(45, connect=15),
            follow_redirects=False,
        )
        self.headers = headers or {}
        self.interval = interval
        self.sleep, self.monotonic = sleep, monotonic
        self.last = float("-inf")

    def _retry_delay(self, response: httpx.Response | None, attempt: int) -> float:
        fallback = min(2**attempt, 30)
        if response is None:
            return fallback
        value = response.headers.get("retry-after", "")
        try:
            delay = float(value)
        except ValueError:
            try:
                delay = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
            except (TypeError, ValueError):
                delay = fallback
        return max(fallback, min(delay, 60))

    def get(self, url: str, params: dict | None = None) -> tuple[dict, httpx.Response]:
        for attempt in range(5):
            self.sleep(max(0.0, self.interval - (self.monotonic() - self.last)))
            self.last = self.monotonic()
            try:
                response = self.client.get(url, params=params, headers=self.headers)
            except httpx.TransportError:
                if attempt == 4:
                    raise PipelineError(
                        f"{self.source}: network request failed after retries"
                    ) from None
                self.sleep(self._retry_delay(None, attempt))
                continue
            # Archive original bytes before interpretation, including failed responses.
            envelope = {
                "schema_version": 1,
                "source": self.source,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "request": {"method": "GET", "url": url, "params": params or {}},
                "response": {
                    "status": response.status_code,
                    "headers": {
                        k: v
                        for k, v in response.headers.items()
                        if k.lower()
                        in (
                            "date",
                            "content-type",
                            "etag",
                            "last-modified",
                            "retry-after",
                            "x-ratelimit-limit",
                            "x-ratelimit-remaining",
                            "x-ratelimit-reset",
                        )
                    },
                    "body_base64": base64.b64encode(response.content).decode(),
                },
            }
            ref = self.archive.snapshot(envelope)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                self.sleep(self._retry_delay(response, attempt))
                continue
            if not 200 <= response.status_code < 300:
                # Do not log request headers, response text or URLs with contact information.
                raise PipelineError(
                    f"{self.source}: HTTP {response.status_code}; preserved response {ref['key']}"
                )
            return ref, response
        raise PipelineError(f"{self.source}: retry limit reached")

    def close(self) -> None:
        self.client.close()
