import base64
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from stock_predictor.config import PipelineError
from stock_predictor.storage.archive import Archive

SENSITIVE_PARAMS = {"api_key", "key", "token", "access_token", "secret", "password"}


def public_params(params: dict | None) -> dict:
    return {
        key: "[REDACTED]" if key.lower() in SENSITIVE_PARAMS else value
        for key, value in (params or {}).items()
    }


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
        max_response_bytes: int = 64 * 1024 * 1024,
        max_attempts: int = 5,
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
        self.max_response_bytes = max_response_bytes
        self.max_attempts = max_attempts

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

    def get(
        self,
        url: str,
        params: dict | None = None,
        *,
        acceptable_statuses: tuple[int, ...] = (),
    ) -> tuple[dict, httpx.Response]:
        for attempt in range(self.max_attempts):
            self.sleep(max(0.0, self.interval - (self.monotonic() - self.last)))
            self.last = self.monotonic()
            try:
                with self.client.stream(
                    "GET", url, params=params, headers=self.headers
                ) as streamed:
                    chunks, size, truncated = [], 0, False
                    for chunk in streamed.iter_bytes():
                        if size + len(chunk) > self.max_response_bytes:
                            truncated = True
                            break
                        chunks.append(chunk)
                        size += len(chunk)
                    response = httpx.Response(
                        streamed.status_code,
                        headers={
                            k: v
                            for k, v in streamed.headers.items()
                            if k.lower() not in ("content-encoding", "content-length")
                        },
                        content=b"".join(chunks),
                        request=streamed.request,
                    )
            except httpx.TransportError:
                if attempt == self.max_attempts - 1:
                    raise PipelineError(
                        f"{self.source}: network request failed after retries"
                    ) from None
                self.sleep(self._retry_delay(None, attempt))
                continue
            # Query credentials (FRED) and any echoed secrets are excluded from archives.
            archived_body = response.content
            for key, value in (params or {}).items():
                if key.lower() in SENSITIVE_PARAMS and isinstance(value, str) and len(value) >= 8:
                    archived_body = archived_body.replace(value.encode(), b"[REDACTED]")
            # Archive bytes before interpretation, including failed responses.
            envelope = {
                "schema_version": 1,
                "source": self.source,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "request": {"method": "GET", "url": url, "params": public_params(params)},
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
                    "body_base64": base64.b64encode(archived_body).decode(),
                    "credential_echo_redacted": archived_body != response.content,
                    "truncated": truncated,
                },
            }
            ref = self.archive.snapshot(envelope)
            if truncated:
                raise PipelineError(
                    f"{self.source}: response exceeds size bound; partial response {ref['key']}"
                )
            if (
                response.status_code in (429, 500, 502, 503, 504)
                and attempt < self.max_attempts - 1
            ):
                self.sleep(self._retry_delay(response, attempt))
                continue
            if (
                not 200 <= response.status_code < 300
                and response.status_code not in acceptable_statuses
            ):
                # Do not log request headers, response text or URLs with contact information.
                error = PipelineError(
                    f"{self.source}: HTTP {response.status_code}; preserved response {ref['key']}"
                )
                if response.status_code in (301, 302, 303, 307, 308):
                    error.redirect_location = response.headers.get("location")
                raise error
            return ref, response
        raise PipelineError(f"{self.source}: retry limit reached")

    def close(self) -> None:
        self.client.close()
