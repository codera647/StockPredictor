import json

from stock_predictor.config import PipelineError, Settings
from stock_predictor.storage.archive import Archive, digest, json_bytes

from .http import SourceClient


class AlpacaCollector:
    """Consume every page and resume only from durably archived responses."""

    def __init__(self, settings: Settings, archive: Archive, client: SourceClient | None = None):
        self.settings, self.archive = settings, archive
        self.client = client or SourceClient(
            "alpaca",
            archive,
            headers={
                "APCA-API-KEY-ID": settings.alpaca_key,
                "APCA-API-SECRET-KEY": settings.alpaca_secret,
            },
        )

    def collect(self, kind: str, params: dict, *, refresh: bool = False) -> dict:
        path = {"bars": "/v2/stocks/bars", "actions": "/v1/corporate-actions"}[kind]
        url = self.settings.alpaca_base + path
        spec = {"collector_version": 1, "kind": kind, "url": url, "params": params}
        task_id = digest(json_bytes(spec))
        key = f"checkpoints/extraction/v1/{task_id}.json"
        body = self.archive.read(key, remote=bool(self.archive.client))
        state = (
            json.loads(body)
            if body
            else {
                "task_id": task_id,
                "spec": spec,
                "pages": [],
                "next_page_token": None,
                "complete": False,
            }
        )
        if state["spec"] != spec:
            raise PipelineError("Checkpoint request does not match its fingerprint")
        # A deliberate refresh produces a new snapshot set; old raw objects remain untouched.
        if refresh and state["complete"]:
            state = {**state, "pages": [], "complete": False, "next_page_token": None}
        if state["complete"]:
            for ref in state["pages"]:
                self.archive.load_snapshot(ref)
            return state
        seen = {page.get("request_page_token") for page in state["pages"]}
        while True:
            token = state["next_page_token"]
            if token in seen:
                raise PipelineError("Repeated pagination token; refusing incomplete collection")
            query = {**params, **({"page_token": token} if token else {})}
            ref, response = self.client.get(url, query)
            try:
                data = response.json()
            except ValueError:
                raise PipelineError(
                    f"Alpaca returned invalid JSON; preserved {ref['key']}"
                ) from None
            next_token, count = validate_page(kind, data)
            if next_token is not None and (next_token in seen or next_token == token):
                raise PipelineError("Repeated pagination token; raw response retained")
            ref = {**ref, "request_page_token": token, "records": count}
            seen.add(token)
            state["pages"].append(ref)
            state["next_page_token"] = next_token
            state["complete"] = next_token is None
            self.archive.json(key, state, immutable=False)
            if state["complete"]:
                return state

    def close(self) -> None:
        self.client.close()


def validate_page(kind: str, data: object) -> tuple[str | None, int]:
    if not isinstance(data, dict) or "next_page_token" not in data:
        raise PipelineError("Unexpected Alpaca pagination schema")
    token = data["next_page_token"]
    if token is not None and (not isinstance(token, str) or not token):
        raise PipelineError("Invalid Alpaca pagination token")
    field = "bars" if kind == "bars" else "corporate_actions"
    records = data.get(field)
    # Empty responses may use null. Missing response fields are schema failures.
    if field not in data or (records is not None and not isinstance(records, dict)):
        raise PipelineError(f"Unexpected Alpaca {field} schema")
    count = 0
    for items in (records or {}).values():
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise PipelineError(f"Unexpected Alpaca {field} records")
        count += len(items)
    return token, count
