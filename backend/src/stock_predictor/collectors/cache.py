"""Resumable HTTP requests shared by evidence collectors, with credential-free keys."""

import base64
import json

from stock_predictor.config import PipelineError
from stock_predictor.storage.archive import Archive, digest, json_bytes

from .http import SourceClient, public_params


class CachedSource:
    def __init__(self, source: SourceClient, archive: Archive, *, refresh: bool = False):
        self.source, self.archive, self.refresh = source, archive, refresh

    def get(self, url: str, params: dict | None = None, *, revision_scope: str = "immutable"):
        spec = {
            "version": 1,
            "source": self.source.source,
            "url": url,
            "params": public_params(params),
            "revision_scope": revision_scope,
        }
        key = f"checkpoints/evidence/v1/{digest(json_bytes(spec))}.json"
        # Frozen query snapshots can be restored from the verified local mirror.
        # Fresh Actions runners fall back to S3 automatically when a file is absent.
        saved = self.archive.read(key)
        if saved and not self.refresh:
            ref = json.loads(saved)["raw"]
            envelope = self.archive.load_snapshot(ref)
            return ref, base64.b64decode(envelope["response"]["body_base64"]), envelope
        ref, response = self.source.get(url, params)
        envelope = self.archive.load_snapshot(ref)
        self.archive.json(key, {"spec": spec, "raw": ref}, immutable=False)
        return ref, response.content, envelope

    def json(self, url: str, params: dict | None = None, *, revision_scope: str = "immutable"):
        ref, body, envelope = self.get(url, params, revision_scope=revision_scope)
        try:
            payload = json.loads(body)
        except ValueError:
            raise PipelineError(f"{self.source.source}: invalid JSON in {ref['key']}") from None
        if not isinstance(payload, dict):
            raise PipelineError(f"{self.source.source}: unexpected JSON schema")
        return payload, ref, envelope

    def close(self):
        self.source.close()
