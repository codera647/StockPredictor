import gzip
import hashlib
import json
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from stock_predictor.config import PipelineError, Settings


def json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class Archive:
    """Immutable objects are addressed by hash; checkpoints alone are mutable.

    A successful write requires S3 confirmation before a checkpoint can advance.
    Reads verify content hashes when supplied. No credentials enter object metadata.
    """

    def __init__(self, root: Path, settings: Settings | None = None):
        self.root = root
        self.client = None
        self.bucket = ""
        if settings:
            self.bucket = settings.s3_bucket
            self.client = boto3.client(
                "s3",
                endpoint_url=settings.s3_endpoint,
                region_name=settings.s3_region,
                aws_access_key_id=settings.s3_key,
                aws_secret_access_key=settings.s3_secret,
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    connect_timeout=15,
                    read_timeout=45,
                    retries={"max_attempts": 3},
                    request_checksum_calculation="when_required",
                    response_checksum_validation="when_required",
                ),
            )

    def _path(self, key: str) -> Path:
        if key.startswith("/") or "\\" in key or any(p in ("", "..", ".") for p in key.split("/")):
            raise PipelineError("Unsafe archive key")
        return self.root.joinpath(*key.split("/"))

    def check_access(self) -> None:
        if self.client:
            try:
                self.client.head_bucket(Bucket=self.bucket)
            except (BotoCoreError, ClientError) as exc:
                raise PipelineError(
                    "Private S3 bucket is inaccessible; check S3 credentials"
                ) from exc

    def write(self, key: str, body: bytes, *, immutable: bool = True) -> dict:
        path = self._path(key)
        sha = digest(body)
        if immutable and path.exists() and digest(path.read_bytes()) != sha:
            raise PipelineError("Immutable archive object collision")
        if self.client:
            try:
                # Hash paths are immutable by construction. Existing bytes are checked as well.
                existing = self.read(key, remote=True)
                if existing is not None and immutable and digest(existing) != sha:
                    raise PipelineError("Remote immutable archive object collision")
                if existing != body:
                    self.client.put_object(
                        Bucket=self.bucket,
                        Key=key,
                        Body=body,
                        ContentType="application/octet-stream",
                        Metadata={"sha256": sha},
                    )
                    confirmed = self.read(key, remote=True)
                    if confirmed is None or digest(confirmed) != sha:
                        raise PipelineError("S3 upload verification failed")
            except (BotoCoreError, ClientError) as exc:
                raise PipelineError("S3 write failed; checkpoint was not advanced") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_bytes(body)
        temp.replace(path)
        return {"key": key, "sha256": sha, "bytes": len(body)}

    def read(self, key: str, *, remote: bool = False, sha256: str | None = None) -> bytes | None:
        path = self._path(key)
        body = None
        if self.client and (remote or not path.exists()):
            try:
                body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in ("NoSuchKey", "404", "NotFound"):
                    raise PipelineError("S3 read failed") from exc
            except BotoCoreError as exc:
                raise PipelineError("S3 read failed") from exc
        elif path.exists():
            body = path.read_bytes()
        if body is not None and sha256 and digest(body) != sha256:
            raise PipelineError("Archive checksum mismatch; refusing corrupted input")
        return body

    def json(self, key: str, value: object, *, immutable: bool = True) -> dict:
        return self.write(key, json_bytes(value), immutable=immutable)

    def snapshot(self, envelope: dict) -> dict:
        raw = json_bytes(envelope)
        body = gzip.compress(raw, mtime=0)
        key = f"raw/{envelope['source']}/{digest(raw)}.json.gz"
        return self.write(key, body)

    def load_snapshot(self, ref: dict) -> dict:
        body = self.read(ref["key"], sha256=ref["sha256"])
        if body is None:
            raise PipelineError("Raw snapshot referenced by checkpoint is missing")
        return json.loads(gzip.decompress(body))
