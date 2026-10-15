"""Pinned public FI-2010 files, chronological windows, and explicit native label mapping."""

import hashlib
import json
import zipfile
from pathlib import Path

import httpx
import numpy as np
import torch
from torch.utils.data import Dataset

REVISION = "ff14d7c2fd38bdfc143389786993d0f0236d4eb8"
URL = (
    "https://raw.githubusercontent.com/zcakhaa/"
    f"DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/{REVISION}/data/data.zip"
)
HORIZONS = (10, 20, 30, 50, 100)
TRAIN_FILE = "Train_Dst_NoAuction_DecPre_CF_7.txt"
TEST_FILES = [f"Test_Dst_NoAuction_DecPre_CF_{n}.txt" for n in (7, 8, 9)]


def checksum(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(root: Path, expected_sha: str) -> tuple[Path, dict]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "data.zip"
    if not path.exists():
        partial = root / "data.zip.partial"
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            with client.stream("GET", URL) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_bytes(1024 * 1024):
                        stream.write(chunk)
        if checksum(partial) != expected_sha:
            raise ValueError("FI-2010 download checksum mismatch")
        partial.replace(path)
    if checksum(path) != expected_sha:
        raise ValueError("FI-2010 cache checksum mismatch")
    return path, {
        "url": URL,
        "revision": REVISION,
        "sha256": expected_sha,
        "bytes": path.stat().st_size,
        "normalization": "provided decimal precision",
    }


def read_file(archive_path: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read only raw LOB columns and supplied labels; never extract untrusted ZIP paths."""
    features, labels = [], []
    with zipfile.ZipFile(archive_path) as archive:
        matches = [n for n in archive.namelist() if Path(n).name == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one FI-2010 member: {name}")
        with archive.open(matches[0]) as stream:
            count = 0
            for row, line in enumerate(stream):
                count += 1
                if row < 40:
                    features.append(np.fromstring(line.decode("ascii"), sep=" ", dtype=np.float32))
                elif 144 <= row < 149:
                    values = np.fromstring(line.decode("ascii"), sep=" ", dtype=np.float32)
                    if not np.isin(values, [1, 2, 3]).all():
                        raise ValueError("Invalid native FI-2010 labels")
                    labels.append(values.astype(np.uint8))
    if count != 149 or len(features) != 40 or len(labels) != 5:
        raise ValueError("Unexpected FI-2010 row schema")
    x, y = np.stack(features).T.copy(), np.stack(labels).T.copy()
    if x.shape[0] != y.shape[0] or not np.isfinite(x).all():
        raise ValueError("Non-finite or misaligned FI-2010 observations")
    return x, y


def prepare(root: Path, zip_path: Path, name: str, sha: str) -> tuple[np.ndarray, np.ndarray]:
    cache = root / sha / name.removesuffix(".txt")
    manifest_path = cache.with_suffix(".json")
    x_path, y_path = cache.with_suffix(".x.npy"), cache.with_suffix(".y.npy")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if checksum(x_path) != manifest["x_sha256"] or checksum(y_path) != manifest["y_sha256"]:
            raise ValueError("Parsed FI-2010 cache checksum mismatch")
    else:
        x, y = read_file(zip_path, name)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(x_path, x, allow_pickle=False)
        np.save(y_path, y, allow_pickle=False)
        manifest_path.write_text(
            json.dumps(
                {
                    "source_sha256": sha,
                    "member": name,
                    "rows": len(x),
                    "x_sha256": checksum(x_path),
                    "y_sha256": checksum(y_path),
                }
            ),
            encoding="utf-8",
        )
    return np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r")


def chronological_ranges(rows: int, fraction: float, purge: int, window: int) -> dict:
    boundary = int(rows * fraction)
    if boundary - purge < window or rows - boundary < window:
        raise ValueError("Not enough observations for disjoint purged windows")
    return {
        "train": (0, boundary - purge),
        "validation": (boundary, rows),
        "boundary": boundary,
        "purge_observations": purge,
    }


class Windows(Dataset):
    def __init__(self, x, native_labels, horizon, start=0, stop=None, window=100, stride=1):
        if horizon not in HORIZONS or window < 1 or stride < 1:
            raise ValueError("Invalid window, stride, or horizon")
        stop = len(x) if stop is None else stop
        if start < 0 or stop > len(x) or stop <= start:
            raise ValueError("Invalid observation range")
        self.x = x
        self.y = native_labels
        self.column = HORIZONS.index(horizon)
        self.window = window
        self.ends = np.arange(start + window - 1, stop, stride, dtype=np.int64)
        if not len(self.ends):
            raise ValueError("Observation range has no complete windows")

    @property
    def targets(self):
        # Official native order: 1 UP, 2 STATIONARY, 3 DOWN. Internal: DOWN, FLAT, UP.
        return 3 - self.y[self.ends, self.column].astype(np.int64)

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, index):
        end = self.ends[index]
        # A copy avoids read-only NumPy/Torch views and never materializes all rolling windows.
        x = np.array(self.x[end - self.window + 1 : end + 1], dtype=np.float32, copy=True)
        y = 3 - int(self.y[end, self.column])
        return torch.from_numpy(x), y
