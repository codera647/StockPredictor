import hashlib
import io
import json
import os
from pathlib import Path

import httpx
import numpy as np
import torch
from torch import nn


def digest(body):
    return hashlib.sha256(body).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(body)
    temp.replace(path)


def save_json(path, value):
    write(path, canonical(value))


def fetch(path, url, sha):
    if not path.exists():
        response = httpx.get(url, timeout=120, follow_redirects=True)
        response.raise_for_status()
        if digest(response.content) != sha:
            raise ValueError("Public source download checksum mismatch")
        write(path, response.content)
    if digest(path.read_bytes()) != sha:
        raise ValueError("Public source cache checksum mismatch")
    return path


def initialize(device, seed, threads=2):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Select a Colab GPU runtime; this recipe requires CUDA")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PlainLSTM(nn.Module):
    def __init__(self, inputs, outputs, hidden=128, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(inputs, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, outputs)

    def sequence(self, x, state=None):
        output, state = self.lstm(x, state)
        return self.head(self.dropout(output)), state

    def forward(self, x):
        output, _ = self.sequence(x)
        return output[:, -1]


def weights(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def restore(body, inputs, outputs, config, device="cpu"):
    model = PlainLSTM(inputs, outputs, config["hidden"], config["dropout"])
    model.load_state_dict(torch.load(io.BytesIO(body), map_location="cpu", weights_only=True))
    return model.to(device).eval()


def row_digest(file_ids, endpoints, labels):
    hasher = hashlib.sha256()
    for values in (
        np.asarray(file_ids, dtype="u1"),
        np.asarray(endpoints, dtype="<i8"),
        np.asarray(labels, dtype="u1"),
    ):
        hasher.update(values.tobytes())
    return hasher.hexdigest()


def source_hash():
    import stock_predictor.training.metrics as metrics
    import stock_predictor_lob.data as data
    import stock_predictor_lob.training as training

    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [Path(data.__file__), Path(training.__file__), Path(metrics.__file__)]
    return digest(
        canonical(
            {str(p.name): digest(p.read_text(encoding="utf-8").encode()) for p in sorted(paths)}
        )
    )
