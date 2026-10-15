import io
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stock_predictor_lstm_compare import classification, price  # noqa: E402
from stock_predictor_lstm_compare.common import (  # noqa: E402
    PlainLSTM,
    digest,
    initialize,
    restore,
    row_digest,
    weights,
)


def test_price_error_score_is_not_direction_accuracy():
    actual = np.array([1000, 1001, 1000, 1002, 1001], dtype=float)
    predicted = np.repeat(1000.0, len(actual))
    result = price.price_metrics(actual, predicted, origin=1000)
    assert result["repository_price_score_percent"] > 99
    assert result["daily_path_direction_accuracy"] == 0.2


def test_plain_lstm_gradients_reload_and_position_sensitivity():
    initialize("cpu", 1234)
    model = PlainLSTM(40, 3, hidden=8)
    inputs = torch.randn(4, 100, 40)
    loss = torch.nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1, 2, 0]))
    loss.backward()
    assert model.lstm.weight_ih_l0.grad.abs().sum() > 0
    model.eval()
    restored = restore(weights(model), 40, 3, {"hidden": 8, "dropout": 0.2})
    with torch.inference_mode():
        torch.testing.assert_close(model(inputs), restored(inputs), atol=0, rtol=0)
        changed = inputs.clone()
        changed[:, -1] += 5
        assert not torch.allclose(model(inputs), model(changed))


def test_price_run_fits_scaler_on_training_only(tmp_path, monkeypatch):
    config = json.loads(Path("configs/lstm-price-check.json").read_text())
    config.update(hidden=8, max_epochs=1, patience=1)
    path = tmp_path / "source.csv"
    values = np.arange(252, dtype=float) + 100
    values[-30:] += 10000
    import pandas as pd

    pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=252), "Close": values}).to_csv(
        path,
        index=False,
    )
    monkeypatch.setattr(price, "fetch", lambda *_: path)
    result = price.run(config, tmp_path, tmp_path / "out")
    assert result["scaler_fit_rows"] == 192
    plan = json.loads((tmp_path / "out/price" / result["recipe"] / "plan.json").read_text())
    assert plan["scaler_min"] == 100 and plan["scaler_range"] == 191
    assert result["lstm"]["rows"] == 30
    assert result["comparable_to_deeplob_classification_accuracy"] is False


def test_matched_classification_freeze_export_and_resume(tmp_path, monkeypatch):
    initialize("cpu", 7)
    config = json.loads(Path("configs/lstm-comparison.colab.json").read_text())
    config.update(device="cpu", hidden=8, max_epochs=1, patience=1)
    generator = np.random.default_rng(5)
    train_x = generator.normal(size=(1600, 40)).astype(np.float32)
    train_y = np.tile((np.arange(1600) % 3 + 1)[:, None], (1, 5)).astype(np.uint8)
    test_x, test_y = train_x[:230], train_y[:230]
    ids = np.repeat(np.arange(3), 31)
    ends = np.tile(np.arange(99, 130), 3)
    labels = np.tile(3 - test_y[np.arange(99, 130), 0].astype(int), 3)
    baseline = json.loads(Path("configs/lstm-comparison.baseline.json").read_text())
    baseline.update(rows=93, row_alignment_sha256=row_digest(ids, ends, labels))
    monkeypatch.setattr(classification, "download_lob", lambda *_: (tmp_path / "data.zip", {}))

    def prepare(_root, _zip, name, _sha):
        return (train_x, train_y) if name == classification.TRAIN_FILE else (test_x, test_y)

    monkeypatch.setattr(classification, "prepare", prepare)
    out = tmp_path / "out"
    ready = classification.run(config, baseline, tmp_path, out)
    assert ready["test_opened"] is False and ready["epochs"] == 1

    def no_fit(*_args, **_kwargs):
        pytest.fail("Frozen or cached evaluation attempted training")

    # Completed fitting can be reused by the evaluation phase.
    real_fit = classification.fit
    result = classification.run(config, baseline, tmp_path, out, evaluate=True)
    monkeypatch.setattr(classification, "fit", no_fit)
    repeated = classification.run(config, baseline, tmp_path, out, evaluate=True)
    assert repeated == result and result["test"]["rows"] == 93
    assert result["row_alignment_sha256"] == baseline["row_alignment_sha256"]
    with pytest.raises(ValueError, match="changed recipe"):
        classification.run({**config, "seed": 8}, baseline, tmp_path, out, True)
    directory = out / "classification" / result["recipe"]
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["labels"], labels)
        assert "file_id" in saved.files
    # Recover best weights from the committed epoch state after an interrupted finalization.
    monkeypatch.setattr(classification, "fit", real_fit)
    (directory / "training.json").unlink()
    (directory / "best.pt").write_bytes(b"interrupted write")
    train = classification.Windows(train_x, train_y, 10, 0, 1180, window=100)
    val = classification.Windows(train_x, train_y, 10, 1280, 1500, window=100)
    body, restored_training = real_fit(train, val, config, directory, result["recipe"])
    assert digest(body) == result["weights_sha256"] and restored_training["epochs"] == 1
    assert isinstance(torch.load(io.BytesIO(body), weights_only=True), dict)
