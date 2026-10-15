import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stock_predictor_lob.data import Windows, chronological_ranges, read_file  # noqa: E402
from stock_predictor_lob.models import build_model  # noqa: E402
from stock_predictor_lob.pipeline import authorize_test, validate  # noqa: E402
from stock_predictor_lob.training import fit, predict, restore, state_bytes  # noqa: E402


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def arrays(rows=500):
    x = np.random.default_rng(5).normal(size=(rows, 40)).astype(np.float32)
    y = np.tile((np.arange(rows) % 3 + 1)[:, None], (1, 5)).astype(np.uint8)
    return x, y


def test_windows_use_endpoint_labels_and_do_not_cross_split():
    x, y = arrays(1000)
    ranges = chronological_ranges(1000, 0.8, 100, 100)
    train = Windows(x, y, 100, *ranges["train"], window=100, stride=7)
    validation = Windows(x, y, 100, *ranges["validation"], window=100)
    assert train.ends[-1] + 100 < ranges["boundary"]
    assert validation.ends[0] - 99 == ranges["boundary"]
    inputs, label = validation[0]
    np.testing.assert_array_equal(inputs.numpy(), x[800:900])
    assert label == 3 - y[899, 4]
    np.testing.assert_array_equal(validation.targets, 3 - y[899:, 4].astype(np.int64))


def test_native_label_order_is_explicit():
    x, y = arrays(102)
    dataset = Windows(x, y, 10, window=100)
    np.testing.assert_array_equal(dataset.targets, [2, 1, 0])


def test_parser_rejects_corrupt_schema_and_labels(tmp_path):
    path = tmp_path / "data.zip"
    rows = ["1 2 3\n"] * 149
    rows[144] = "1 9 3\n"
    with zipfile.ZipFile(path, "w") as stream:
        stream.writestr("nested/example.txt", "".join(rows))
    with pytest.raises(ValueError, match="native FI-2010 labels"):
        read_file(path, "example.txt")
    rows[144] = "1 2 3\n"
    with zipfile.ZipFile(path, "w") as stream:
        stream.writestr("nested/example.txt", "".join(rows))
    x, y = read_file(path, "example.txt")
    assert x.shape == (3, 40) and y.shape == (3, 5)
    with pytest.raises(ValueError, match="one FI-2010 member"):
        read_file(path, "absent.txt")


@pytest.mark.parametrize("name", ["deeplob", "translob"])
def test_models_train_and_reload_logits(name):
    model = build_model(name)
    inputs = torch.randn(4, 100, 40)
    loss = torch.nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1, 2, 0]))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    model.eval()
    restored = restore(name, 100, state_bytes(model))
    with torch.inference_mode():
        torch.testing.assert_close(model(inputs), restored(inputs), atol=0, rtol=0)


def test_translob_encoded_prefix_is_causal():
    model = build_model("translob").eval()
    inputs = torch.randn(2, 100, 40)
    changed = inputs.clone()
    changed[:, 60:] += 100
    with torch.inference_mode():
        original, perturbed = model.encode(inputs), model.encode(changed)
    torch.testing.assert_close(original[:, :60], perturbed[:, :60], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(original[:, 60:], perturbed[:, 60:])


def test_holdout_policy_rejects_changed_models_and_config():
    frozen = {"recipe": "abc", "models": {"deeplob": "123"}}
    authorize_test(None, frozen)
    authorize_test(frozen, json.loads(json.dumps(frozen)))
    with pytest.raises(ValueError, match="already opened"):
        authorize_test(frozen, {**frozen, "models": {"deeplob": "456"}})
    config = json.loads(Path("configs/lob.fi2010.json").read_text())
    validate(config)
    with pytest.raises(ValueError, match="purge"):
        validate({**config, "purge_observations": 0})
    with pytest.raises(ValueError, match="every eligible window"):
        validate({**config, "test_stride": 10})


def test_training_cache_verifies_recipe_and_weights(tmp_path):
    x, y = arrays()
    train = Windows(x, y, 10, 0, 300, stride=20)
    validation = Windows(x, y, 10, 350, 500, stride=20)
    config = {
        "seed": 1,
        "window": 100,
        "batch_size": 8,
        "max_epochs": 1,
        "patience": 1,
        "training_seconds_per_model": 60,
        "learning_rates": {"translob": 0.0001},
        "translob_hidden_l2": 0.0001,
    }
    body, result = fit("translob", 10, train, validation, config, tmp_path, "abc", lambda _: None)
    same, reused = fit("translob", 10, train, validation, config, tmp_path, "abc", lambda _: None)
    assert same == body and reused == result
    assert result["epochs_completed"] == 1 and result["best_epoch"] == 1
    p = predict(restore("translob", 100, body), validation, 8)
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-12)
    with pytest.raises(ValueError, match="frozen recipe"):
        fit("translob", 10, train, validation, config, tmp_path, "changed", lambda _: None)
    # Only state tensors are needed to restore inference; no executable upstream model pickle.
    assert isinstance(torch.load(io.BytesIO(body), weights_only=True), dict)
    # Simulate a disconnect after a best-weight update but before the latest epoch commit.
    (tmp_path / "training.json").unlink()
    (tmp_path / "best.pt").write_bytes(b"interrupted update")
    recovered, resumed = fit(
        "translob",
        10,
        train,
        validation,
        config,
        tmp_path,
        "abc",
        lambda _: None,
    )
    assert recovered == body and resumed["epochs_completed"] == 1


def test_train_only_does_not_parse_test_files(tmp_path, monkeypatch):
    from stock_predictor.storage.archive import Archive
    from stock_predictor_lob import pipeline
    from stock_predictor_lob.data import TRAIN_FILE
    from stock_predictor_lob.training import metrics

    x, y = arrays(5000)
    source = tmp_path / "source.zip"
    source.write_bytes(b"synthetic source fixture")
    config = json.loads(Path("configs/lob.fi2010.json").read_text())
    parsed = []

    def prepare(_root, _zip, name, _sha):
        assert name == TRAIN_FILE, "training-only phase opened a test member"
        parsed.append(name)
        return x, y

    def fit(name, horizon, train, validation, config, directory, recipe, log):
        result = {
            "validation": metrics(
                validation.targets, np.tile([0.3, 0.4, 0.3], (len(validation), 1))
            ),
            "epochs_completed": 1,
            "best_epoch": 1,
            "stop_reason": "fixture",
            "elapsed_seconds": 0.0,
            "train_class_counts": [50, 50, 50],
        }
        return state_bytes(build_model(name)), result

    monkeypatch.setattr(pipeline, "prepare", prepare)
    monkeypatch.setattr(pipeline, "download", lambda *_: (source, {"sha256": "fixture"}))
    monkeypatch.setattr(pipeline, "fit", fit)
    archive = Archive(tmp_path / "archive")
    result = pipeline.run(config, archive, tmp_path / "downloads", tmp_path / "out", True)
    assert len(result["models"]) == 4 and result["test_opened"] is False
    assert parsed == [TRAIN_FILE]
    assert archive.read(pipeline.POLICY_KEY) is None


def test_colab_profile_requires_gpu_and_full_windows():
    config = json.loads(Path("configs/lob.fi2010.colab.json").read_text())
    validate(config)
    assert config["device"] == "cuda"
    assert config["training_stride"] == config["validation_stride"] == 1
