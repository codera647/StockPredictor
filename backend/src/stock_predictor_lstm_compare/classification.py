"""Retrospective plain-LSTM comparison on exactly the exposed DeepLOB k=10 test rows."""

import io
import json
import logging
import time

import numpy as np
import torch
from stock_predictor_lob.data import TEST_FILES, TRAIN_FILE, Windows, chronological_ranges, prepare
from stock_predictor_lob.data import download as download_lob
from stock_predictor_lob.training import loader, metrics, predict
from torch import nn

from stock_predictor_lstm_compare.common import (
    PlainLSTM,
    canonical,
    digest,
    initialize,
    restore,
    row_digest,
    save_json,
    source_hash,
    weights,
    write,
)

LOG = logging.getLogger(__name__)


def validate(config, baseline):
    if (
        config["window"] != 100
        or config["horizon"] != 10
        or config.get("production_ready") is not False
        or config.get("test_already_exposed") is not True
    ):
        raise ValueError("Comparison requires the frozen 100-input/horizon-10 research task")
    if any(config[k] != 1 for k in ("training_stride", "validation_stride", "test_stride")):
        raise ValueError("All eligible windows are required for the matched comparison")
    if config["source_sha256"] != baseline["dataset_sha256"] or baseline["horizon"] != 10:
        raise ValueError("Comparison source/target does not match the DeepLOB baseline")
    if any(config[k] < 1 for k in ("max_epochs", "patience", "batch_size", "training_seconds")):
        raise ValueError("Invalid training budget")


def checkpoint(path, state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    write(path, buffer.getvalue())


def fit(train, validation, config, directory, recipe):
    result_path = directory / "training.json"
    best_path = directory / "best.pt"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        body = best_path.read_bytes()
        if result["recipe"] != recipe or digest(body) != result["weights_sha256"]:
            raise ValueError("Completed comparison checkpoint mismatch")
        return body, result
    device = config["device"]
    model = PlainLSTM(40, 3, config["hidden"], config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    latest = directory / "latest.pt"
    history, elapsed, stale, best_loss, best_body, best_epoch = [], 0.0, 0, 1e99, b"", 0
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if saved["recipe"] != recipe:
            raise ValueError("Comparison epoch checkpoint recipe mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        history, elapsed, stale = saved["history"], saved["elapsed"], saved["stale"]
        best_loss, best_body, best_epoch = (
            saved["best_loss"],
            saved["best_body"],
            saved["best_epoch"],
        )
    start = time.monotonic()
    stop = "max_epochs"
    for epoch in range(len(history), config["max_epochs"]):
        if stale >= config["patience"]:
            stop = "early_stopping"
            break
        if history and elapsed + time.monotonic() - start >= config["training_seconds"]:
            stop = "time_budget"
            break
        model.train()
        loss_sum, rows, last_log = 0.0, 0, time.monotonic()
        for x, y in loader(train, config["batch_size"], True, config["seed"] + epoch):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite classification loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            rows += len(y)
            if time.monotonic() - last_log > 30:
                LOG.info("Plain LSTM epoch=%d training=%d/%d", epoch + 1, rows, len(train))
                last_log = time.monotonic()
        evaluated = metrics(validation.targets, predict(model, validation, config["batch_size"]))
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": loss_sum / rows,
                "validation_log_loss": evaluated["log_loss"],
                "validation_accuracy": evaluated["accuracy"],
                "validation_macro_f1": evaluated["macro_f1"],
            }
        )
        if evaluated["log_loss"] < best_loss:
            best_loss, best_body, stale, best_epoch = (
                evaluated["log_loss"],
                weights(model),
                0,
                epoch + 1,
            )
        else:
            stale += 1
        state = {
            "recipe": recipe,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
            "elapsed": elapsed + time.monotonic() - start,
            "stale": stale,
            "history": history,
            "best_loss": best_loss,
            "best_body": best_body,
            "best_epoch": best_epoch,
        }
        checkpoint(latest, state)
        write(best_path, best_body)
        LOG.info(
            "Plain LSTM epoch=%d val_accuracy=%.4f val_macro_f1=%.4f",
            epoch + 1,
            evaluated["accuracy"],
            evaluated["macro_f1"],
        )
    write(best_path, best_body)
    model = restore(best_body, 40, 3, config, device)
    result = {
        "recipe": recipe,
        "weights_sha256": digest(best_body),
        "history": history,
        "epochs": len(history),
        "best_epoch": best_epoch,
        "stop_reason": stop,
        "elapsed_seconds": elapsed + time.monotonic() - start,
        "train_windows": len(train),
        "validation_windows": len(validation),
        "train_class_counts": np.bincount(train.targets, minlength=3).tolist(),
        "validation": metrics(validation.targets, predict(model, validation, config["batch_size"])),
    }
    save_json(result_path, result)
    return best_body, result


def run(config, baseline, downloads, output, evaluate=False):
    validate(config, baseline)
    initialize(config["device"], config["seed"], config["threads"])
    recipe = {
        "config": config,
        "baseline": baseline,
        "code_sha256": source_hash(),
        "torch": str(torch.__version__),
    }
    recipe_sha = digest(canonical(recipe))
    policy_path = output / "classification_frozen.json"
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else None
    if policy and policy["recipe"] != recipe_sha:
        raise ValueError("Comparison test already opened; refusing a changed recipe")
    directory = output / "classification" / recipe_sha
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / "plan.json", recipe)
    report_path = directory / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        saved = np.load(directory / "predictions.npz", allow_pickle=False)
        if digest((directory / "best.pt").read_bytes()) != report["weights_sha256"]:
            raise ValueError("Cached comparison model is corrupted")
        if digest((directory / "predictions.npz").read_bytes()) != report["predictions_sha256"]:
            raise ValueError("Cached comparison predictions are corrupted")
        recomputed = metrics(saved["labels"], saved["probabilities"])
        saved.close()
        if not np.isclose(recomputed["accuracy"], report["test"]["accuracy"], atol=1e-12):
            raise ValueError("Cached comparison accuracy failed recomputation")
        save_json(output / "classification_report.json", report)
        return report
    zip_path, source = download_lob(downloads, config["source_sha256"])
    x, y = prepare(downloads, zip_path, TRAIN_FILE, config["source_sha256"])
    ranges = chronological_ranges(len(x), 0.8, 100, 100)
    train = Windows(x, y, 10, *ranges["train"], window=100)
    val = Windows(x, y, 10, ranges["validation"][0], len(x) - 100, window=100)
    if policy:
        body = (directory / "best.pt").read_bytes()
        training = json.loads((directory / "training.json").read_text())
    else:
        body, training = fit(train, val, config, directory, recipe_sha)
    selected = {"recipe": recipe_sha, "weights_sha256": digest(body)}
    if policy and policy != selected:
        raise ValueError("Frozen comparison weights changed")
    if not evaluate:
        ready = {
            "recipe": recipe_sha,
            "test_opened": policy is not None,
            "validation": training["validation"],
            "epochs": training["epochs"],
            "best_epoch": training["best_epoch"],
        }
        save_json(output / "classification_training_ready.json", ready)
        return ready
    save_json(policy_path, selected)
    model = restore(body, 40, 3, config, config["device"])
    ps, ys, ids, ends, by_file = [], [], [], [], {}
    for index, member in enumerate(TEST_FILES):
        test_x, test_y = prepare(downloads, zip_path, member, config["source_sha256"])
        dataset = Windows(test_x, test_y, 10, stop=len(test_x) - 100, window=100)
        LOG.info("Plain LSTM final evaluation %s: %d windows", member, len(dataset))
        p = predict(model, dataset, config["batch_size"], LOG.info)
        ps.append(p)
        ys.append(dataset.targets)
        ids.append(np.full(len(dataset), index, dtype=np.uint8))
        ends.append(dataset.ends)
        by_file[member] = metrics(dataset.targets, p)
        probe = torch.stack([dataset[i][0] for i in range(min(8, len(dataset)))])
        reloaded = restore(body, 40, 3, config, config["device"])
        with torch.inference_mode():
            restored = reloaded(probe.to(config["device"])).double().softmax(1).cpu().numpy()
        np.testing.assert_allclose(p[: len(probe)], restored, atol=1e-5, rtol=1e-5)
    p, y, ids, ends = map(np.concatenate, (ps, ys, ids, ends))
    alignment = row_digest(ids, ends, y)
    if len(y) != baseline["rows"] or alignment != baseline["row_alignment_sha256"]:
        raise ValueError("Test rows or labels do not exactly match the verified DeepLOB baseline")
    result = metrics(y, p)
    prior = np.array(training["train_class_counts"], dtype=float)
    prior /= prior.sum()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, probabilities=p, labels=y, file_id=ids, endpoint=ends)
    write(directory / "predictions.npz", buffer.getvalue())
    report = {
        "recipe": recipe_sha,
        "source": source,
        "model": "plain_128_unit_lstm_classifier",
        "weights_sha256": digest(body),
        "predictions_sha256": digest(buffer.getvalue()),
        "row_alignment_sha256": alignment,
        "test": result,
        "test_by_file": by_file,
        "training_prior_baseline": metrics(y, np.tile(prior, (len(y), 1))),
        "deeplob_baseline": baseline,
        "validation": training["validation"],
        "epochs": training["epochs"],
        "best_epoch": training["best_epoch"],
        "beats_deeplob_accuracy": result["accuracy"] > baseline["accuracy"],
        "accuracy_difference_percentage_points": 100 * (result["accuracy"] - baseline["accuracy"]),
        "balanced_accuracy_difference_percentage_points": 100
        * (result["balanced_accuracy"] - baseline["balanced_accuracy"]),
        "macro_f1_difference_percentage_points": 100 * (result["macro_f1"] - baseline["macro_f1"]),
        "retrospective_exposed_test_comparison": True,
        "production_ready": False,
        "not_an_exact_reproduction_of_repository_price_forecaster": True,
    }
    save_json(report_path, report)
    save_json(output / "classification_report.json", report)
    return report
