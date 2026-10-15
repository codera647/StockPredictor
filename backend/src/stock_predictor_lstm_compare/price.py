"""Controlled port of the repository's price forecaster, with a train-only scaler."""

import io
import json
import logging
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from stock_predictor_lstm_compare.common import (
    PlainLSTM,
    canonical,
    digest,
    fetch,
    initialize,
    restore,
    save_json,
    source_hash,
    weights,
    write,
)

LOG = logging.getLogger(__name__)


def price_metrics(actual, predicted, origin):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or not np.isfinite(predicted).all() or (actual <= 0).any():
        raise ValueError("Invalid price comparison arrays")
    relative_rmse = float(np.sqrt(np.mean(((actual - predicted) / (actual + 1)) ** 2)))
    true_changes = np.sign(np.diff(np.r_[origin, actual]))
    predicted_changes = np.sign(np.diff(np.r_[origin, predicted]))
    return {
        "rows": len(actual),
        "repository_price_score_percent": 100 * (1 - relative_rmse),
        "mae": float(np.abs(actual - predicted).mean()),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "daily_path_direction_accuracy": float((true_changes == predicted_changes).mean()),
        "direction_definition": "sign of consecutive changes along each 30-day forecast path",
    }


@torch.inference_mode()
def forecast(model, history, days, lower, scale):
    model.eval()
    device = next(model.parameters()).device
    normalized = (np.asarray(history) - lower) / scale
    context = torch.tensor(normalized[:-1], dtype=torch.float32, device=device)[None, :, None]
    _, state = model.sequence(context)
    current = torch.tensor([[[normalized[-1]]]], dtype=torch.float32, device=device)
    predicted = []
    for _ in range(days):
        output, state = model.sequence(current, state)
        current = output[:, -1:, :]
        predicted.append(float(current.item()) * scale + lower)
    return np.asarray(predicted)


def run(config, downloads, output):
    initialize(config["device"], config["seed"], config["threads"])
    if config.get("production_ready") is not False:
        raise ValueError("Price check is research only")
    url = (
        "https://raw.githubusercontent.com/huseinzol05/Stock-Prediction-Models/"
        f"{config['upstream_revision']}/dataset/GOOG-year.csv"
    )
    source = fetch(downloads / "GOOG-year.csv", url, config["csv_sha256"])
    frame = pd.read_csv(source)
    dates = pd.to_datetime(frame.Date, errors="raise")
    prices = frame.Close.to_numpy(dtype=float)
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("Daily price dates must be strictly increasing")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Daily prices must be positive and finite")
    test_start = len(prices) - config["test_days"]
    val_start = test_start - config["validation_days"]
    if val_start < 20:
        raise ValueError("Insufficient training history")
    train = prices[:val_start]
    lower, scale = float(train.min()), float(train.max() - train.min())
    if scale <= 0:
        raise ValueError("Constant price history cannot fit this min-max scaler")
    recipe = {
        "config": config,
        "code_sha256": source_hash(),
        "torch": str(torch.__version__),
        "train_rows": val_start,
        "validation_rows": test_start - val_start,
        "test_rows": len(prices) - test_start,
        "scaler_min": lower,
        "scaler_range": scale,
    }
    recipe_sha = digest(canonical(recipe))
    directory = output / "price" / recipe_sha
    report_path = directory / "report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / "plan.json", recipe)
    model = PlainLSTM(1, 1, config["hidden"], config["dropout"]).to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    normalized = torch.tensor((train - lower) / scale, dtype=torch.float32, device=config["device"])
    best_loss, best_body, stale, history, selected_epoch = float("inf"), None, 0, [], 0
    start = time.monotonic()
    for epoch in range(config["max_epochs"]):
        model.train()
        state, losses = None, []
        for begin in range(0, len(train) - 1, config["chunk"]):
            end = min(begin + config["chunk"], len(train) - 1)
            inputs = normalized[begin:end][None, :, None]
            targets = normalized[begin + 1 : end + 1][None, :, None]
            optimizer.zero_grad(set_to_none=True)
            predicted, state = model.sequence(inputs, state)
            loss = nn.functional.mse_loss(predicted, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite price model loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            state = tuple(value.detach() for value in state)
            losses.append(float(loss.detach()))
        val_predictions = forecast(model, train, test_start - val_start, lower, scale)
        val_rmse = float(np.sqrt(np.mean((prices[val_start:test_start] - val_predictions) ** 2)))
        history.append(
            {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_rmse": val_rmse}
        )
        if val_rmse < best_loss:
            best_loss, best_body, stale, selected_epoch = val_rmse, weights(model), 0, epoch + 1
            write(directory / "best.pt", best_body)
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            LOG.info("Price LSTM epoch=%d validation_RMSE=%.4f", epoch + 1, val_rmse)
        if stale >= config["patience"]:
            break
    selected = restore(best_body, 1, 1, config, config["device"])
    prediction = forecast(selected, prices[:test_start], len(prices) - test_start, lower, scale)
    reload_prediction = forecast(
        restore(best_body, 1, 1, config, config["device"]),
        prices[:test_start],
        len(prices) - test_start,
        lower,
        scale,
    )
    np.testing.assert_allclose(prediction, reload_prediction, atol=1e-8, rtol=0)
    actual, origin = prices[test_start:], float(prices[test_start - 1])
    naive = np.repeat(origin, len(actual))
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        actual=actual,
        predicted=prediction,
        persistence=naive,
        dates=dates.iloc[test_start:].dt.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
    )
    write(directory / "predictions.npz", buffer.getvalue())
    report = {
        "recipe": recipe_sha,
        "source_url": url,
        "source_sha256": config["csv_sha256"],
        "model": "repository-inspired 1-layer 128-unit LSTM price regressor",
        "mode": "30-day recursive forecast with actual history available at forecast origin",
        "best_epoch": selected_epoch,
        "epochs": len(history),
        "history": history,
        "elapsed_seconds": time.monotonic() - start,
        "lstm": price_metrics(actual, prediction, origin),
        "persistence": price_metrics(actual, naive, origin),
        "model_sha256": digest(best_body),
        "predictions_sha256": digest(buffer.getvalue()),
        "scaler_fit_rows": val_start,
        "test_first_date": str(dates.iloc[test_start].date()),
        "test_last_date": str(dates.iloc[-1].date()),
        "comparable_to_deeplob_classification_accuracy": False,
        "production_ready": False,
        "method_changes": [
            "PyTorch port rather than original legacy TensorFlow execution",
            "192/30/30 chronological split reserves validation",
            "Min-max scaler fitted on training only",
            "Dropout disabled at inference; one fixed seed",
            "Select minimum validation recursive-forecast RMSE; no test tuning",
        ],
    }
    save_json(report_path, report)
    save_json(output / "price_report.json", report)
    return report
