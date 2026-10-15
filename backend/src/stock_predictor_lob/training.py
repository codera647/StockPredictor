import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from stock_predictor.storage.archive import digest
from stock_predictor.training.metrics import classification_metrics
from torch import nn
from torch.utils.data import DataLoader

from stock_predictor_lob.models import build_model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(dataset, batch, shuffle=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )


@torch.inference_mode()
def predict(model, dataset, batch, progress=None):
    model.eval()
    device = next(model.parameters()).device
    result = []
    last_log = time.monotonic()
    for i, (x, _) in enumerate(loader(dataset, batch)):
        # Float64 softmax makes probability serialization/metric validation unambiguous.
        result.append(model(x.to(device)).double().softmax(dim=1).cpu().numpy())
        if progress and time.monotonic() - last_log > 30:
            progress(f"inference {min((i + 1) * batch, len(dataset))}/{len(dataset)}")
            last_log = time.monotonic()
    return np.concatenate(result)


def metrics(y, probabilities):
    result = classification_metrics(y, probabilities)
    result["macro_f1"] = float(
        f1_score(
            y,
            probabilities.argmax(1),
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )
    return result


def state_bytes(model):
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return stream.getvalue()


def restore(name, window, body, device="cpu"):
    model = build_model(name, window)
    model.load_state_dict(torch.load(io.BytesIO(body), map_location="cpu", weights_only=True))
    return model.to(device).eval()


def fit(name, horizon, train, validation, config, directory: Path, recipe, log):
    """Epoch checkpoints resume the same frozen recipe, with no test access."""
    directory.mkdir(parents=True, exist_ok=True)
    best_path, latest_path = directory / "best.pt", directory / "latest.pt"
    result_path = directory / "training.json"
    seed = config["seed"] + horizon
    seed_all(seed)
    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA GPU is required by this recipe; select a Colab GPU runtime")
    model = build_model(name, config["window"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rates"][name])
    start_epoch, elapsed, stale, history, best_loss, best_epoch = 0, 0.0, 0, [], 1e99, 0
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["recipe"] != recipe or digest(best_path.read_bytes()) != result["model_sha256"]:
            raise ValueError("Completed LOB checkpoint does not match the frozen recipe")
        return best_path.read_bytes(), result
    if latest_path.exists():
        saved = torch.load(latest_path, map_location="cpu", weights_only=True)
        if saved["recipe"] != recipe:
            raise ValueError("LOB training checkpoint recipe mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng_state"])
        if device == "cuda" and saved.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        start_epoch, elapsed, stale = saved["epoch"], saved["elapsed"], saved["stale"]
        history, best_loss, best_epoch = saved["history"], saved["best_loss"], saved["best_epoch"]
        if not best_path.exists() or digest(best_path.read_bytes()) != saved["best_sha256"]:
            recovered = saved.get("best_body", b"")
            if digest(recovered) != saved["best_sha256"]:
                raise ValueError("Best LOB checkpoint is missing or corrupted")
            best_path.write_bytes(recovered)
    criterion = nn.CrossEntropyLoss()
    start_time = time.monotonic()
    stopped = "max_epochs"
    for epoch in range(start_epoch, config["max_epochs"]):
        if stale >= config["patience"]:
            stopped = "early_stopping"
            break
        budget_used = elapsed + time.monotonic() - start_time
        if history and budget_used >= config["training_seconds_per_model"]:
            stopped = "time_budget"
            break
        model.train()
        total_loss, total_rows = 0.0, 0
        last_log = time.monotonic()
        for x, y in loader(train, config["batch_size"], True, seed + epoch):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if name == "translob":
                loss = loss + config["translob_hidden_l2"] * model.hidden.weight.square().sum()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite LOB training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            total_rows += len(y)
            if time.monotonic() - last_log > 30:
                log(f"{name} k={horizon} epoch={epoch + 1} trained={total_rows}/{len(train)}")
                last_log = time.monotonic()
        probabilities = predict(model, validation, config["batch_size"])
        evaluation = metrics(validation.targets, probabilities)
        val_loss = evaluation["log_loss"]
        item = {
            "epoch": epoch + 1,
            "training_loss": total_loss / total_rows,
            "validation_log_loss": val_loss,
            "validation_accuracy": evaluation["accuracy"],
            "validation_macro_f1": evaluation["macro_f1"],
        }
        history.append(item)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            temporary = best_path.with_suffix(".tmp")
            temporary.write_bytes(state_bytes(model))
            temporary.replace(best_path)
        else:
            stale += 1
        elapsed_now = elapsed + time.monotonic() - start_time
        temporary = latest_path.with_suffix(".tmp")
        torch.save(
            {
                "recipe": recipe,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                "epoch": epoch + 1,
                "elapsed": elapsed_now,
                "stale": stale,
                "history": history,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "best_sha256": digest(best_path.read_bytes()),
                "best_body": best_path.read_bytes(),
            },
            temporary,
        )
        temporary.replace(latest_path)
        log(
            f"{name} k={horizon} epoch={epoch + 1}: val_accuracy={evaluation['accuracy']:.4f} "
            f"val_macro_f1={evaluation['macro_f1']:.4f}, elapsed={elapsed_now:.1f}s"
        )
        if stale >= config["patience"]:
            stopped = "early_stopping"
            break
    body = best_path.read_bytes()
    selected = restore(name, config["window"], body, device)
    probabilities = predict(selected, validation, config["batch_size"])
    result = {
        "recipe": recipe,
        "name": name,
        "horizon": horizon,
        "seed": seed,
        "model_sha256": digest(body),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "stop_reason": stopped,
        "elapsed_seconds": elapsed + time.monotonic() - start_time,
        "training_windows": len(train),
        "validation_windows": len(validation),
        "train_class_counts": np.bincount(train.targets, minlength=3).tolist(),
        "validation": metrics(validation.targets, probabilities),
        "history": history,
    }
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(result_path)
    return body, result
