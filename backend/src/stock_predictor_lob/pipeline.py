import argparse
import io
import json
import logging
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from stock_predictor.config import PipelineError, Settings, require, safe_url
from stock_predictor.storage.archive import Archive, digest, json_bytes

from stock_predictor_lob.data import (
    TEST_FILES,
    TRAIN_FILE,
    Windows,
    chronological_ranges,
    download,
    prepare,
)
from stock_predictor_lob.training import fit, metrics, predict, restore

LOG = logging.getLogger(__name__)
POLICY_KEY = "checkpoints/lob/fi2010-test-policy.json"
COMPLETED_KEY = "checkpoints/lob/completed.json"


def validate(config):
    if config.get("schema_version") != 1 or config.get("production_ready") is not False:
        raise ValueError("LOB benchmark must remain a separate research experiment")
    if config.get("architectures") != ["deeplob", "translob"]:
        raise ValueError("The frozen comparison requires DeepLOB and TransLOB")
    if config.get("horizons") != [10, 100] or config.get("test_stride") != 1:
        raise ValueError("The frozen test requires horizons 10/100 and every eligible window")
    if config.get("purge_observations", 0) < max(config["horizons"]):
        raise ValueError("Chronological split purge must cover all target horizons")
    if not 0.5 <= config["training_fraction"] <= 0.9 or config["window"] != 100:
        raise ValueError("Unsupported chronological split or input window")
    for key in (
        "training_stride",
        "validation_stride",
        "batch_size",
        "max_epochs",
        "patience",
        "training_seconds_per_model",
        "threads",
    ):
        if config.get(key, 0) < 1:
            raise ValueError(f"Invalid LOB setting: {key}")
    if len(config["source_sha256"]) != 64:
        raise ValueError("FI-2010 source must have a pinned SHA-256")
    if config.get("device", "cpu") not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or cuda")


def code_hash():
    return digest(
        json_bytes(
            {
                path.name: digest(path.read_text(encoding="utf-8").encode())
                for path in sorted(Path(__file__).parent.glob("*.py"))
            }
        )
    )


def storage(env_file, root, upload):
    if not upload:
        return Archive(root)
    load_dotenv(env_file, override=False)
    # This public-data experiment needs only the existing private artifact-storage credentials.
    settings = Settings(
        alpaca_base="https://data.alpaca.markets",
        alpaca_feed="sip",
        alpaca_key="",
        alpaca_secret="",
        s3_endpoint=safe_url(require("S3_ENDPOINT_URL"), "S3_ENDPOINT_URL"),
        s3_region=require("S3_REGION"),
        s3_bucket=require("S3_BUCKET_NAME"),
        s3_key=require("S3_ACCESS_KEY_ID"),
        s3_secret=require("S3_SECRET_ACCESS_KEY"),
    )
    archive = Archive(root, settings)
    archive.check_access()
    return archive


def store_blob(archive, category, extension, body):
    return archive.write(f"lob/{category}/{digest(body)}.{extension}", body)


def checked_json(archive, ref):
    raw = archive.read(ref["key"], sha256=ref["sha256"])
    if raw is None:
        raise ValueError("Missing benchmark artifact")
    return json.loads(raw)


def authorize_test(existing, frozen):
    if existing is not None and existing != frozen:
        raise ValueError(
            "FI-2010 test already opened for another recipe/model set; "
            "refusing automatic tuning against an exposed holdout"
        )


def write_report(output, report, ref):
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    ready_path = output / "training_ready.json"
    if ready_path.exists():
        ready = json.loads(ready_path.read_text())
        ready["test_opened"] = True
        ready_path.write_text(json.dumps(ready, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "report": ref,
                "recipe": report["recipe"],
                "models": {
                    k: {
                        "accuracy": m["test"]["accuracy"],
                        "macro_f1": m["test"]["macro_f1"],
                        "epochs": m["epochs"],
                        "best_epoch": m["best_epoch"],
                    }
                    for k, m in report["models"].items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def verify_completed(archive, output):
    pointer = archive.read(COMPLETED_KEY, remote=bool(archive.client))
    if pointer is None:
        return None
    report_ref = json.loads(pointer)
    checked = []

    def read(ref):
        raw = archive.read(ref["key"], sha256=ref["sha256"], remote=bool(archive.client))
        if raw is None:
            raise ValueError("Missing remote FI-2010 artifact")
        checked.append(ref["key"])
        return raw

    report = json.loads(read(report_ref))
    source = json.loads(read(report["source"]))
    read(source["artifact"])
    json.loads(read(report["plan"]))
    recomputed = {}
    for identifier, model in report["models"].items():
        restore(model["name"], 100, read(model["model"]))
        json.loads(read(model["training"]))
        with np.load(io.BytesIO(read(model["predictions"])), allow_pickle=False) as predictions:
            measured = metrics(predictions["labels"], predictions["probabilities"])
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier_score"):
            if not np.isclose(measured[key], model["test"][key], atol=1e-12, rtol=0):
                raise ValueError("Archived FI-2010 metrics failed recomputation")
        recomputed[identifier] = {k: measured[k] for k in ("accuracy", "macro_f1")}
    write_report(output, report, report_ref)
    verification = {
        "recipe": report["recipe"],
        "remote": bool(archive.client),
        "verified_objects": len(set(checked)),
        "recomputed": recomputed,
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    return verification


def run(config, archive, downloads, output, train_only=False):
    validate(config)
    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA GPU unavailable; select Runtime > Change runtime type > GPU")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config["threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    recipe = {
        "config": config,
        "code_sha256": code_hash(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.system(),
    }
    recipe_sha = digest(json_bytes(recipe))
    pointer_key = f"checkpoints/lob/{recipe_sha}.json"
    completed = archive.read(pointer_key)
    if completed:
        report_ref = json.loads(completed)
        report = checked_json(archive, report_ref)
        archive.json(COMPLETED_KEY, report_ref)
        write_report(output, report, report_ref)
        return {
            "cache_reused": True,
            "recipe": recipe_sha,
            "models": {
                k: {"accuracy": m["test"]["accuracy"], "macro_f1": m["test"]["macro_f1"]}
                for k, m in report["models"].items()
            },
        }
    existing_raw = archive.read(POLICY_KEY)
    existing_policy = json.loads(existing_raw) if existing_raw else None
    if existing_policy and existing_policy["recipe"] != recipe_sha:
        raise ValueError("FI-2010 holdout already exposed; reuse its recorded frozen recipe")
    zip_path, source = download(downloads, config["source_sha256"])
    source["artifact"] = store_blob(archive, "source", "zip", zip_path.read_bytes())
    source_ref = store_blob(archive, "source", "json", json_bytes(source))
    train_x, train_y = prepare(downloads, zip_path, TRAIN_FILE, config["source_sha256"])
    ranges = chronological_ranges(
        len(train_x), config["training_fraction"], config["purge_observations"], config["window"]
    )
    # Discard each file's final maximum-horizon endpoints instead of trusting truncated labels.
    ranges["validation"] = (ranges["validation"][0], len(train_x) - max(config["horizons"]))
    plan = {
        "recipe": recipe,
        "recipe_sha256": recipe_sha,
        "source": source_ref,
        "train_file_observations": len(train_x),
        "ranges": ranges,
        "classes": ["DOWN", "FLAT", "UP"],
        "native_mapping": {"1": "UP", "2": "FLAT", "3": "DOWN"},
        "test_files": TEST_FILES,
        "test_opened": False,
        "selection": "minimum chronological validation log loss; test never selects models",
    }
    plan_ref = store_blob(archive, "plans", "json", json_bytes(plan))
    LOG.info("Frozen plan %s; source/range checks complete", recipe_sha)
    models = {}
    for horizon in config["horizons"]:
        train = Windows(
            train_x, train_y, horizon, *ranges["train"], config["window"], config["training_stride"]
        )
        validation = Windows(
            train_x,
            train_y,
            horizon,
            *ranges["validation"],
            config["window"],
            config["validation_stride"],
        )
        for name in config["architectures"]:
            identifier = f"{name}_k{horizon}"
            if existing_policy:
                models[identifier] = existing_policy["models"][identifier]
                continue
            LOG.info("Training %s: train=%d validation=%d", identifier, len(train), len(validation))
            body, training = fit(
                name,
                horizon,
                train,
                validation,
                config,
                output / recipe_sha / identifier,
                recipe_sha,
                LOG.info,
            )
            model_ref = store_blob(archive, "models", "pt", body)
            training_ref = store_blob(archive, "training", "json", json_bytes(training))
            models[identifier] = {
                "name": name,
                "horizon": horizon,
                "model": model_ref,
                "training": training_ref,
                "validation": training["validation"],
                "epochs": training["epochs_completed"],
                "best_epoch": training["best_epoch"],
                "stop_reason": training["stop_reason"],
                "training_seconds": training["elapsed_seconds"],
                "train_class_counts": training["train_class_counts"],
            }
    frozen = {"recipe": recipe_sha, "models": models}
    if train_only:
        selection_ref = store_blob(archive, "selections", "json", json_bytes(frozen))
        output.mkdir(parents=True, exist_ok=True)
        ready = {
            "recipe": recipe_sha,
            "test_opened": existing_policy is not None,
            "selection": selection_ref,
            "models": {
                k: {
                    "validation_accuracy": m["validation"]["accuracy"],
                    "validation_macro_f1": m["validation"]["macro_f1"],
                    "epochs": m["epochs"],
                }
                for k, m in models.items()
            },
        }
        (output / "training_ready.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
        return ready
    authorize_test(existing_policy, frozen)
    # All architectures/horizons are frozen before any test file is parsed or evaluated.
    archive.json(POLICY_KEY, frozen)
    LOG.info("All four checkpoints frozen. Opening three test-day files once for this recipe.")
    test_arrays = [
        prepare(downloads, zip_path, name, config["source_sha256"]) for name in TEST_FILES
    ]
    test_counts = {}
    for identifier, entry in models.items():
        name, horizon = entry["name"], entry["horizon"]
        body = archive.read(entry["model"]["key"], sha256=entry["model"]["sha256"])
        if body is None:
            raise ValueError("Missing frozen model")
        model = restore(name, config["window"], body, device)
        probabilities, labels, files, ends = [], [], [], []
        per_file = {}
        for file_id, ((x, y), member) in enumerate(zip(test_arrays, TEST_FILES, strict=True)):
            dataset = Windows(
                x,
                y,
                horizon,
                stop=len(x) - max(config["horizons"]),
                window=config["window"],
                stride=config["test_stride"],
            )
            LOG.info("Evaluating %s / %s (%d windows)", identifier, member, len(dataset))
            p = predict(model, dataset, config["batch_size"], LOG.info)
            reloaded = restore(name, config["window"], body, device)
            probe = torch.stack([dataset[i][0] for i in range(min(8, len(dataset)))])
            with torch.inference_mode():
                check = reloaded(probe.to(device)).double().softmax(1).cpu().numpy()
            if not np.allclose(p[: len(probe)], check, atol=1e-6, rtol=1e-6):
                raise ValueError("Restored model changed test probabilities")
            probabilities.append(p)
            labels.append(dataset.targets)
            files.append(np.full(len(dataset), file_id, dtype=np.uint8))
            ends.append(dataset.ends)
            per_file[member] = metrics(dataset.targets, p)
            test_counts[member] = {"observations": len(x), "windows": len(dataset)}
        p, y = np.concatenate(probabilities), np.concatenate(labels)
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            probabilities=p,
            labels=y,
            file_id=np.concatenate(files),
            endpoint=np.concatenate(ends),
        )
        prediction_ref = store_blob(archive, "predictions", "npz", buffer.getvalue())
        prior = np.array(entry["train_class_counts"], dtype=float)
        prior /= prior.sum()
        entry["test"] = metrics(y, p)
        entry["test_by_file"] = per_file
        entry["training_prior_baseline"] = metrics(y, np.tile(prior, (len(y), 1)))
        entry["predictions"] = prediction_ref
        entry["reload_verified"] = True
        LOG.info(
            "%s test accuracy=%.4f macro_f1=%.4f",
            identifier,
            entry["test"]["accuracy"],
            entry["test"]["macro_f1"],
        )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "recipe": recipe_sha,
        "plan": plan_ref,
        "source": source_ref,
        "models": models,
        "test_files": test_counts,
        "production_ready": False,
        "above_80_percent_accuracy": [k for k, m in models.items() if m["test"]["accuracy"] > 0.8],
        "limitations": [
            "Training budgets, decimal normalization and optimizer/implementation settings "
            "differ from published paper protocols; not an exact reproduction.",
            "Public files lack instrument IDs and internal session boundaries. Windows "
            "respect file and split boundaries; within-file boundaries cannot be verified.",
            "Ten historical Finnish-stock days are not evidence for US daily forecasts.",
            "Original daily holdout, production model and deployment remain unchanged.",
            "FI-2010 source licensing/redistribution rights need separate review; "
            "download is public, not a grant of commercial data rights.",
        ],
    }
    ref = store_blob(archive, "reports", "json", json_bytes(report))
    archive.json(pointer_key, ref)
    archive.json(COMPLETED_KEY, ref)
    write_report(output, report, ref)
    return {
        "cache_reused": False,
        "recipe": recipe_sha,
        "report": ref,
        "models": {
            k: {"accuracy": m["test"]["accuracy"], "macro_f1": m["test"]["macro_f1"]}
            for k, m in models.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Separate FI-2010 DeepLOB/TransLOB benchmark")
    parser.add_argument("--config", type=Path, default=Path("configs/lob.fi2010.json"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--archive", type=Path, default=Path("data/archive"))
    parser.add_argument("--downloads", type=Path, default=Path("data/downloads/fi2010"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/lob"))
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Fit/select all four models and leave the test files closed",
    )
    parser.add_argument(
        "--reuse-completed",
        action="store_true",
        help="Verify the frozen report instead of refitting on an exposed test",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        archive = storage(args.env, args.archive, not args.no_upload)
        result = verify_completed(archive, args.output) if args.reuse_completed else None
        if result is None:
            result = run(config, archive, args.downloads, args.output, args.train_only)
        print(json.dumps(result, indent=2), flush=True)
    except (PipelineError, ValueError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
