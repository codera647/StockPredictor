"""Recover the original Colab export collision without changing frozen package sources."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np


def resume(config_path, archive_path, downloads, output):
    from stock_predictor.storage.archive import Archive, digest, json_bytes
    from stock_predictor_lob import pipeline

    archive = Archive(archive_path)
    frozen_raw = archive.read(pipeline.POLICY_KEY)
    if frozen_raw is None:
        raise ValueError("No frozen final-test policy found; refusing to train in recovery mode")
    frozen = json.loads(frozen_raw)
    expected = {"deeplob_k10", "translob_k10", "deeplob_k100", "translob_k100"}
    if set(frozen["models"]) != expected:
        raise ValueError("Recovery requires the original four frozen models")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for model in frozen["models"].values():
        for ref in (model["model"], model["training"]):
            if archive.read(ref["key"], sha256=ref["sha256"]) is None:
                raise ValueError("Missing frozen model/training artifact; refusing recovery")
    before = pipeline.code_hash()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "fix": "rename prediction array file to file_id at the NumPy export boundary",
        "frozen_recipe": frozen["recipe"],
        "package_code_sha256": before,
        "recovery_script_sha256": digest(Path(__file__).read_bytes()),
        "frozen_models": {key: value["model"] for key, value in frozen["models"].items()},
        "retraining_permitted": False,
    }
    archive.write(f"lob/recovery/{digest(json_bytes(provenance))}.json", json_bytes(provenance))
    (output / "export_recovery.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    original_save = np.savez_compressed
    original_fit = pipeline.fit

    def save_arrays(destination, *args, **arrays):
        # The original pipeline uses the reserved keyword "file" for its file-ID array.
        if "file" in arrays:
            if "file_id" in arrays:
                raise ValueError("Ambiguous prediction file identifiers")
            arrays["file_id"] = arrays.pop("file")
        return original_save(destination, *args, **arrays)

    def no_training(*_args, **_kwargs):
        raise RuntimeError("Recovery cannot train; only frozen checkpoint evaluation is allowed")

    np.savez_compressed = save_arrays
    pipeline.fit = no_training
    try:
        print("Resuming frozen final-test evaluation; no training will run.", flush=True)
        result = pipeline.run(config, archive, downloads, output)
        verified = pipeline.verify_completed(archive, output)
        if archive.read(pipeline.POLICY_KEY) != frozen_raw or pipeline.code_hash() != before:
            raise RuntimeError("Recovery changed the frozen policy or package source")
        print(json.dumps({"evaluation": result, "verification": verified}, indent=2), flush=True)
        return result
    finally:
        np.savez_compressed = original_save
        pipeline.fit = original_fit


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resume(args.config, args.archive, args.downloads, args.output)


if __name__ == "__main__":
    main()
