"""Build a self-contained notebook and private local FI-2010 handoff; never include .env."""

import base64
import hashlib
import io
import json
import shutil
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/colab"
NOTEBOOK = ROOT / "notebooks/StockPredictor_FI2010_Colab.ipynb"
SOURCE = ROOT / "data/downloads/fi2010/data.zip"
EXPECTED = "7a5564290a504bdc104dccaf44aa6c0b3857bf1db6b0517b9b88577c2def3e5f"


def sha(body):
    return hashlib.sha256(body).hexdigest()


def zip_bytes(files):
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, body)
    return result.getvalue()


def source_files():
    names = [
        "backend/src/stock_predictor/__init__.py",
        "backend/src/stock_predictor/config.py",
        "backend/src/stock_predictor/storage/__init__.py",
        "backend/src/stock_predictor/storage/archive.py",
        "backend/src/stock_predictor/training/__init__.py",
        "backend/src/stock_predictor/training/metrics.py",
        "backend/tests/test_lob.py",
        "configs/lob.fi2010.json",
        "configs/lob.fi2010.colab.json",
        "notebooks/requirements-colab.txt",
    ]
    names.extend(
        p.relative_to(ROOT).as_posix()
        for p in sorted((ROOT / "backend/src/stock_predictor_lob").glob("*.py"))
    )
    files = {name: (ROOT / name).read_bytes() for name in names}
    files["BUNDLE_MANIFEST.json"] = json.dumps(
        {name: sha(body) for name, body in files.items()},
        sort_keys=True,
        indent=2,
    ).encode()
    return files


def cell(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    result = {
        "cell_type": kind,
        "metadata": {},
        "id": sha(source.encode())[:12],
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        compile(source, "notebook cell", "exec")
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook(bundle):
    encoded = base64.b64encode(bundle).decode()
    cells = [
        cell(
            "markdown",
            """
        # StockPredictor: FI-2010 DeepLOB + TransLOB on a free Colab GPU

        Complete, self-contained training notebook. Select **Runtime → Change runtime type → GPU**,
        then run the cells in order. The notebook keeps Colab's CUDA-enabled PyTorch and needs no
        API keys, `.env`, GitHub token or paid service. A GPU must be available; free Colab access
        and runtime duration are variable: https://research.google.com/colaboratory/faq.html.

        This trains **four models**: DeepLOB and TransLOB, each at horizons 10 and 100. It uses
        every eligible rolling window, 100 observations × 40 order-book features, chronological
        train/validation, a 100-observation purge, and three untouched test-day files. Model and
        optimizer checkpoints persist to your Drive after every complete epoch. Rerunning the
        same notebook/configuration resumes the last saved epoch; an interrupted epoch repeats.

        FI-2010 contains ten historical days from five Finnish stocks. Its event-based results
        do not measure our five-session US daily forecasts. Published 80–90% scores are **not
        guaranteed** here; normalization, optimizer and framework details differ. Source data
        are publicly downloadable; review original data terms before commercial use.
        """,
        ),
        cell(
            "code",
            """
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        from pathlib import Path
        from google.colab import drive

        drive.mount("/content/drive")
        DRIVE_ROOT = Path("/content/drive/MyDrive/StockPredictor_FI2010")
        DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
        WORK = Path("/content/StockPredictor_FI2010_code")
        DOWNLOADS = Path("/content/StockPredictor_FI2010_data")
        ARCHIVE = DRIVE_ROOT / "archive"
        RESULTS = DRIVE_ROOT / "results"
        print("Persistent checkpoints and results:", DRIVE_ROOT)
        """,
        ),
        cell(
            "markdown",
            """
        ## Install the bundled code and small dependencies

        The source is embedded below with a checksum, so no repository authentication is needed.
        This does **not** install the project's CPU-only Torch dependency. Keep the bundled code
        unchanged while resuming a run: the source/version/configuration hash identifies it.
        """,
        ),
        cell(
            "code",
            f"""
        import base64, hashlib, io, json, subprocess, sys, zipfile

        EMBEDDED_SOURCE = "{encoded}"
        SOURCE_BUNDLE_SHA256 = "{sha(bundle)}"
        payload = base64.b64decode(EMBEDDED_SOURCE)
        assert hashlib.sha256(payload).hexdigest() == SOURCE_BUNDLE_SHA256
        WORK.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as source_archive:
            for member in source_archive.infolist():
                target = (WORK / member.filename).resolve()
                if not target.is_relative_to(WORK.resolve()):
                    raise ValueError("Unsafe source-bundle path")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source_archive.read(member))
        manifest = json.loads((WORK / "BUNDLE_MANIFEST.json").read_text())
        for name, expected in manifest.items():
            assert hashlib.sha256((WORK / name).read_bytes()).hexdigest() == expected, name
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                        str(WORK / "notebooks/requirements-colab.txt")], check=True)
        sys.path.insert(0, str(WORK / "backend/src"))
        print("Verified bundled files:", len(manifest))
        """,
        ),
        cell(
            "code",
            """
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU. Select Runtime → Change runtime type → GPU.")
        print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
        environment = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(0), "python": sys.version}
        (DRIVE_ROOT / "environment.json").write_text(json.dumps(environment, indent=2))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(WORK / "backend/src")
        subprocess.run([sys.executable, "-m", "pytest", "-q", "backend/tests/test_lob.py"],
                       cwd=WORK, env=env, check=True)
        """,
        ),
        cell(
            "markdown",
            """
        ## Complete dataset

        Default: download the complete 56,278,154-byte public ZIP automatically. No account/key.
        If you uploaded the supplied `fi2010_complete.zip` to Drive, set `DATASET_ZIP` below to
        its full path. Both options use identical SHA-256-verified bytes. Original files contain
        149 rows: 40 raw features, 104 engineered rows (unused), and five supplied target rows.
        Targets are mapped to **DOWN=0, FLAT=1, UP=2**. No daily dataset is substituted here.
        """,
        ),
        cell(
            "code",
            """
        import shutil
        from stock_predictor_lob.data import checksum, download, prepare, TRAIN_FILE, Windows
        from stock_predictor_lob.data import chronological_ranges

        DATASET_ZIP = ""  # Optional: /content/drive/MyDrive/.../fi2010_complete.zip
        config = json.loads((WORK / "configs/lob.fi2010.colab.json").read_text())
        DOWNLOADS.mkdir(parents=True, exist_ok=True)
        if DATASET_ZIP:
            supplied = Path(DATASET_ZIP)
            assert checksum(supplied) == config["source_sha256"], "Dataset checksum mismatch"
            shutil.copyfile(supplied, DOWNLOADS / "data.zip")
        zip_path, source = download(DOWNLOADS, config["source_sha256"])
        print("Complete dataset:", zip_path, source["bytes"], "bytes")
        print("SHA-256:", source["sha256"])
        # Inspect only the official training member here. Test files stay unparsed until evaluation.
        train_x, train_y = prepare(DOWNLOADS, zip_path, TRAIN_FILE, config["source_sha256"])
        ranges = chronological_ranges(len(train_x), config["training_fraction"],
                                      config["purge_observations"], config["window"])
        ranges["validation"] = (ranges["validation"][0], len(train_x) - max(config["horizons"]))
        for split in ["train", "validation"]:
            dataset = Windows(train_x, train_y, 10, *ranges[split], window=100, stride=1)
            print(split, "observations", ranges[split], "windows", len(dataset))
        del train_x, train_y, dataset
        """,
        ),
        cell(
            "markdown",
            """
        ## Freeze the GPU training recipe

        Default: full-window fitting/validation (stride 1), batch 128, up to 150 epochs,
        patience 15, and 90 minutes per model checked between epochs (up to six hours for four
        models, plus parsing and evaluation). Free Colab may interrupt earlier; Drive checkpoints
        allow continuation. No completion-time guarantee. Change settings **before** the first
        run; after opening the test, the pipeline rejects a different recipe/model set.

        This is a larger training budget than the optional bounded local CPU profile. It still
        uses decimal-precision-normalized files and different implementation/optimizer details
        from the original papers, so report it as our comparison rather than exact replication.
        """,
        ),
        cell(
            "code",
            """
        # Optional edits belong here, BEFORE any training/test evaluation.
        # config["batch_size"] = 64  # Use this before starting if GPU memory is limited.
        config_path = WORK / "configs/run.colab.json"
        config_path.write_text(json.dumps(config, indent=2))
        (DRIVE_ROOT / "run_config.json").write_text(json.dumps(config, indent=2))
        print(json.dumps(config, indent=2))

        def run_pipeline(*options):
            command = [sys.executable, "-u", "-m", "stock_predictor_lob.pipeline",
                       "--config", str(config_path), "--downloads", str(DOWNLOADS),
                       "--archive", str(ARCHIVE), "--output", str(RESULTS), "--no-upload", *options]
            RESULTS.mkdir(parents=True, exist_ok=True)
            with (RESULTS / "training.log").open("a", encoding="utf-8") as log:
                with subprocess.Popen(command, cwd=WORK, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end="")
                        log.write(line)
                        log.flush()
                    code = process.wait()
                    if code:
                        raise RuntimeError(f"Pipeline exit code {code}; see the log above.")
        """,
        ),
        cell(
            "code",
            """
        # Train/resume all four models; this step does not open test files.
        run_pipeline("--train-only")
        """,
        ),
        cell(
            "code",
            """
        import pandas as pd
        ready_path = RESULTS / "training_ready.json"
        if ready_path.exists():
            ready = json.loads(ready_path.read_text())
            display(pd.DataFrame.from_dict(ready["models"], orient="index"))
            print("Test already opened for this study:", ready["test_opened"])
        else:
            print("A completed cached report already exists; view it below.")
        """,
        ),
        cell(
            "markdown",
            """
        ## Final test evaluation

        Run this after accepting the training recipe and validation selection. All four model
        hashes are frozen before any test member is parsed. Every eligible test window is scored
        at each horizon; file boundaries are respected and the last 100 endpoints per file are
        excluded. Internal instrument boundaries are absent from the public files and cannot be
        verified. This immutable holdout policy prevents automatic retuning on test results.
        """,
        ),
        cell(
            "code",
            """
        RUN_FINAL_TEST = True  # Set False to stop after training and keep the test closed.
        if RUN_FINAL_TEST:
            run_pipeline()
            run_pipeline("--reuse-completed")  # Check hashes and recompute prediction metrics.
        """,
        ),
        cell(
            "code",
            """
        report_path = RESULTS / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            rows = []
            for identifier, model in report["models"].items():
                rows.append({"model": identifier, "test_accuracy": model["test"]["accuracy"],
                             "balanced_accuracy": model["test"]["balanced_accuracy"],
                             "macro_f1": model["test"]["macro_f1"],
                             "baseline_accuracy": model["training_prior_baseline"]["accuracy"],
                             "epochs": model["epochs"], "best_epoch": model["best_epoch"]})
            display(pd.DataFrame(rows).set_index("model"))
            print("Models actually above 80% accuracy:", report["above_80_percent_accuracy"])
            print("Full report and per-file confusion matrices:", report_path)
        else:
            print("No test report yet. Training checkpoints remain in Drive.")
        """,
        ),
        cell(
            "markdown",
            """
        ## Export results

        The export includes frozen weights, training histories, predictions, reports, config,
        hashes and holdout policy. Large source data and optimizer-resume checkpoints remain in
        Drive and are excluded from this compact export. To continue after disconnection, reopen
        this same notebook, reconnect a GPU and use the same Drive folder/code/configuration.
        Colab/PyTorch version changes may prevent exact checkpoint continuation.
        """,
        ),
        cell(
            "code",
            """
        export_path = DRIVE_ROOT / "StockPredictor_FI2010_results.zip"
        with zipfile.ZipFile(export_path, "w", compression=zipfile.ZIP_DEFLATED) as exported:
            for file in sorted(ARCHIVE.rglob("*")):
                if file.is_file() and file.suffix != ".zip":
                    exported.write(file, "archive/" + file.relative_to(ARCHIVE).as_posix())
            for file in sorted(RESULTS.glob("*.json")):
                exported.write(file, "results/" + file.name)
            for name in ["run_config.json", "environment.json"]:
                file = DRIVE_ROOT / name
                if file.exists():
                    exported.write(file, name)
        print("Export saved:", export_path)
        print("Download this ZIP from Google Drive when the run is complete.")
        """,
        ),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": NOTEBOOK.name},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    }


def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists() or sha(SOURCE.read_bytes()) != EXPECTED:
        raise ValueError("Complete pinned FI-2010 dataset is required; download it first")
    files = source_files()
    bundle = zip_bytes(files)
    notebook = build_notebook(bundle)
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    (OUTPUT / NOTEBOOK.name).write_bytes(NOTEBOOK.read_bytes())
    dataset = OUTPUT / "fi2010_complete.zip"
    if not dataset.exists() or sha(dataset.read_bytes()) != EXPECTED:
        shutil.copyfile(SOURCE, dataset)
    members = []
    with zipfile.ZipFile(SOURCE) as archive:
        for member in archive.infolist():
            if member.filename.endswith(".txt"):
                with archive.open(member) as stream:
                    observations = len(stream.readline().split())
                    row_count = 1 + sum(1 for _ in stream)
                members.append(
                    {
                        "file": Path(member.filename).name,
                        "observations": observations,
                        "source_rows": row_count,
                        "uncompressed_bytes": member.file_size,
                    }
                )
    manifest = {
        "source_sha256": EXPECTED,
        "source_bytes": dataset.stat().st_size,
        "source_bundle_sha256": sha(bundle),
        "notebook_sha256": sha(NOTEBOOK.read_bytes()),
        "files": members,
        "total_observations": sum(m["observations"] for m in members),
        "feature_rows": 40,
        "native_label_rows": 5,
        "horizons": [10, 20, 30, 50, 100],
        "native_labels": {"1": "UP", "2": "FLAT", "3": "DOWN"},
        "normalization": "provided decimal precision",
        "credentials_included": False,
        "training_executed": False,
    }
    (OUTPUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (ROOT / "notebooks/fi2010.dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    complete = OUTPUT / "StockPredictor_Colab_Complete.zip"
    with zipfile.ZipFile(complete, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(dataset, "data/fi2010_complete.zip")
        archive.write(NOTEBOOK, NOTEBOOK.name)
        archive.write(OUTPUT / "dataset_manifest.json", "dataset_manifest.json")
        for name, body in files.items():
            archive.writestr(name, body)
        for name in ["COLAB_TRAINING.md", "LOB_BENCHMARK.md"]:
            if (ROOT / name).exists():
                archive.write(ROOT / name, name)
    print(
        json.dumps(
            {
                "notebook": str(NOTEBOOK),
                "dataset": str(dataset),
                "complete_package": str(complete),
                "package_bytes": complete.stat().st_size,
                "manifest": manifest,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    build()
