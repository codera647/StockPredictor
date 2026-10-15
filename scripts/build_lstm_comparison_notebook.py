"""Package two clearly separated LSTM checks without touching the trained DeepLOB study."""

import base64
import json
from pathlib import Path

from build_colab_bundle import cell, sha, zip_bytes

ROOT = Path(__file__).resolve().parents[1]


def files():
    names = [
        "backend/src/stock_predictor/__init__.py",
        "backend/src/stock_predictor/config.py",
        "backend/src/stock_predictor/storage/__init__.py",
        "backend/src/stock_predictor/storage/archive.py",
        "backend/src/stock_predictor/training/__init__.py",
        "backend/src/stock_predictor/training/metrics.py",
        "backend/src/stock_predictor_lob/__init__.py",
        "backend/src/stock_predictor_lob/data.py",
        "backend/src/stock_predictor_lob/models.py",
        "backend/src/stock_predictor_lob/training.py",
        "backend/tests/test_lstm_compare.py",
        "configs/lstm-price-check.json",
        "configs/lstm-comparison.colab.json",
        "configs/lstm-comparison.baseline.json",
        "notebooks/requirements-lstm-compare.txt",
    ]
    names += [
        p.relative_to(ROOT).as_posix()
        for p in sorted((ROOT / "backend/src/stock_predictor_lstm_compare").glob("*.py"))
    ]
    entries = {name: (ROOT / name).read_bytes() for name in names}
    entries["BUNDLE_MANIFEST.json"] = json.dumps(
        {name: sha(body) for name, body in entries.items()},
        sort_keys=True,
        indent=2,
    ).encode()
    return entries


def notebook(bundle):
    encoded = base64.b64encode(bundle).decode()
    cells = [
        cell(
            "markdown",
            """
        # Plain LSTM versus DeepLOB: matched direction classification and a separate price check

        This notebook answers two different questions without mixing their scores:

        1. **Price experiment:** a modern PyTorch port inspired by the repository's 1-layer,
           128-unit price LSTM, trained on its public GOOG-year.csv with train-only scaling.
           Report the original price-error score, RMSE and a last-price baseline.
        2. **Comparable experiment:** adapt that plain LSTM architecture to 40 order-book inputs,
           100-observation windows and the same DOWN/FLAT/UP labels at horizon 10 as DeepLOB.
           Evaluate exactly the same 138,990 test rows against DeepLOB's verified **82.94%**.

        A 95% price-error score is not 95% correct directions. The price model and the adapted
        classifier are separate models. This is not a bit-for-bit TensorFlow notebook reproduction.
        Reference: https://github.com/huseinzol05/Stock-Prediction-Models/blob/master/deep-learning/1.lstm.ipynb

        **The FI-2010 test is already exposed.** This is a retrospective benchmark comparison,
        not a fresh blind test or evidence of live daily forecasting performance. A single frozen
        configuration selects its checkpoint on validation log loss; do not tune on test results.
        Keep the original DeepLOB notebook/Drive directory intact. This notebook uses a separate
        directory and does not load, retrain or overwrite its checkpoints.
        """,
        ),
        cell(
            "markdown",
            """
        ## 1. Setup and persistent Drive checkpoints

        Select **Runtime → Change runtime type → GPU**. No `.env`, paid API or manual dataset
        upload is needed. The notebook downloads the pinned public datasets. Drive stores epoch
        checkpoints and outputs; rerun the same code/configuration to resume an interrupted epoch
        sequence. GPU availability and runtime limits depend on Colab's free allocation.
        """,
        ),
        cell(
            "code",
            """
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import sys, io, json, base64, hashlib, zipfile, subprocess
        from pathlib import Path
        from google.colab import drive
        drive.mount("/content/drive")
        WORK = Path("/content/StockPredictor_LSTM_Comparison_code")
        OUTPUT = Path("/content/drive/MyDrive/StockPredictor_LSTM_Comparison/results")
        DOWNLOADS = Path("/content/StockPredictor_LSTM_Comparison_data")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        print("New study folder:", OUTPUT.parent)
        """,
        ),
        cell(
            "code",
            f"""
        SOURCE_BUNDLE = "{encoded}"
        EXPECTED_SHA256 = "{sha(bundle)}"
        body = base64.b64decode(SOURCE_BUNDLE)
        assert hashlib.sha256(body).hexdigest() == EXPECTED_SHA256
        with zipfile.ZipFile(io.BytesIO(body)) as source:
            for member in source.infolist():
                target = (WORK / member.filename).resolve()
                if not target.is_relative_to(WORK.resolve()):
                    raise ValueError("Unsafe source path")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(member))
        manifest = json.loads((WORK / "BUNDLE_MANIFEST.json").read_text())
        for name, expected in manifest.items():
            assert hashlib.sha256((WORK / name).read_bytes()).hexdigest() == expected
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                        str(WORK / "notebooks/requirements-lstm-compare.txt")], check=True)
        sys.path.insert(0, str(WORK / "backend/src"))
        import torch
        assert torch.cuda.is_available(), "Select a GPU runtime for the FI-2010 classifier"
        print("GPU:", torch.cuda.get_device_name(0), "Torch:", torch.__version__)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(WORK / "backend/src")
        subprocess.run([sys.executable, "-m", "pytest", "-q", "backend/tests/test_lstm_compare.py"],
                       cwd=WORK, env=env, check=True)
        """,
        ),
        cell(
            "code",
            """
        def run_experiment(mode, *options):
            command = [sys.executable, "-u", "-m", "stock_predictor_lstm_compare.pipeline", mode,
                       "--downloads", str(DOWNLOADS), "--output", str(OUTPUT), *options]
            with (OUTPUT / "run.log").open("a", encoding="utf-8") as log:
                with subprocess.Popen(command, cwd=WORK, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end="")
                        log.write(line)
                        log.flush()
                    status = process.wait()
                    if status:
                        raise RuntimeError(f"Experiment exit code {status}; see log above")
        """,
        ),
        cell(
            "markdown",
            """
        ## 2. Original-style price forecasting check

        Public GOOG daily data: 192 training days, 30 validation days, final 30 test days.
        Train-only min-max scaling; 128-unit stateful LSTM; 5-step truncated sequences;
        Adam 0.01; output dropout 0.2; up to 300 epochs with patience 30, selected on validation
        recursive-forecast RMSE. Inference uses a 30-day recursive forecast and no future prices.
        The test forecast origin includes the observed validation history, which is already past.
        PyTorch initialization, validation selection, inference dropout handling and omitted
        post-forecast smoothing differ from the source notebook. This small check runs on CPU.

        The repository-style price score is `100 × (1 − RMS((actual − forecast)/(actual + 1)))`.
        It can be high even for constant-price forecasts. Compare with the last-price baseline.
        The displayed daily-path sign agreement is also a different target from FI-2010 labels.
        """,
        ),
        cell(
            "code",
            """
        run_experiment("price")
        """,
        ),
        cell(
            "code",
            """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        price_report = json.loads((OUTPUT / "price_report.json").read_text())
        display(pd.DataFrame({"LSTM": price_report["lstm"],
                              "Last-price baseline": price_report["persistence"]}).loc[
            ["repository_price_score_percent", "rmse", "mae", "daily_path_direction_accuracy"]])
        with np.load(OUTPUT / "price" / price_report["recipe"] / "predictions.npz",
                     allow_pickle=False) as predictions:
            fig, ax = plt.subplots(figsize=(10, 4))
            for key, label in [("actual", "Actual"), ("predicted", "LSTM"),
                               ("persistence", "Last-price baseline")]:
                ax.plot(predictions["dates"].astype("datetime64[D]"), predictions[key], label=label)
            ax.set(title="GOOG: 30-day recursive forecast", ylabel="Close price (USD)")
            ax.legend()
            ax.grid(alpha=0.2)
            fig.autofmt_xdate()
            plt.show()
        print("Price-error scores do not establish a win over DeepLOB classification accuracy.")
        """,
        ),
        cell(
            "markdown",
            """
        ## 3. Matched FI-2010 classification: train and select the LSTM

        This model changes the source forecaster's task: 40 inputs, 100-step history, 3-class
        logits and cross-entropy. It preserves a plain 128-unit, one-layer LSTM with 0.2 output
        dropout. Adam 0.001 matches the DeepLOB comparison's learning rate; seed 20261025.
        Train and validation use every eligible window with exactly the previous purge/file
        boundaries: **203,601 training / 50,751 validation**. Provided decimal normalization.
        Up to 150 epochs, patience 15 and 90 minutes checked between epochs. Model/optimizer and
        CPU/CUDA RNG checkpoints persist after every epoch. Reconnect with the same notebook,
        versions, configuration and Drive directory to resume. Price-check training is small
        and restarts if interrupted; the large classifier resumes committed epochs.
        """,
        ),
        cell(
            "code",
            """
        run_experiment("classify")  # Fits/selects the classifier; test evaluation stays closed.
        """,
        ),
        cell(
            "code",
            """
        ready_path = OUTPUT / "classification_training_ready.json"
        if ready_path.exists():
            ready = json.loads(ready_path.read_text())
            print("Selected epoch:", ready["best_epoch"], "of", ready["epochs"])
            display(pd.Series({k: ready["validation"][k]
                               for k in ["accuracy", "balanced_accuracy", "macro_f1", "log_loss"]}))
        """,
        ),
        cell(
            "markdown",
            """
        ## 4. Evaluate frozen weights on the exact DeepLOB horizon-10 test rows

        The first evaluation freezes the selected weight hash and recipe. It rejects later
        changes, scores all 138,990 rows, checks their file IDs/endpoints/native labels against
        the verified DeepLOB row checksum, saves predictions and verifies reload probes.
        This previously exposed test is a retrospective comparison. There is no guarantee of
        beating 82.94%, and a higher price-error score alone will not count as an improvement.
        """,
        ),
        cell(
            "code",
            """
        RUN_FINAL_COMPARISON = True
        if RUN_FINAL_COMPARISON:
            run_experiment("classify", "--evaluate")
        """,
        ),
        cell(
            "code",
            """
        path = OUTPUT / "classification_report.json"
        if path.exists():
            comparison = json.loads(path.read_text())
            keys = ["accuracy", "balanced_accuracy", "macro_f1"]
            table = pd.DataFrame({"DeepLOB horizon 10": {k: comparison["deeplob_baseline"][k]
                                                       for k in keys},
                                  "Plain LSTM h10": {k: comparison["test"][k] for k in keys}})
            display((100 * table).round(2))
            print("LSTM exceeds DeepLOB accuracy:", comparison["beats_deeplob_accuracy"])
            print("Accuracy difference (percentage points):",
                  comparison["accuracy_difference_percentage_points"])
            ax = (100 * table).plot.bar(figsize=(9, 4), ylim=(0, 100), rot=0)
            ax.set(ylabel="Score (%)", title="Same test rows: classification comparison")
            ax.grid(axis="y", alpha=0.2)
            plt.show()
        else:
            print("Final comparison has not run yet.")
        """,
        ),
        cell(
            "markdown",
            """
        ## 5. Export model weights, checkpoints, predictions and reports

        Download this ZIP from Drive after training and share its local extracted folder for
        verification. Original DeepLOB checkpoints and its Drive folder are untouched. New
        results stay research-only; no production model is automatically replaced.
        """,
        ),
        cell(
            "code",
            """
        exported = OUTPUT.parent / "StockPredictor_LSTM_Comparison_results.zip"
        with zipfile.ZipFile(exported, "w", compression=zipfile.ZIP_DEFLATED) as package:
            for file in sorted(OUTPUT.rglob("*")):
                if file.is_file() and file.suffix in [".json", ".npz", ".pt"]:
                    package.write(file, "results/" + file.relative_to(OUTPUT).as_posix())
        print("Results saved:", exported)
        """,
        ),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "StockPredictor_LSTM_Comparison.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
    }


def main():
    entries = files()
    source = zip_bytes(entries)
    result = notebook(source)
    path = ROOT / "notebooks/StockPredictor_LSTM_Comparison.ipynb"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    output = ROOT / "artifacts/lstm-comparison/colab"
    output.mkdir(parents=True, exist_ok=True)
    (output / path.name).write_bytes(path.read_bytes())
    (output / "source.zip").write_bytes(source)
    manifest = {
        "notebook_sha256": sha(path.read_bytes()),
        "source_sha256": sha(source),
        "embedded_files": len(entries) - 1,
        "code_cells_compiled": sum(c["cell_type"] == "code" for c in result["cells"]),
        "gpu_training_executed": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
