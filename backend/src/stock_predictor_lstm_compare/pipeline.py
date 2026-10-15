import argparse
import json
import logging
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Price-score check and matched FI-2010 plain LSTM")
    parser.add_argument("mode", choices=["price", "classify"])
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--baseline", type=Path, default=Path("configs/lstm-comparison.baseline.json")
    )
    parser.add_argument("--downloads", type=Path, default=Path("data/downloads"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/lstm-comparison"))
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config_path = args.config or Path(
        "configs/lstm-price-check.json"
        if args.mode == "price"
        else "configs/lstm-comparison.colab.json"
    )
    config = json.loads(config_path.read_text())
    if args.mode == "price":
        from stock_predictor_lstm_compare.price import run

        report = run(config, args.downloads / "husein_lstm", args.output)
        result = {
            k: report[k]
            for k in (
                "lstm",
                "persistence",
                "best_epoch",
                "epochs",
                "comparable_to_deeplob_classification_accuracy",
            )
        }
    else:
        from stock_predictor_lstm_compare.classification import run

        baseline = json.loads(args.baseline.read_text())
        report = run(config, baseline, args.downloads / "fi2010", args.output, args.evaluate)
        result = {
            k: v
            for k, v in report.items()
            if k
            in (
                "recipe",
                "test",
                "validation",
                "epochs",
                "best_epoch",
                "beats_deeplob_accuracy",
                "accuracy_difference_percentage_points",
                "test_opened",
            )
        }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
