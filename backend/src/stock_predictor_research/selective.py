import numpy as np
import pandas as pd


def risk_coverage(labels, probabilities, index, config, baseline=None):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    results = []
    for threshold in config["confidence_thresholds"]:
        accepted = probabilities.max(axis=1) >= threshold
        count = int(accepted.sum())
        row = {
            "threshold": threshold,
            "accepted": count,
            "coverage": count / len(labels),
            "dates": int(index.loc[accepted, "session_date"].nunique()),
            "accuracy": float((probabilities[accepted].argmax(axis=1) == labels[accepted]).mean())
            if count
            else None,
            "lower_95_date_block_accuracy": None,
            "baseline_on_same_accepted_rows": float(
                (baseline[accepted].argmax(axis=1) == labels[accepted]).mean()
            )
            if count and baseline is not None
            else None,
        }
        if (
            count >= config["minimum_selective_examples"]
            and row["dates"] >= config["minimum_selective_dates"]
        ):
            aggregate = (
                pd.DataFrame(
                    {
                        "date": index.session_date,
                        "accepted": accepted.astype(int),
                        "correct": (accepted & (probabilities.argmax(axis=1) == labels)).astype(
                            int
                        ),
                    }
                )
                .groupby("date", sort=True)
                .sum()
                .to_numpy()
            )
            length = min(config["bootstrap_block_sessions"], len(aggregate))
            rng = np.random.default_rng(config["seed"])
            scores = []
            for _ in range(config["bootstrap_repetitions"]):
                starts = rng.integers(
                    0, len(aggregate) - length + 1, size=int(np.ceil(len(aggregate) / length))
                )
                positions = np.concatenate([np.arange(s, s + length) for s in starts])[
                    : len(aggregate)
                ]
                sample = aggregate[positions].sum(axis=0)
                scores.append(sample[1] / sample[0] if sample[0] else 0.0)
            row["lower_95_date_block_accuracy"] = float(np.quantile(scores, 0.05))
        results.append(row)
    return results


def select_rule(curve, target):
    qualified = [
        row
        for row in curve
        if row["lower_95_date_block_accuracy"] is not None
        and row["lower_95_date_block_accuracy"] >= target
    ]
    if not qualified:
        return {
            "status": "target_not_supported",
            "threshold": None,
            "enabled": False,
            "target_accuracy": target,
            "guarantee": False,
            "fresh_holdout_required": True,
        }
    chosen = max(qualified, key=lambda row: row["coverage"])
    return {
        "status": "research_rule_selected",
        "threshold": chosen["threshold"],
        "enabled": False,
        "target_accuracy": target,
        "guarantee": False,
        "selection_diagnostics": chosen,
        "fresh_holdout_required": True,
    }
