
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        default="outputs"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    records = []

    for model_name in ["rnn", "lstm"]:
        metrics_file = (
            results_dir / model_name / "metrics.json"
        )

        if not metrics_file.exists():
            raise FileNotFoundError(metrics_file)

        with open(metrics_file) as f:
            metrics = json.load(f)

        records.append(metrics)

    comparison = pd.DataFrame(records)

    comparison.to_csv(
        results_dir / "model_comparison.csv",
        index=False
    )

    print("\nModel Comparison")
    print(comparison.to_string(index=False))

    metrics_to_plot = [
        ("test_accuracy", "Test Accuracy"),
        ("test_f1", "Test F1-score")
    ]

    for metric, title in metrics_to_plot:
        plt.figure(figsize=(7, 5))

        plt.bar(
            comparison["model_type"].str.upper(),
            comparison[metric]
        )

        plt.title(title)
        plt.ylabel(title)
        plt.ylim(0, 1)
        plt.tight_layout()

        plt.savefig(
            results_dir / f"{metric}_comparison.png"
        )

        plt.close()

    plt.figure(figsize=(7, 5))

    plt.bar(
        comparison["model_type"].str.upper(),
        comparison["training_seconds"]
    )

    plt.title("Training Time Comparison")
    plt.ylabel("Seconds")
    plt.tight_layout()

    plt.savefig(
        results_dir / "training_time_comparison.png"
    )

    plt.close()

    print(f"\nComparison saved to {results_dir}")


if __name__ == "__main__":
    main()
