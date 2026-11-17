"""Evaluation metrics and persisted reports for trained classifiers."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from fasalmitra.config import BATCH_SIZE, MAX_IMAGES_PER_CLASS
from fasalmitra.data import load_or_create_manifest, make_dataset


def evaluate_model(model, records, class_names, output_dir, prefix="evaluation",
                   batch_size=BATCH_SIZE, save_details=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset(records, batch_size=batch_size, training=False)
    probabilities = model.predict(dataset, verbose=0)
    y_true = np.asarray([record["label"] for record in records], dtype=np.int64)
    y_pred = np.argmax(probabilities, axis=1)
    labels = list(range(len(class_names)))
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    metrics = {
        "loss": float(
            -np.log(
                np.clip(
                    probabilities[np.arange(len(y_true)), y_true],
                    np.finfo(np.float32).eps,
                    1.0,
                )
            ).mean()
        ),
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "samples": int(len(y_true)),
    }
    if save_details:
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        with (output_dir / "{}_metrics.json".format(prefix)).open(
            "w", encoding="utf-8"
        ) as file:
            json.dump({"summary": metrics, "classification_report": report}, file, indent=2)
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(max(10, len(class_names) * 0.35),) * 2)
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set(
            xticks=labels,
            yticks=labels,
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted label",
            ylabel="True label",
            title="{} confusion matrix".format(prefix.replace("_", " ").title()),
        )
        plt.setp(axis.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
        figure.tight_layout()
        figure.savefig(output_dir / "{}_confusion_matrix.png".format(prefix), dpi=150)
        plt.close(figure)
    return metrics


def update_comparison_summary(results_root, method, metrics, elapsed_seconds, client_count):
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    summary_path = results_root / "summary.csv"
    rows = []
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    rows = [row for row in rows if row["Method"] != method]
    rows.append(
        {
            "Method": method,
            "Accuracy": "{:.3f}".format(metrics["accuracy"]),
            "Macro-F1": "{:.3f}".format(metrics["macro_f1"]),
            "Weighted-F1": "{:.3f}".format(metrics["weighted_f1"]),
            "Train time (s)": str(round(elapsed_seconds)),
            "Clients": str(client_count),
        }
    )
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("Method", "Accuracy", "Macro-F1", "Weighted-F1", "Train time (s)", "Clients"),
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def _main():
    parser = argparse.ArgumentParser(description="Evaluate a saved FasalMitra Keras model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", default="data/PlantVillage")
    parser.add_argument("--manifest", default="results/data_manifest.json")
    parser.add_argument("--results-dir", default="results/evaluation")
    parser.add_argument("--max-per-class", type=int, default=MAX_IMAGES_PER_CLASS)
    args = parser.parse_args()
    manifest = load_or_create_manifest(
        args.data_dir, args.manifest, args.max_per_class
    )
    import tensorflow as tf

    model = tf.keras.models.load_model(args.model)
    metrics = evaluate_model(
        model, manifest["test"], manifest["class_names"], args.results_dir
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    _main()
