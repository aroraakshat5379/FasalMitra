"""Run a reproducible five-client FedAvg simulation."""

import argparse
import csv
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf

from fasalmitra.config import (
    BATCH_SIZE,
    FED_ROUNDS,
    IMAGE_SIZE,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    MAX_IMAGES_PER_CLASS,
    SEED,
)
from fasalmitra.data import load_or_create_manifest, make_dataset
from fasalmitra.evaluate import evaluate_model, update_comparison_summary
from fasalmitra.model import build_model


def _main():
    parser = argparse.ArgumentParser(description="Train the five-client FedAvg simulation.")
    parser.add_argument("--data-dir", default="data/PlantVillage")
    parser.add_argument("--manifest", default="results/data_manifest.json")
    parser.add_argument("--results-dir", default="results/fedavg")
    parser.add_argument("--max-per-class", type=int, default=MAX_IMAGES_PER_CLASS)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--rounds", type=int, default=FED_ROUNDS)
    parser.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--rebuild-data", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.local_epochs < 1:
        parser.error("--rounds and --local-epochs must be positive")
    tf.keras.utils.set_random_seed(SEED)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=results_dir / "training.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    manifest = load_or_create_manifest(
        args.data_dir,
        args.manifest,
        args.max_per_class,
        seed=SEED,
        rebuild=args.rebuild_data,
    )
    image_size = (args.image_size, args.image_size)
    global_model = build_model(
        len(manifest["class_names"]),
        image_size=image_size,
        weights=args.weights,
        learning_rate=args.learning_rate,
    )
    local_datasets = {
        client: make_dataset(records, args.batch_size, image_size, training=True)
        for client, records in manifest["clients"].items()
    }
    metrics_path = results_dir / "round_metrics.csv"
    started_at = perf_counter()
    logging.info(
        "Starting FedAvg: %s rounds, %s local epochs, %s clients",
        args.rounds,
        args.local_epochs,
        len(manifest["clients"]),
    )
    client_model = build_model(
        len(manifest["class_names"]),
        image_size=image_size,
        weights="none",
        learning_rate=args.learning_rate,
    )
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=("round", "loss", "accuracy", "macro_f1", "weighted_f1", "samples"),
        )
        writer.writeheader()
        for round_number in range(1, args.rounds + 1):
            global_weights = global_model.get_weights()
            client_weights = []
            client_sizes = []
            for client, records in manifest["clients"].items():
                client_model.set_weights(global_weights)
                client_model.compile(
                    optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )
                client_model.fit(
                    local_datasets[client],
                    epochs=args.local_epochs,
                    verbose=0,
                    shuffle=False,
                )
                client_weights.append(client_model.get_weights())
                client_sizes.append(len(records))

            total_examples = sum(client_sizes)
            aggregated_weights = [
                np.average(
                    np.stack([weights[layer] for weights in client_weights], axis=0),
                    axis=0,
                    weights=client_sizes,
                )
                for layer in range(len(global_weights))
            ]
            global_model.set_weights(aggregated_weights)
            metrics = evaluate_model(
                global_model,
                manifest["test"],
                manifest["class_names"],
                results_dir,
                prefix="round_{:02d}".format(round_number),
                batch_size=args.batch_size,
                save_details=(round_number == args.rounds),
            )
            metrics["round"] = round_number
            writer.writerow(metrics)
            metrics_file.flush()
            logging.info("Round %s metrics: %s", round_number, metrics)
            print(
                "Round {}/{}: loss={:.4f}, accuracy={:.4f}, macro_f1={:.4f} "
                "({} client-training examples)".format(
                    round_number,
                    args.rounds,
                    metrics["loss"],
                    metrics["accuracy"],
                    metrics["macro_f1"],
                    total_examples,
                )
            )

    train_seconds = perf_counter() - started_at
    global_model.optimizer.build(global_model.trainable_variables)
    global_model.save(results_dir / "fedavg_model.keras")
    summary_path = update_comparison_summary(
        results_dir.parent,
        "FedAvg ({} non-IID clients)".format(len(manifest["clients"])),
        metrics,
        train_seconds,
        client_count=len(manifest["clients"]),
    )
    _write_convergence_plot(results_dir, results_dir.parent, metrics_path)
    logging.info("FedAvg training complete: %s", metrics)
    logging.info("Comparison summary saved to %s", summary_path)
    print("Per-round results written to {}".format(metrics_path))


def _write_convergence_plot(results_dir, results_root, metrics_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(metrics_path).open("r", newline="", encoding="utf-8") as file:
        rounds = list(csv.DictReader(file))
    summary_path = Path(results_root) / "summary.csv"
    central_accuracy = None
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["Method"] == "Centralized (pooled data)":
                    central_accuracy = float(row["Accuracy"])
                    break

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        [int(row["round"]) for row in rounds],
        [float(row["accuracy"]) for row in rounds],
        marker="o",
        label="FedAvg global test accuracy",
    )
    if central_accuracy is not None:
        axis.axhline(
            central_accuracy,
            color="gray",
            linestyle="--",
            label="Centralized baseline ({:.2f})".format(central_accuracy),
        )
    axis.set(
        xlabel="Federated round",
        ylabel="Accuracy",
        ylim=(0, 1.02),
        title="FedAvg convergence",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(Path(results_dir) / "fedavg_convergence.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    _main()
