"""Train the centralized MobileNetV2 baseline."""

import argparse
import csv
import logging
from pathlib import Path
from time import perf_counter

import tensorflow as tf

from fasalmitra.config import (
    BATCH_SIZE,
    CENTRAL_EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    MAX_IMAGES_PER_CLASS,
    SEED,
)
from fasalmitra.data import load_or_create_manifest, make_dataset
from fasalmitra.evaluate import evaluate_model, update_comparison_summary
from fasalmitra.model import build_model


def _main():
    parser = argparse.ArgumentParser(description="Train the centralized PlantVillage baseline.")
    parser.add_argument("--data-dir", default="data/PlantVillage")
    parser.add_argument("--manifest", default="results/data_manifest.json")
    parser.add_argument("--results-dir", default="results/centralized")
    parser.add_argument("--max-per-class", type=int, default=MAX_IMAGES_PER_CLASS)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--epochs", type=int, default=CENTRAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--rebuild-data", action="store_true")
    args = parser.parse_args()
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
    model = build_model(
        len(manifest["class_names"]),
        image_size=image_size,
        weights=args.weights,
        learning_rate=args.learning_rate,
    )
    train_data = make_dataset(
        manifest["train"], args.batch_size, image_size, training=True
    )
    started_at = perf_counter()
    logging.info("Starting centralized training for %s epochs", args.epochs)
    history = model.fit(train_data, epochs=args.epochs, verbose=1, shuffle=False)
    train_seconds = perf_counter() - started_at
    history_path = results_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(("epoch", "loss", "accuracy"))
        for index, (loss, accuracy) in enumerate(
            zip(history.history["loss"], history.history["accuracy"]), start=1
        ):
            writer.writerow((index, loss, accuracy))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(range(1, len(history.history["accuracy"]) + 1), history.history["accuracy"])
    axis.set(xlabel="Epoch", ylabel="Training accuracy", title="Centralized training")
    figure.tight_layout()
    figure.savefig(results_dir / "training_curve.png", dpi=150)
    plt.close(figure)
    model.save(results_dir / "centralized_model.keras")
    metrics = evaluate_model(
        model,
        manifest["test"],
        manifest["class_names"],
        results_dir,
        prefix="centralized",
        batch_size=args.batch_size,
    )
    summary_path = update_comparison_summary(
        results_dir.parent,
        "Centralized (pooled data)",
        metrics,
        train_seconds,
        client_count=1,
    )
    logging.info("Centralized training complete: %s", metrics)
    logging.info("Comparison summary saved to %s", summary_path)
    print("Centralized test metrics: {}".format(metrics))


if __name__ == "__main__":
    _main()
