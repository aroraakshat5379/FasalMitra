"""PlantVillage discovery, deterministic splitting, and TensorFlow input."""

import argparse
import json
import logging
import os
import random
from pathlib import Path

from fasalmitra.config import (
    BATCH_SIZE,
    CLIENT_CROPS,
    IMAGE_SIZE,
    MAX_IMAGES_PER_CLASS,
    NONIID_OVERLAP,
    SEED,
    TEST_FRACTION,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _crop_from_class(class_name):
    if "___" in class_name:
        crop = class_name.split("___", 1)[0]
    elif "__" in class_name:
        crop = class_name.split("__", 1)[0]
    else:
        crop = class_name.split("_", 1)[0]
    return crop


def _client_for_class(class_name):
    crop = _crop_from_class(class_name).lower()
    matches = [
        client
        for client, crops in CLIENT_CROPS.items()
        if any(crop.startswith(prefix) for prefix in crops)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Crop {!r} from class {!r} must belong to exactly one configured client. "
            "Update CLIENT_CROPS in fasalmitra/config.py.".format(crop, class_name)
        )
    return matches[0]


def _scan_classes(data_dir):
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError("PlantVillage directory does not exist: {}".format(data_dir))

    classes = {}
    split_dirs = [
        data_dir / split_name
        for split_name in ("train", "test", "val", "valid", "validation")
        if (data_dir / split_name).is_dir()
    ]
    source_dirs = split_dirs or [data_dir]
    for source_dir in source_dirs:
        for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
            with os.scandir(class_dir) as entries:
                files = sorted(
                    (class_dir / entry.name).as_posix()
                    for entry in entries
                    if entry.is_file(follow_symlinks=True)
                    and Path(entry.name).suffix.lower() in IMAGE_SUFFIXES
                )
            if files:
                classes.setdefault(class_dir.name, []).extend(files)
    for class_name in classes:
        classes[class_name].sort()
    if not classes:
        raise ValueError(
            "No class folders containing JPG or PNG images were found in {}".format(data_dir)
        )
    return data_dir, classes


def build_manifest(data_dir, max_per_class=MAX_IMAGES_PER_CLASS,
                   test_fraction=TEST_FRACTION, seed=SEED,
                   overlap_fraction=NONIID_OVERLAP):
    """Create a stratified test set and crop-skewed client training partitions."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be in the range [0, 1)")
    data_dir, class_files = _scan_classes(data_dir)
    class_names = sorted(class_files)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    rng = random.Random(seed)
    train_records = []
    test_records = []

    for class_name in class_names:
        files = list(class_files[class_name])
        rng.shuffle(files)
        if max_per_class is not None:
            if max_per_class < 2:
                raise ValueError("max_per_class must be at least 2, or None for all images")
            files = files[:max_per_class]
        if len(files) < 2:
            raise ValueError(
                "Class {!r} needs at least two images for a train/test split".format(class_name)
            )
        test_count = min(len(files) - 1, max(1, round(len(files) * test_fraction)))
        label = class_to_index[class_name]
        test_records.extend(
            {"path": path, "label": label, "class_name": class_name}
            for path in files[:test_count]
        )
        train_records.extend(
            {"path": path, "label": label, "class_name": class_name}
            for path in files[test_count:]
        )

    clients = {client: [] for client in CLIENT_CROPS}
    by_class = {class_name: [] for class_name in class_names}
    for record in train_records:
        by_class[record["class_name"]].append(record)
    for class_name, records in by_class.items():
        rng.shuffle(records)
        owner = _client_for_class(class_name)
        owner_count = int(round((1 - overlap_fraction) * len(records)))
        clients[owner].extend(records[:owner_count])
        peers = [client for client in clients if client != owner]
        for record in records[owner_count:]:
            clients[rng.choice(peers)].append(record)

    empty_clients = [client for client, records in clients.items() if not records]
    if empty_clients:
        raise ValueError(
            "No training images assigned to {}. Check CLIENT_CROPS against the dataset.".format(
                ", ".join(empty_clients)
            )
        )
    return {
        "data_dir": str(data_dir),
        "seed": seed,
        "max_per_class": max_per_class,
        "test_fraction": test_fraction,
        "overlap_fraction": overlap_fraction,
        "client_crops": {
            client: list(crops) for client, crops in CLIENT_CROPS.items()
        },
        "class_names": class_names,
        "class_to_index": class_to_index,
        "train": train_records,
        "test": test_records,
        "clients": clients,
    }


def load_or_create_manifest(data_dir, manifest_path, max_per_class=MAX_IMAGES_PER_CLASS,
                            test_fraction=TEST_FRACTION, seed=SEED, rebuild=False,
                            overlap_fraction=NONIID_OVERLAP):
    manifest_path = Path(manifest_path)
    if manifest_path.exists() and not rebuild:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        expected = {
            "data_dir": str(Path(data_dir).expanduser().resolve()),
            "seed": seed,
            "max_per_class": max_per_class,
            "test_fraction": test_fraction,
            "overlap_fraction": overlap_fraction,
            "client_crops": {
                client: list(crops) for client, crops in CLIENT_CROPS.items()
            },
        }
        actual = {key: manifest.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                "Cached manifest settings differ from this run. Use --rebuild-data to regenerate it."
            )
        return manifest

    manifest = build_manifest(
        data_dir, max_per_class, test_fraction, seed, overlap_fraction
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    temporary_path.replace(manifest_path)
    return manifest


def make_dataset(records, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, training=False):
    """Build a deterministic tf.data dataset from manifest records."""
    import tensorflow as tf

    paths = [record["path"] for record in records]
    labels = [record["label"] for record in records]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, image_size)
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(
            buffer_size=max(len(records), 1), seed=SEED, reshuffle_each_iteration=True
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _main():
    parser = argparse.ArgumentParser(description="Create and cache PlantVillage data splits.")
    parser.add_argument("--data-dir", default="data/PlantVillage")
    parser.add_argument("--manifest", default="results/data_manifest.json")
    parser.add_argument("--max-per-class", type=int, default=MAX_IMAGES_PER_CLASS)
    parser.add_argument("--rebuild", "--rebuild-data", dest="rebuild", action="store_true")
    args = parser.parse_args()
    manifest = load_or_create_manifest(
        args.data_dir, args.manifest, args.max_per_class, rebuild=args.rebuild
    )
    manifest_path = Path(args.manifest)
    results_dir = manifest_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=results_dir / "data.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    _write_split_plots(manifest, results_dir)
    logging.info(
        "Prepared %s classes: %s train, %s test images",
        len(manifest["class_names"]),
        len(manifest["train"]),
        len(manifest["test"]),
    )
    print(
        "Cached {} classes: {} train, {} test images across {} clients at {}".format(
            len(manifest["class_names"]),
            len(manifest["train"]),
            len(manifest["test"]),
            len(manifest["clients"]),
            args.manifest,
        )
    )


def _write_split_plots(manifest, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    class_names = manifest["class_names"]
    total_counts = Counter(record["class_name"] for record in manifest["train"] + manifest["test"])
    ordered_names = sorted(class_names, key=lambda name: total_counts[name])
    figure, axis = plt.subplots(figsize=(10, max(7, len(class_names) * 0.25)))
    axis.barh(
        [name.replace("___", " | ").replace("_", " ") for name in ordered_names],
        [total_counts[name] for name in ordered_names],
    )
    axis.set(xlabel="Sampled images", title="PlantVillage sampled class distribution")
    figure.tight_layout()
    figure.savefig(Path(results_dir) / "class_distribution.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    crops = sorted({_crop_from_class(name) for name in class_names})
    bottom = [0] * len(manifest["clients"])
    client_names = list(manifest["clients"])
    for crop in crops:
        counts = [
            sum(
                _crop_from_class(record["class_name"]) == crop
                for record in manifest["clients"][client]
            )
            for client in client_names
        ]
        axis.bar(client_names, counts, bottom=bottom, label=crop)
        bottom = [value + count for value, count in zip(bottom, counts)]
    axis.set(ylabel="Training images", title="Crop-skewed client partitions")
    axis.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    figure.tight_layout()
    figure.savefig(Path(results_dir) / "client_split.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    _main()
