import tempfile
import unittest
from pathlib import Path

from fasalmitra.data import (
    _client_for_class,
    _crop_from_class,
    build_manifest,
    load_or_create_manifest,
)


class ManifestTests(unittest.TestCase):
    def _dataset(self, root):
        for class_name in (
            "Apple___Apple_scab",
            "Tomato___healthy",
            "Potato___Late_blight",
            "Grape___Black_rot",
            "Corn___Common_rust",
            "Raspberry___healthy",
        ):
            class_dir = root / class_name
            class_dir.mkdir()
            for index in range(5):
                (class_dir / "image_{}.jpg".format(index)).write_bytes(b"test")

    def test_split_is_deterministic_disjoint_and_crop_grouped(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "dataset"
            data_dir.mkdir()
            self._dataset(data_dir)

            first = build_manifest(data_dir, max_per_class=5, seed=42)
            second = build_manifest(data_dir, max_per_class=5, seed=42)
            self.assertEqual(first, second)
            train_paths = {record["path"] for record in first["train"]}
            test_paths = {record["path"] for record in first["test"]}
            self.assertFalse(train_paths & test_paths)
            self.assertEqual(len(train_paths | test_paths), 30)
            client_paths = {
                record["path"]
                for records in first["clients"].values()
                for record in records
            }
            self.assertEqual(client_paths, train_paths)
            self.assertEqual(
                sum(
                    record["class_name"] == "Tomato___healthy"
                    for record in first["clients"]["client_1"]
                ),
                3,
            )
            self.assertEqual(
                sum(
                    record["class_name"] == "Tomato___healthy"
                    for records in first["clients"].values()
                    for record in records
                ),
                4,
            )
            self.assertIn(
                "Tomato___healthy",
                {record["class_name"] for record in first["clients"]["client_1"]},
            )
            self.assertIn(
                "Apple___Apple_scab",
                {record["class_name"] for record in first["clients"]["client_2"]},
            )
            self.assertIn(
                "Corn___Common_rust",
                {record["class_name"] for record in first["clients"]["client_3"]},
            )

    def test_plantvillage_crop_names_are_parsed(self):
        self.assertEqual(
            _crop_from_class("Cherry_(including_sour)___Powdery_mildew"),
            "Cherry_(including_sour)",
        )
        self.assertEqual(_crop_from_class("Corn_(maize)___Common_rust"), "Corn_(maize)")
        self.assertEqual(_crop_from_class("Pepper,_bell___Bacterial_spot"), "Pepper,_bell")
        self.assertEqual(_client_for_class("Pepper,_bell___Bacterial_spot"), "client_4")

    def test_unknown_crop_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one configured client"):
            _client_for_class("Unknown_crop___Unknown_disease")

    def test_kaggle_train_test_folders_are_combined_before_global_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            class_names = (
                "Apple___Apple_scab",
                "Tomato___healthy",
                "Potato___Late_blight",
                "Grape___Black_rot",
                "Raspberry___healthy",
            )
            for class_name in class_names:
                for split in ("train", "val"):
                    class_dir = data_dir / split / class_name
                    class_dir.mkdir(parents=True)
                    (class_dir / "{}.jpg".format(split)).write_bytes(b"test")

            manifest = build_manifest(data_dir, max_per_class=2, seed=42)
            self.assertEqual(len(manifest["class_names"]), 5)
            self.assertEqual(len(manifest["train"]) + len(manifest["test"]), 10)
            self.assertEqual(len(manifest["test"]), 5)
            self.assertEqual(
                {
                    Path(record["path"]).parent.parent.name
                    for record in manifest["train"] + manifest["test"]
                },
                {"train", "val"},
            )

    def test_manifest_cache_rejects_mismatched_sampling_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "dataset"
            data_dir.mkdir()
            self._dataset(data_dir)
            manifest_path = root / "manifest.json"
            saved = load_or_create_manifest(data_dir, manifest_path, max_per_class=4)
            self.assertEqual(
                saved,
                load_or_create_manifest(data_dir, manifest_path, max_per_class=4),
            )
            with self.assertRaisesRegex(ValueError, "settings differ"):
                load_or_create_manifest(data_dir, manifest_path, max_per_class=3)
            with self.assertRaisesRegex(ValueError, "settings differ"):
                load_or_create_manifest(
                    data_dir, manifest_path, max_per_class=4, overlap_fraction=0.2
                )


if __name__ == "__main__":
    unittest.main()
