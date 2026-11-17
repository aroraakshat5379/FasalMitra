import unittest
from pathlib import Path

from fasalmitra.predict import (
    LOW_CONFIDENCE_MESSAGE,
    display_label,
    interpret_probabilities,
    load_advisories,
)


class PredictionInterpretationTests(unittest.TestCase):
    def setUp(self):
        self.classes = ["Potato___Late_blight", "Tomato___healthy", "Tomato___Early_blight"]
        self.advisories = {"Potato___Late_blight": {"summary": "Seed advice"}}

    def test_accepts_high_confidence_prediction_and_attaches_advice(self):
        result = interpret_probabilities(
            [0.9, 0.05, 0.05], self.classes, threshold=0.65, advisories=self.advisories
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["prediction"]["class_name"], "Potato___Late_blight")
        self.assertEqual(result["advisory"]["summary"], "Seed advice")
        self.assertEqual(len(result["top_predictions"]), 3)

    def test_rejects_low_confidence_prediction_without_advice(self):
        result = interpret_probabilities(
            [0.4, 0.35, 0.25], self.classes, threshold=0.65, advisories=self.advisories
        )
        self.assertFalse(result["accepted"])
        self.assertIsNone(result["prediction"])
        self.assertIsNone(result["advisory"])
        self.assertEqual(result["message"], LOW_CONFIDENCE_MESSAGE)

    def test_validates_probability_shape_and_threshold(self):
        with self.assertRaisesRegex(ValueError, "match"):
            interpret_probabilities([1.0], self.classes, threshold=0.5)
        with self.assertRaisesRegex(ValueError, "threshold"):
            interpret_probabilities([0.9, 0.05, 0.05], self.classes, threshold=1.1)

    def test_display_label_is_readable(self):
        self.assertEqual(display_label("Tomato___Early_blight"), "Tomato — Early blight")

    def test_seed_advisory_has_six_sourced_entries(self):
        root = Path(__file__).resolve().parents[1]
        advisories = load_advisories(root / "fasalmitra" / "advisory.json")
        self.assertEqual(len(advisories), 6)
        for advisory in advisories.values():
            self.assertTrue(advisory["source"]["url"].startswith("https://"))
            self.assertTrue(advisory["immediate_actions"])
            self.assertTrue(advisory["prevention"])


if __name__ == "__main__":
    unittest.main()
