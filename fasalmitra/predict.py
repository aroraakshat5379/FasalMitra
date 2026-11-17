"""Run one-image FasalMitra predictions and attach seed advisory content."""

import argparse
import json
import math
from pathlib import Path

from fasalmitra.config import IMAGE_SIZE


LOW_CONFIDENCE_MESSAGE = (
    "The model is not sufficiently confident in this photo. Please consult an "
    "agricultural extension expert and, if possible, submit a clearer image of "
    "a single affected leaf."
)


def display_label(class_name):
    """Turn a PlantVillage class identifier into a readable crop and condition."""
    return class_name.replace("___", " — ").replace("_", " ")


def load_advisories(advisory_path):
    """Load advisory entries keyed by their exact PlantVillage class name."""
    path = Path(advisory_path)
    with path.open(encoding="utf-8") as file:
        advisories = json.load(file)
    if not isinstance(advisories, dict):
        raise ValueError("Advisory JSON must contain an object keyed by class name.")
    return advisories


def interpret_probabilities(probabilities, class_names, threshold, advisories=None, top_k=3):
    """Convert model probabilities into a thresholded, serializable prediction."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    values = [float(value) for value in probabilities]
    if len(values) != len(class_names):
        raise ValueError("Probability count must match the number of manifest classes.")
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Probabilities must be a non-empty finite vector.")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    ordered_indices = sorted(range(len(values)), key=values.__getitem__, reverse=True)
    best_index = ordered_indices[0]
    class_name = class_names[best_index]
    confidence = float(values[best_index])
    alternatives = [
        {
            "class_name": class_names[index],
            "display_name": display_label(class_names[index]),
            "confidence": values[index],
        }
        for index in ordered_indices[:top_k]
    ]
    result = {
        "accepted": confidence >= threshold,
        "threshold": threshold,
        "confidence": confidence,
        "top_predictions": alternatives,
    }
    if confidence < threshold:
        result.update({"message": LOW_CONFIDENCE_MESSAGE, "prediction": None, "advisory": None})
        return result

    advisory = (advisories or {}).get(class_name)
    result.update(
        {
            "message": "Prediction accepted. Confirm with a local expert before treatment.",
            "prediction": {
                "class_name": class_name,
                "display_name": display_label(class_name),
            },
            "advisory": advisory,
        }
    )
    if advisory is None:
        result["advisory_message"] = (
            "No seeded advisory is available for this class yet. Consult an agricultural expert."
        )
    return result


def predict_image(model, image_path, class_names, threshold=0.65, advisories=None, top_k=3):
    """Preprocess one image, run a Keras model, and interpret its output."""
    import tensorflow as tf

    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError("Image does not exist: {}".format(image_path))
    input_shape = model.input_shape
    height, width = input_shape[1:3]
    if height is None or width is None:
        height, width = IMAGE_SIZE
    image = tf.keras.utils.load_img(image_path, target_size=(height, width))
    image_array = tf.keras.utils.img_to_array(image)
    probabilities = model.predict(tf.expand_dims(image_array, axis=0), verbose=0)[0]
    return interpret_probabilities(probabilities, class_names, threshold, advisories, top_k)


def _print_human_readable(result):
    print("Confidence: {:.1%}".format(result["confidence"]))
    if not result["accepted"]:
        print(result["message"])
        return
    print("Prediction: {}".format(result["prediction"]["display_name"]))
    print(result["message"])
    advisory = result["advisory"]
    if advisory is None:
        print(result["advisory_message"])
        return
    print("Advice: {}".format(advisory["summary"]))
    print("Immediate actions: {}".format("; ".join(advisory["immediate_actions"])))
    print("Prevention: {}".format("; ".join(advisory["prevention"])))
    print("Source: {}".format(advisory["source"]["url"]))


def _main():
    parser = argparse.ArgumentParser(description="Predict one PlantVillage leaf image and show advice.")
    parser.add_argument("--image", required=True, help="Path to a leaf image.")
    parser.add_argument("--model", default="results/centralized/centralized_model.keras")
    parser.add_argument("--manifest", default="results/data_manifest.json")
    parser.add_argument("--advisory", default="fasalmitra/advisory.json")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    with Path(args.manifest).open(encoding="utf-8") as file:
        manifest = json.load(file)
    advisories = load_advisories(args.advisory)
    import tensorflow as tf

    model = tf.keras.models.load_model(args.model)
    result = predict_image(
        model, args.image, manifest["class_names"], args.threshold, advisories, args.top_k
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human_readable(result)


if __name__ == "__main__":
    _main()
