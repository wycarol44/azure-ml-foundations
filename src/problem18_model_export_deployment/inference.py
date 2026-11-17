
import json
import argparse
import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "artifacts" / "model.keras"
CONFIG_PATH = BASE_DIR / "artifacts" / "model_config.json"


def load_model_and_config():

    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load preprocessing configuration
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    return model, config


def preprocess_image(image_path, config):

    image_size = tuple(config["input_size"])

    # Load image and resize
    image = tf.keras.utils.load_img(
        image_path,
        target_size=image_size,
        color_mode="rgb"
    )

    # Convert image to numpy array
    image = tf.keras.utils.img_to_array(image)

    # Apply MobileNetV2 preprocessing
    image = preprocess_input(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


def predict(image_path):

    model, config = load_model_and_config()

    image = preprocess_image(image_path, config)

    # Run prediction
    probability = float(
        model.predict(image, verbose=0)[0][0]
    )

    threshold = config.get("threshold", 0.5)
    class_names = config["class_names"]

    predicted_index = 1 if probability >= threshold else 0

    confidence = (
        probability
        if predicted_index == 1
        else 1 - probability
    )

    result = {
        "image": str(image_path),
        "prediction": class_names[predicted_index],
        "confidence": round(confidence, 4),
        "dog_probability": round(probability, 4)
    }

    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image"
    )

    args = parser.parse_args()

    result = predict(args.image)

    print(json.dumps(result, indent=4))
