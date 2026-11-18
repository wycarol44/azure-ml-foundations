
import os
import json
import base64
import numpy as np
import tensorflow as tf

from io import BytesIO
from PIL import Image
from pathlib import Path

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def init():

    global model
    global config

    # Azure ML model directory
    model_dir = Path(os.environ["AZUREML_MODEL_DIR"])

    # Load model
    model = tf.keras.models.load_model(
        model_dir / "model.keras"
    )

    # Load config
    with open(model_dir / "model_config.json", "r") as f:
        config = json.load(f)


def run(raw_data):

    try:

        data = json.loads(raw_data)

        # Decode Base64 image
        image_bytes = base64.b64decode(
            data["image_base64"],
            validate=True
        )

        # Decode image and convert to RGB
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False
        )

        # Resize image
        image = tf.image.resize(
            image,
            tuple(config["input_size"])
        )

        # Apply MobileNetV2 preprocessing
        image = preprocess_input(image)

        # Add batch dimension
        image = tf.expand_dims(image, axis=0)

        # Run inference
        probability = float(
            model.predict(image, verbose=0)[0][0]
        )

        threshold = config.get("threshold", 0.5)

        class_names = config["class_names"]

        index = 1 if probability >= threshold else 0

        confidence = (
            probability
            if index == 1
            else 1 - probability
        )

        return {
            "prediction": class_names[index],
            "confidence": round(confidence, 4),
            "dog_probability": round(probability, 4)
        }

    except Exception as e:

        return {
            "error": str(e)
        }
