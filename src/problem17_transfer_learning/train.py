
import argparse
import json
import os
import time
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt


DATA_URL = (
    "https://storage.googleapis.com/"
    "mledu-datasets/cats_and_dogs_filtered.zip"
)

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def load_data(max_batches=None):
    print("Downloading Cats vs Dogs dataset...")

    zip_path = tf.keras.utils.get_file(
        fname="cats_and_dogs_filtered.zip",
        origin=DATA_URL,
        extract=True
    )

    base_dir = Path(zip_path).parent / "cats_and_dogs_filtered"
    train_dir = base_dir / "train"
    validation_dir = base_dir / "validation"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True,
        seed=42
    )

    # Split the original validation directory into
    # separate validation and test subsets.
    val_ds = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        validation_split=0.5,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        validation_split=0.5,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    def preprocess(images, labels):
        images = tf.keras.applications.mobilenet_v2.preprocess_input(
            images
        )
        return images, labels

    train_ds = train_ds.map(
        preprocess, num_parallel_calls=AUTOTUNE
    )
    val_ds = val_ds.map(
        preprocess, num_parallel_calls=AUTOTUNE
    )
    test_ds = test_ds.map(
        preprocess, num_parallel_calls=AUTOTUNE
    )

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    if max_batches:
        train_ds = train_ds.take(max_batches)
        val_ds = val_ds.take(max_batches)
        test_ds = test_ds.take(max_batches)

    return train_ds, val_ds, test_ds


def build_model():
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        weights="imagenet"
    )

    # Stage 1: Freeze pretrained backbone.
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(160, 160, 3))

    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    return model, backbone


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )


def train_stage(
    model,
    train_ds,
    val_ds,
    epochs,
    learning_rate,
    stage_name
):
    compile_model(model, learning_rate)

    print(f"\nStarting {stage_name}")
    print("Learning rate:", learning_rate)
    print("Trainable parameters:",
          sum(
              tf.size(v).numpy()
              for v in model.trainable_variables
          ))

    start = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            )
        ]
    )

    elapsed = time.time() - start

    print(f"{stage_name} completed in {elapsed:.2f} seconds")

    return history, elapsed


def plot_history(head_history, fine_history, output_dir):
    metrics = ["accuracy", "loss"]

    for metric in metrics:
        train_values = (
            head_history.history[metric]
            + fine_history.history[metric]
        )
        val_values = (
            head_history.history[f"val_{metric}"]
            + fine_history.history[f"val_{metric}"]
        )

        plt.figure(figsize=(8, 5))
        plt.plot(train_values, label="Training")
        plt.plot(val_values, label="Validation")

        boundary = len(head_history.history[metric])
        plt.axvline(
            boundary - 1,
            linestyle="--",
            label="Fine-tuning begins"
        )

        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.title(f"Transfer Learning - {metric.capitalize()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_curve.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--head_epochs", type=int, default=5)
    parser.add_argument("--finetune_epochs", type=int, default=5)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("TensorFlow version:", tf.__version__)
    print("Available GPUs:", tf.config.list_physical_devices("GPU"))

    train_ds, val_ds, test_ds = load_data(
        max_batches=args.max_batches or None
    )

    model, backbone = build_model()

    # Stage 1: Train the new classification head.
    head_history, head_time = train_stage(
        model,
        train_ds,
        val_ds,
        args.head_epochs,
        1e-3,
        "Stage 1 - Feature Extraction"
    )

    # Stage 2: Unfreeze selected backbone layers.
    backbone.trainable = True

    for layer in backbone.layers[:-30]:
        layer.trainable = False

    # Keep BatchNormalization layers frozen.
    for layer in backbone.layers:
        if isinstance(
            layer, tf.keras.layers.BatchNormalization
        ):
            layer.trainable = False

    fine_history, fine_time = train_stage(
        model,
        train_ds,
        val_ds,
        args.finetune_epochs,
        1e-5,
        "Stage 2 - Fine-tuning"
    )

    print("\nEvaluating final model...")
    test_results = model.evaluate(test_ds, return_dict=True)

    model.save(output_dir / "transfer_learning_model.keras")

    plot_history(head_history, fine_history, output_dir)

    results = {
        "head_training_time_seconds": head_time,
        "finetuning_time_seconds": fine_time,
        "total_training_time_seconds": head_time + fine_time,
        "test_metrics": {
            key: float(value)
            for key, value in test_results.items()
        },
        "head_epochs_completed": len(
            head_history.history["loss"]
        ),
        "finetune_epochs_completed": len(
            fine_history.history["loss"]
        ),
        "backbone_trainable_layers": sum(
            1 for layer in backbone.layers
            if layer.trainable
        )
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nFinal results:")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
