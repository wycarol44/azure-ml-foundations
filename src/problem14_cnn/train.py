
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", default="data/cifar10_split.npz")
parser.add_argument("--output_dir", default="outputs")
parser.add_argument("--epochs", type=int, default=15)
args = parser.parse_args()

DATA_FILE = Path(args.data_file)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 1. Load prepared dataset
data = np.load(DATA_FILE)

x_train = data["x_train"]
y_train = data["y_train"]
x_val = data["x_val"]
y_val = data["y_val"]
x_test = data["x_test"]
y_test = data["y_test"]

print("Training:", x_train.shape)
print("Validation:", x_val.shape)
print("Testing:", x_test.shape)

# 2. Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Build CNN
model = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 4. Train model
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(OUTPUT_DIR / "best_model.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=args.epochs,
    batch_size=64,
    callbacks=callbacks
)

# 5. Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_history.png")
plt.close()

# 6. Evaluate test dataset
test_loss, test_accuracy = model.evaluate(
    x_test, y_test, verbose=1
)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# 7. Generate predictions
y_prob = model.predict(x_test)
y_pred = np.argmax(y_prob, axis=1)

# 8. Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=range(10)
)

plt.figure(figsize=(11, 9))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.title("CIFAR-10 Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.close()

# 9. Classification report
report = classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)

with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
    f.write(report)

# 10. Inspect misclassified images
incorrect_indices = np.where(y_test != y_pred)[0]

print("Misclassified images:", len(incorrect_indices))

num_display = min(12, len(incorrect_indices))

if num_display > 0:
    plt.figure(figsize=(12, 9))

    for i, index in enumerate(incorrect_indices[:num_display]):
        plt.subplot(3, 4, i + 1)
        plt.imshow(x_test[index])

        actual = CLASS_NAMES[int(y_test[index])]
        predicted = CLASS_NAMES[int(y_pred[index])]
        confidence = float(np.max(y_prob[index]))

        plt.title(
            f"Actual: {actual}\n"
            f"Predicted: {predicted}\n"
            f"Confidence: {confidence:.2f}"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "misclassified_images.png")
    plt.close()

# 11. Save misclassification details
error_records = []

for index in incorrect_indices:
    error_records.append({
        "image_index": int(index),
        "actual_class": CLASS_NAMES[int(y_test[index])],
        "predicted_class": CLASS_NAMES[int(y_pred[index])],
        "confidence": float(np.max(y_prob[index]))
    })

import pandas as pd

pd.DataFrame(error_records).to_csv(
    OUTPUT_DIR / "misclassified_images.csv",
    index=False
)

# 12. Save model and metrics
model.save(OUTPUT_DIR / "final_cnn_model.keras")

with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump({
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "misclassified_count": int(len(incorrect_indices)),
        "class_names": CLASS_NAMES
    }, f, indent=4)

print("Training and evaluation completed.")
print("Results saved to:", OUTPUT_DIR.resolve())
