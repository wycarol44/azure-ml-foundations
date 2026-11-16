
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download CIFAR-10 automatically.
# The original download is cached by Keras in its local dataset cache.
(x_train_full, y_train_full), (x_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

# Convert labels from shape (n, 1) to (n,)
y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

# Create 90% training and 10% validation
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full,
    y_train_full,
    test_size=0.10,
    random_state=42,
    stratify=y_train_full
)

# Save all three datasets in one file
output_file = DATA_DIR / "cifar10_split.npz"

np.savez_compressed(
    output_file,
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    x_test=x_test,
    y_test=y_test
)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("Dataset prepared successfully.")
print("Saved to:", output_file)
print("Training images:", len(x_train))
print("Validation images:", len(x_val))
print("Test images:", len(x_test))
print("Classes:", class_names)
