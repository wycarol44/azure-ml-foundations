
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

VOCAB_SIZE = 10000
MAX_LENGTH = 200
RANDOM_STATE = 42


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading IMDB dataset...")

    (x_train_full, y_train_full), (x_test, y_test) = imdb.load_data(
        num_words=VOCAB_SIZE
    )

    print("Padding sequences...")

    x_train_full = pad_sequences(
        x_train_full,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    x_test = pad_sequences(
        x_test,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    y_train_full = np.asarray(y_train_full, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    output_file = DATA_DIR / "imdb_sequences.npz"

    np.savez_compressed(
        output_file,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test
    )

    print("\nDataset preparation complete.")
    print(f"Training samples:   {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples:       {len(x_test)}")
    print(f"Sequence length:    {MAX_LENGTH}")
    print(f"Vocabulary size:    {VOCAB_SIZE}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
