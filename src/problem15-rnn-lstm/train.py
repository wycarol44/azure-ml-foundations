
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger
)


def build_model(model_type, vocab_size=10000, max_length=200):
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=64,
            input_length=max_length,
            mask_zero=True
        )
    ])

    if model_type == "rnn":
        model.add(SimpleRNN(64))
    elif model_type == "lstm":
        model.add(LSTM(64))
    else:
        raise ValueError("model_type must be rnn or lstm")

    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_history(history, output_dir):
    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(output_dir / "history.json", "w") as f:
        json.dump(history_dict, f, indent=4)

    plt.figure(figsize=(10, 4))

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
    plt.savefig(output_dir / "training_history.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        choices=["rnn", "lstm"],
        required=True
    )

    parser.add_argument(
        "--data_file",
        default="data/imdb_sequences.npz"
    )

    parser.add_argument(
        "--output_dir",
        default="outputs"
    )

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)

    # Optional small dataset for local smoke testing
    parser.add_argument("--max_samples", type=int, default=0)

    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.data_file}")

    data = np.load(args.data_file)

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    if args.max_samples > 0:
        x_train = x_train[:args.max_samples]
        y_train = y_train[:args.max_samples]
        x_val = x_val[:args.max_samples]
        y_val = y_val[:args.max_samples]
        x_test = x_test[:args.max_samples]
        y_test = y_test[:args.max_samples]

    print(f"Model: {args.model_type.upper()}")
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")

    model = build_model(args.model_type)

    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True
        ),
        CSVLogger(str(output_dir / "training_log.csv"))
    ]

    start_time = time.time()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2
    )

    training_seconds = time.time() - start_time

    print("\nEvaluating on test data...")

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    probabilities = model.predict(x_test, verbose=0).flatten()
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        target_names=["Negative", "Positive"],
        output_dict=True
    )

    matrix = confusion_matrix(y_test, predictions)

    model.save(output_dir / "final_model.keras")

    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(
            classification_report(
                y_test,
                predictions,
                target_names=["Negative", "Positive"]
            )
        )

    metrics = {
        "model_type": args.model_type,
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy),
        "test_f1": float(f1),
        "training_seconds": float(training_seconds),
        "epochs_completed": len(history.history["loss"])
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Negative", "Positive"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{args.model_type.upper()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    save_history(history, output_dir)

    print("\nEvaluation Results")
    print(json.dumps(metrics, indent=4))

    print("\nClassification Report")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Negative", "Positive"]
    ))

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
