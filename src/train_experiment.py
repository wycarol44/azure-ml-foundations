import argparse
import json
import os

import joblib
import mlflow

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--run-name", type=str, required=True)
parser.add_argument("--C", type=float, default=1.0)
parser.add_argument("--n-estimators", type=int, default=100)
parser.add_argument("--max-depth", type=int, default=10)
parser.add_argument("--learning-rate", type=float, default=0.05)

args = parser.parse_args()


df = pd.read_csv(args.data)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)

df = df.drop(columns=["customerID"])

y = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Churn"])

numeric = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical = X.select_dtypes(
    include=["object"]
).columns.tolist()


preprocessor = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        numeric
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical
    )
])


if args.model == "logistic":
    classifier = LogisticRegression(
        C=args.C,
        max_iter=1000
    )

elif args.model == "random_forest":
    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

elif args.model == "gradient_boosting":
    classifier = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        random_state=42
    )

else:
    raise ValueError("Unsupported model")


pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


with mlflow.start_run(run_name=args.run_name):

    mlflow.log_param("model", args.model)
    mlflow.log_param("C", args.C)
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }

    mlflow.log_metrics(metrics)

    os.makedirs("outputs", exist_ok=True)

    joblib.dump(pipeline, "outputs/model.pkl")

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    mlflow.log_artifact("outputs/model.pkl")
    mlflow.log_artifact("outputs/metrics.json")

    print(metrics)