import os

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

subscription_id = (
    os.getenv("AZURE_SUBSCRIPTION_ID")
    or os.getenv("SUBSCRIPTION_ID")
)

if not subscription_id:
    raise ValueError(
        "Missing Azure subscription ID. Set AZURE_SUBSCRIPTION_ID or SUBSCRIPTION_ID "
        "before running this script."
    )

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=os.getenv("AZURE_RESOURCE_GROUP", "rg-aml-foundations"),
    workspace_name=os.getenv("AZURE_ML_WORKSPACE", "aml-foundations-ws")
)

configs = [
    {
        "name": "logistic-c1",
        "model": "logistic",
        "C": 1.0
    },
    {
        "name": "logistic-c01",
        "model": "logistic",
        "C": 0.1
    },
    {
        "name": "rf-depth10",
        "model": "random_forest",
        "n_estimators": 100,
        "max_depth": 10
    },
    {
        "name": "rf-depth20",
        "model": "random_forest",
        "n_estimators": 200,
        "max_depth": 20
    },
    {
        "name": "gradient-boosting",
        "model": "gradient_boosting",
        "n_estimators": 100,
        "learning_rate": 0.05
    }
]

for config in configs:

    model = config["model"]

    job = command(
        code=".",
        command=(
            f"python src/train_experiment.py "
            f"--data data/telco_churn_v2.csv "
            f"--model {model} "
            f"--run-name {config['name']} "
            + " ".join(
                f"--{key.replace('_', '-')} {value}"
                for key, value in config.items()
                if key not in ["name", "model"]
            )
        ),
        environment="azureml:churn-training-env:1",
        compute="cpu-cluster",
        experiment_name="customer-churn-classification",
        display_name=config["name"]
    )

    submitted = ml_client.jobs.create_or_update(job)

    print(config["name"], submitted.name)