
from pathlib import Path

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, Output
from azure.identity import InteractiveBrowserCredential


# Project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Azure workspace configuration
SUBSCRIPTION_ID = "8599274b-cd20-4061-b400-3e92c97f5d71"
RESOURCE_GROUP = "rg-aml-foundations"
WORKSPACE_NAME = "aml-foundations-ws"

# Replace with your actual compute cluster name
COMPUTE_NAME = "rnn-lstm-cpu-cluster"


def main():

    # Authenticate through browser
    credential = InteractiveBrowserCredential()

    # Connect to Azure ML workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )

    print("Connected to Azure ML workspace.")

    # Define the training environment
    env = Environment(
        name="rnn-lstm-training-env",
        description="TensorFlow environment for RNN and LSTM sentiment analysis",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
        conda_file=str(PROJECT_DIR / "environment.yml")
    )

    # Register/update environment
    env = ml_client.environments.create_or_update(env)

    print(f"Environment ready: {env.name}:{env.version}")

    # Run both models sequentially, then compare results
    job_command = (
        "python train.py "
        "--model_type rnn "
        "--data_file data/imdb_sequences.npz "
        "--output_dir ${{outputs.results}}/rnn "
        "--epochs 8 "
        "--batch_size 128 "
        "&& "
        "python train.py "
        "--model_type lstm "
        "--data_file data/imdb_sequences.npz "
        "--output_dir ${{outputs.results}}/lstm "
        "--epochs 8 "
        "--batch_size 128 "
        "&& "
        "python compare_models.py "
        "--results_dir ${{outputs.results}}"
    )

    job = command(
        code=str(PROJECT_DIR),
        command=job_command,
        environment=env,
        compute=COMPUTE_NAME,
        experiment_name="problem15-rnn-lstm-sentiment",
        display_name="imdb-rnn-lstm-comparison",
        description=(
            "Train and compare Simple RNN and LSTM "
            "models on the IMDB sentiment dataset"
        ),
        outputs={
            "results": Output(
                type="uri_folder",
                mode="rw_mount"
            )
        }
    )

    submitted_job = ml_client.jobs.create_or_update(job)

    print("\nJob submitted successfully.")
    print(f"Job name: {submitted_job.name}")
    print(f"Studio URL: {submitted_job.studio_url}")


if __name__ == "__main__":
    main()
