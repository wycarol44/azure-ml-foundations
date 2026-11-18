
from pathlib import Path

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, Output
from azure.identity import InteractiveBrowserCredential

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Azure workspace settings
SUBSCRIPTION_ID = "8599274b-cd20-4061-b400-3e92c97f5d71"
RESOURCE_GROUP = "rg-aml-foundations"
WORKSPACE_NAME = "aml-foundations-ws"

# Replace with your actual Azure compute cluster name
COMPUTE_NAME = "cnn-compute"

# Authenticate using browser login
credential = InteractiveBrowserCredential()

# Connect to Azure ML workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME
)

# Create the remote training environment
environment = Environment(
    name="cnn-image-classification-env",
    description="TensorFlow environment for CIFAR-10 CNN training",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
    conda_file=str(PROJECT_DIR / "environment.yml")
)

# Define Azure ML command job
job = command(
    code=str(PROJECT_DIR),

    command=(
        "python train.py "
        "--data_file data/cifar10_split.npz "
        "--output_dir ${{outputs.results}} "
        "--epochs 15"
    ),

    environment=environment,
    compute=COMPUTE_NAME,

    experiment_name="problem14-cnn-image-classification",
    display_name="cifar10-cnn-training",

    outputs={
        "results": Output(type="uri_folder")
    }
)

# Submit job
submitted_job = ml_client.jobs.create_or_update(job)

print("Job submitted successfully.")
print("Job name:", submitted_job.name)
print("Job status:", submitted_job.status)
print("Studio URL:", submitted_job.studio_url)
