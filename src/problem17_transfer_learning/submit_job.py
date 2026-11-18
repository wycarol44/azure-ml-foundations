from azure.ai.ml import MLClient, command, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment
from azure.identity import InteractiveBrowserCredential


# Connect to your existing Azure ML workspace.
ml_client = MLClient.from_config(
    credential=InteractiveBrowserCredential()
)

# Create the training environment.
env = Environment(
    name="problem17-transfer-learning-env",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
)

# Define the Azure ML command job.
job = command(
    code=".",
    command=(
        "python train.py "
        "--head_epochs 5 "
        "--finetune_epochs 5 "
        "--output_dir ${{outputs.results}}"
    ),
    environment=env,
    compute="YOUR_COMPUTE_CLUSTER_NAME",
    experiment_name="problem17-transfer-learning",
    display_name="mobilenetv2-transfer-learning",
    outputs={
        "results": Output(type=AssetTypes.URI_FOLDER)
    }
)

# Submit the job.
returned_job = ml_client.jobs.create_or_update(job)

print("Job submitted successfully!")
print("Job name:", returned_job.name)
print("Studio URL:", returned_job.studio_url)