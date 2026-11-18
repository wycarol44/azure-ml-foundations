
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Model,
    Environment,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration
)
from azure.ai.ml.constants import AssetTypes

from azure.identity import InteractiveBrowserCredential


# ---------------------------------
# Azure configuration
# ---------------------------------

SUBSCRIPTION_ID = "YOUR_SUBSCRIPTION_ID"

RESOURCE_GROUP = "rg-aml-foundations"

WORKSPACE_NAME = "aml-foundations-ws"

ENDPOINT_NAME = "cats-dogs-inference"


# ---------------------------------
# Authenticate
# ---------------------------------

credential = InteractiveBrowserCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME
)


# ---------------------------------
# Register model
# ---------------------------------

model = Model(
    path="artifacts",
    name="cats-dogs-transfer-learning",
    type=AssetTypes.CUSTOM_MODEL,
    description="MobileNetV2 Cats vs Dogs model"
)

registered_model = ml_client.models.create_or_update(model)


# ---------------------------------
# Create environment
# ---------------------------------

environment = Environment(
    name="problem18-inference-env",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"
)

registered_environment = ml_client.environments.create_or_update(
    environment
)


# ---------------------------------
# Create endpoint
# ---------------------------------

endpoint = ManagedOnlineEndpoint(
    name=ENDPOINT_NAME,
    description="Cats vs Dogs image classification",
    auth_mode="key"
)

ml_client.online_endpoints.begin_create_or_update(
    endpoint
).result()


# ---------------------------------
# Create deployment
# ---------------------------------

deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=ENDPOINT_NAME,
    model=registered_model,
    environment=registered_environment,
    code_configuration=CodeConfiguration(
        code=".",
        scoring_script="score.py"
    ),
    instance_type="Standard_DS3_v2",
    instance_count=1
)

ml_client.online_deployments.begin_create_or_update(
    deployment
).result()


# ---------------------------------
# Configure traffic
# ---------------------------------

endpoint = ml_client.online_endpoints.get(
    ENDPOINT_NAME
)

endpoint.traffic = {
    "blue": 100
}

ml_client.online_endpoints.begin_create_or_update(
    endpoint
).result()


print("Deployment completed successfully.")
print("Endpoint:", ENDPOINT_NAME)
