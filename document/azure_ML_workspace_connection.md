## Azure ML Workspace Connection

To connect the local Python project to Azure Machine Learning, configure the Azure subscription, resource group, and workspace information.

### 1. Find Your Azure Subscription ID

1. Sign in to the [Azure Portal](https://portal.azure.com/).
2. Search for **Subscriptions**.
3. Select your Azure subscription.
4. Copy the **Subscription ID** from the subscription overview page.

Alternatively, use Azure CLI:

```bash
az login
az account show --query id --output tsv
```

### 2. Configure the MLClient

Install the required packages:

```bash
pip install azure-ai-ml azure-identity
```

Connect to your Azure ML workspace:

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<SUBSCRIPTION_ID>",
    resource_group_name="rg-aml-foundations",
    workspace_name="aml-foundations-ws"
)
```

Replace `<SUBSCRIPTION_ID>` with your actual Azure Subscription ID.

### 3. Authenticate and Verify the Connection

Authenticate using Azure CLI:

```bash
az login
```

Verify that the workspace connection works:

```python
workspace = ml_client.workspaces.get("aml-foundations-ws")

print("Workspace:", workspace.name)
print("Location:", workspace.location)
```

### 4. Prerequisites

Before running the code, make sure:

- An active Azure subscription is available.
- Resource group `rg-aml-foundations` has been created.
- Azure ML workspace `aml-foundations-ws` has been created.
- Your account has permission to access the workspace.
- Azure CLI authentication is successful.

**Note:** Azure ML compute and other cloud resources may incur charges. Stop or delete unused compute resources to avoid unnecessary costs.

**Security:** Do not commit subscription credentials, passwords, access tokens, or other secrets to GitHub. The Subscription ID is an identifier, not an authentication secret.