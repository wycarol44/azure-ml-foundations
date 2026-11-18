from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential


SUBSCRIPTION_ID = "YOUR_SUBSCRIPTION_ID"

ml_client = MLClient(
    InteractiveBrowserCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name="rg-aml-foundations",
    workspace_name="aml-foundations-ws"
)

result = ml_client.online_endpoints.invoke(
    endpoint_name="cats-dogs-inference",
    request_file="request.json"
)

print("Endpoint prediction:")
print(result)