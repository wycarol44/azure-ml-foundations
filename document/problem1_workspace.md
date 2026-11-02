# Problem 1 - Azure ML Workspace

Workspace name: aml-foundations-ws
Resource group: rg-aml-foundations
Region: East US
Subscription: [your subscription]

Associated resources:
- Storage account: amlfoundations7716083272
- Key Vault: amlfoundations8560120597
- Application Insights: amlfoundations0112830974
- Container Registry:

## Workspace role

The Azure ML workspace is the central management
boundary for machine learning assets and activities.

It organizes:
- Data assets
- Compute resources
- Environments
- Jobs and experiments
- Models
- Components and pipelines

The workspace does not itself execute model training.
Compute resources execute the workloads.