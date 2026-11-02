# Problem 6 - Azure ML Designer

Pipeline:
customer-churn-designer-pipeline


flowchart TD
    A["Telco Churn Dataset v2"] --> B["Clean Missing Data"]
    B --> C["Select Columns"]
    C --> D["Split Data 80/20"]

    D -->|Training data| E["Train Logistic Regression"]
    D -->|Testing data| F["Score Logistic Regression"]
    E --> F
    F --> G["Evaluate Logistic Regression"]

    D -->|Training data| H["Train Decision Forest"]
    D -->|Testing data| I["Score Decision Forest"]
    H --> I
    I --> J["Evaluate Decision Forest"]


Dataset:
telco-churn v2

Preprocessing:
- Removed customerID
- Handled missing values
- Selected Churn as target

Split:
- 80% training
- 20% testing
- Random seed: 42

Algorithms:
1. Logistic Regression
2. Decision Forest

Results:
- Logistic Regression metrics:
- Decision Forest metrics:

Best algorithm:
[Your choice]

Reason:
[Explain using actual results]