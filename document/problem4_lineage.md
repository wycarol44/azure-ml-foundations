# Problem 4 - Data Asset Versioning

**Dataset:** Telco Customer Churn

## Version 1
- **Source:** Original public CSV
- **Row count:** 7,043
- **Schema:** 21 columns, including customer information, service subscriptions, contract details, billing information, and the `Churn` target variable.
- **Data type:** `TotalCharges` was originally stored as a string/object column.

## Version 2
- **Cleaning:** Converted `TotalCharges` to a numeric data type.
- **Invalid rows:** Removed rows with invalid or missing `TotalCharges` values.
- **Row count:** 7,032
- **Schema:** Same 21 columns as Version 1, with `TotalCharges` converted to a numeric data type.

## Training
- **Model:** Logistic Regression
- **Random seed:** 42
- **Train/test split:** 80/20
- **Comparison:** The same model configuration and evaluation process were used for both dataset versions.

## Results

| Metric | Version 1 | Version 2 |
|---|---:|---:|
| Accuracy | 0.8055 | 0.8038 |
| F1 | 0.6040 | 0.6080 |
| ROC-AUC | 0.8419 | 0.8359 |

## Conclusion

Cleaning the data resulted in small changes in model performance.

Version 1 achieved slightly higher accuracy (0.8055 vs. 0.8038) and ROC-AUC (0.8419 vs. 0.8359). However, Version 2 achieved a slightly higher F1 score (0.6080 vs. 0.6040).

These results suggest that removing invalid `TotalCharges` records did not significantly improve overall model performance. Version 2 contains fewer records but has more consistent data types, making it cleaner and more suitable for machine learning workflows.

Since the same model, random seed, and train/test split were used, the experiment demonstrates how data preparation and dataset versioning can affect model evaluation results.

Overall, Version 2 provides cleaner and more consistent data, while Version 1 performs slightly better in terms of accuracy and ROC-AUC. The preferred version depends on the evaluation metric and the business objective.