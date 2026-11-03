# Problem 7 — Feature Engineering Reflection

## 1. Experiment Information

- **Date:** November 3, 2026
- **Dataset source/version:** IBM Telco Customer Churn dataset; version to be verified
- **Number of rows / columns:** 7,043 rows / 21 columns (inferred from test set size; verify against actual dataset)
- **Random seed:** 42 (assuming the same configuration as the previous experiment)
- **Train/test split:** 80/20
- **Class balance (churn / no churn):** Test set: 374 churn customers (26.5%) and 1,035 non-churn customers (73.5%)

## 2. Hypotheses Before Training

| Engineered feature | Business/technical rationale | Expected effect |
|---|---|---|
| `tenure_group` | Groups customers by how long they have stayed with the company. Customer churn behavior may differ between new and long-term customers. | May help the model identify nonlinear churn patterns and improve recall or F1. |
| `avg_monthly_spend` | Represents the customer's average monthly spending and may help identify customers with higher financial commitments. | May improve the model's ability to identify customers at risk of churning. |
| `lifetime_value_proxy` | Estimates the customer's accumulated value using tenure and monthly charges. | May help distinguish customer segments with different retention value. |
| `service_count` | Counts the number of services subscribed to by each customer. Customers using more services may have different churn behavior. | May improve customer segmentation and predictive performance. |
| `has_support_or_security` | Indicates whether a customer subscribes to technical support or online security services. These services may be associated with customer engagement and retention. | May help identify differences in churn risk between service groups. |
| Other: | No additional engineered features confirmed. | To be evaluated in future experiments. |

## 3. Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (original features) | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8413 | Not recorded |
| Engineered features | 0.7324 | 0.4975 | 0.7914 | 0.6109 | 0.8420 | Not recorded |

### Confusion Matrices

**Baseline Model**

```text
[[747 288]
 [ 81 293]]
```

**Engineered Model**

```text
[[736 299]
 [ 78 296]]
```

## 4. Interpretation

1. **Did feature engineering improve the primary metric? By how much?**

   Assuming F1 is the primary metric, feature engineering did not improve overall performance. F1 decreased from 0.6136 to 0.6109, a reduction of approximately 0.0027 (0.27 percentage points).

2. **Which metric changed most: precision, recall, F1, or ROC-AUC?**

   Recall increased from 0.7834 to 0.7914, an improvement of approximately 0.80 percentage points. Precision decreased by approximately 0.68 percentage points, while F1 decreased by approximately 0.27 percentage points. ROC-AUC increased slightly from 0.8413 to 0.8420.

3. **Which engineered features seem useful? What evidence supports this?**

   The small increase in recall and ROC-AUC suggests that the engineered model may identify some additional churn customers and slightly improve its ability to distinguish between churn and non-churn customers. However, these aggregate metrics do not establish which individual feature contributed to the changes. Further feature-level analysis is needed.

4. **Did any engineered feature appear redundant or noisy?**

   The overall results do not show a meaningful improvement in F1 or accuracy. This may indicate that the engineered features provide limited additional information, or that some features overlap with information already present in the original dataset. Individual feature analysis is needed before deciding which features to remove.

5. **Did the model become slower or more complex?**

   Fit time was not recorded, so training speed cannot be compared. The engineered model requires additional feature preparation, which increases preprocessing complexity.

## 5. Limitations and Next Experiment

- **Potential leakage or data-quality concerns:** Ensure that feature engineering is applied consistently to training and testing data. All preprocessing and learned transformations should be fitted using training data only. Verify that engineered features do not use information unavailable at prediction time.
- **Features I would remove or redesign:** Review features that overlap with existing variables. Evaluate whether `tenure_group` adds useful information beyond the original `tenure` feature.
- **One experiment I would run next:** Compare the baseline and engineered models using cross-validation and feature ablation. Add engineered features one at a time to identify which features contribute meaningful improvements. Record fit time and compare F1, recall, and ROC-AUC.

## 6. Conclusion

Feature engineering produced a small increase in recall and ROC-AUC, but slightly reduced accuracy, precision, and F1. The results suggest that the engineered features may help identify more customers who are likely to churn, although the overall improvement is limited. I would keep the engineered features for further investigation rather than assuming they improve the model. My next step would be to evaluate individual features through cross-validation and feature ablation to determine which features provide measurable value.