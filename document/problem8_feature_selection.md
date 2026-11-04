# Problem 8 — Feature Selection Reflection

## 1. Experiment Information

- **Date:** November 3, 2026
- **Dataset:** IBM Telco Customer Churn
- **Dataset version:** To be confirmed
- **Original number of features:** 
- **Selected number of features:** 
- **Feature selection method:** 
- **Model:** Logistic Regression
- **Random seed:** 42
- **Train/test split:** 80/20

## 2. Feature Selection Approach

### Objective

The objective of this experiment is to reduce the number of input features while maintaining or improving model performance.

Feature selection can help:
- Remove irrelevant or redundant features.
- Reduce model complexity.
- Improve training efficiency.
- Make model results easier to interpret.
- Identify the most relevant factors associated with customer churn.

### Method

- **Selection technique:** 
- **Selection criteria:** 
- **Number of features before selection:** 
- **Number of features after selection:** 

## 3. Selected Features

| Feature | Selected? | Business/Technical Rationale |
|---|---|---|
| tenure | | Customer relationship duration |
| MonthlyCharges | | Monthly customer spending |
| TotalCharges | | Total accumulated charges |
| Contract | | Contract commitment may influence churn |
| InternetService | | Type of internet service |
| OnlineSecurity | | Availability of security services |
| TechSupport | | Availability of technical support |
| PaymentMethod | | Payment preferences and behavior |
| PaperlessBilling | | Billing preferences |
| Other features | | |

### Final Selected Feature List

```text
To be completed after running feature selection.
```

## 4. Model Performance Comparison

| Metric | Baseline (All Features) | Selected Features | Difference |
|---|---:|---:|---:|
| Accuracy | | | |
| Precision | | | |
| Recall | | | |
| F1 | | | |
| ROC-AUC | | | |
| Training Time (s) | | | |

### Confusion Matrix

**Baseline Model**

```text
To be completed.
```

**Feature-Selected Model**

```text
To be completed.
```

## 5. Analysis Questions

### 8.6 Analysis Questions

**1. Did feature selection improve performance?**

- Did Accuracy, F1, and ROC-AUC increase or decrease?
- Was the performance difference significant or relatively small?
- Did feature selection improve the primary evaluation metric?

Answer:

**2. Did it reduce training time?**

- Compare training time before and after feature selection.
- Did reducing the number of features make model training faster?
- Was the improvement meaningful?

Answer:

**3. Which features were selected?**

- List the final selected features.
- How many features were removed?
- Which features were consistently selected by the selection method?

Answer:

**4. Did the selected features make business sense?**

- Are the selected features related to customer behavior, service usage, billing, or contract commitments?
- Do they align with common business explanations for customer churn?
- Are any unexpected features selected?

Answer:

**5. Did the model lose useful information?**

- Did removing features reduce Recall, F1, or ROC-AUC?
- Could any removed features contain useful information about customer churn?
- Did the model become less effective at identifying customers likely to churn?

Answer:

**6. Would you prefer a smaller model with similar performance?**

- Does the smaller model offer easier interpretation and maintenance?
- Is the small performance difference acceptable for the business objective?
- Would the reduced complexity justify using fewer features?

Answer:

## 6. Limitations and Next Experiment

- **Potential limitations:** Feature selection results may depend on the selection method, model configuration, random seed, and training dataset.
- **Potential information loss:** Removing features may eliminate useful signals or interactions between variables.
- **Data leakage concerns:** Feature selection must be performed using training data only, preferably inside a cross-validation or machine learning pipeline. The test set should remain untouched until final evaluation.
- **Next experiment:** Compare multiple feature selection methods, such as SelectKBest, Recursive Feature Elimination (RFE), and model-based feature selection. Evaluate their performance using the same train/test split and model configuration.

## 7. Conclusion

Feature selection helps evaluate whether a smaller set of input variables can achieve performance comparable to a model trained with all available features.

The preferred model should balance predictive performance, training efficiency, interpretability, and business usefulness.

If the selected-feature model achieves similar F1 and ROC-AUC with fewer features and lower training time, it may be preferable because it is simpler to maintain and explain. However, the final decision should be based on actual experimental results rather than the number of features alone.