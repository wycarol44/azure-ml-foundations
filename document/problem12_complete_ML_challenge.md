# Problem 12 — Complete Advanced ML Challenge

## 1. Objective

The goal of this challenge was to build a complete machine learning workflow using scikit-learn. The workflow integrates data cleaning, feature engineering, feature selection, baseline modeling, hyperparameter tuning, ensemble learning, cross-validation, and final model selection.

## 2. Implementation

I organized the machine learning process into the following stages:

1. **Data Cleaning:** Checked for missing values, duplicate records, incorrect data types, and inconsistent values. Applied appropriate imputation strategies to handle missing data.

2. **Feature Engineering:** Identified relevant features, transformed categorical variables using encoding techniques, and scaled numerical features when necessary.

3. **Baseline Model:** Established a baseline using a simple classification algorithm, such as Logistic Regression, to provide a reference point for evaluating more advanced models.

4. **Feature Selection:** Used feature selection techniques to identify important predictors and reduce unnecessary features. This helped simplify the model and potentially improve generalization.

5. **Pipeline Construction:** Built an sklearn Pipeline to combine preprocessing, feature selection, and model training into one reproducible workflow. This also helped prevent data leakage during cross-validation.

6. **Hyperparameter Tuning:** Applied GridSearchCV to systematically evaluate selected parameter combinations. Used RandomizedSearchCV to explore a broader parameter space more efficiently.

7. **Ensemble Learning:** Evaluated ensemble methods such as Random Forest, Gradient Boosting, and Voting Classifier to determine whether combining multiple models improved predictive performance.

8. **Cross-Validation:** Used cross-validation to evaluate model stability and reduce dependence on a single train-test split.

9. **Final Model Selection:** Compared candidate models using Accuracy, Precision, Recall, F1-score, and ROC-AUC. Selected the final model based on overall performance and the requirements of the classification task.

## 3. Model Evaluation

The following metrics were used to evaluate model performance:

- **Accuracy:** Measures the proportion of correct predictions.
- **Precision:** Measures how many predicted positive cases were actually positive.
- **Recall:** Measures how many actual positive cases were correctly identified.
- **F1-score:** Balances precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between positive and negative classes.

### Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Baseline Model | [Value] | [Value] | [Value] | [Value] | [Value] |
| GridSearchCV Model | [Value] | [Value] | [Value] | [Value] | [Value] |
| RandomizedSearchCV Model | [Value] | [Value] | [Value] | [Value] | [Value] |
| Ensemble Model | [Value] | [Value] | [Value] | [Value] | [Value] |
| Final Selected Model | [Value] | [Value] | [Value] | [Value] | [Value] |

**Best Model:** [Insert model name]

**Selection Reason:** [Explain why this model was selected based on evaluation metrics, cross-validation results, and the project's objectives.]

## 4. Challenges and Lessons Learned

One important challenge was ensuring that preprocessing and feature selection occurred correctly within the cross-validation process. Performing these operations before splitting the data could introduce data leakage and produce overly optimistic evaluation results.

Using an sklearn Pipeline helped address this issue by ensuring that preprocessing and feature selection were fitted separately within each training fold.

Another challenge was balancing model complexity and predictive performance. More complicated models do not always produce better results, and hyperparameter tuning can increase computational cost.

I also learned that Accuracy alone is not sufficient for evaluating classification models, especially when the dataset is imbalanced. Precision, Recall, F1-score, and ROC-AUC provide additional information about model behavior.

## 5. Reflection

This challenge helped me understand how individual machine learning techniques fit together into a complete end-to-end workflow.

Previously, I focused on implementing and evaluating individual models. This project demonstrated the importance of building a structured and reproducible pipeline that handles preprocessing, feature selection, model optimization, and evaluation consistently.

I also learned the practical differences between GridSearchCV and RandomizedSearchCV. GridSearchCV evaluates all specified parameter combinations, while RandomizedSearchCV samples combinations from the parameter distributions and can be more computationally efficient when exploring larger search spaces.

Overall, this challenge improved my understanding of model development, hyperparameter optimization, cross-validation, and evidence-based model selection.