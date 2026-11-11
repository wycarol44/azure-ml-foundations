# Problem 13 --- ANN Classification Reflection

## 1. Objective

The goal of Problem 13 was to build and evaluate an Artificial Neural
Network (ANN) for binary classification using the IBM Telco Customer
Churn dataset. The target variable was `Churn`, where the model predicts
whether a customer is likely to leave the service.

The experiment focused on: - Preparing numerical and categorical
features for a neural network. - Scaling numerical inputs and encoding
categorical variables. - Training a baseline ANN with
TensorFlow/Keras. - Plotting training and validation loss to monitor
learning and possible overfitting. - Comparing regularization
approaches, including Dropout and L2 regularization. - Evaluating
classification performance with Accuracy, Precision, Recall, F1-score,
ROC-AUC, and a confusion matrix.

## 2. Data Preprocessing

The preprocessing workflow included: 1. Removing `customerID`, because
it is an identifier rather than a useful predictive feature. 2.
Converting `TotalCharges` to a numeric type and handling values that
could not be converted. 3. Separating numerical and categorical
features. 4. Scaling numerical features with `StandardScaler`. 5.
Encoding categorical features with `OneHotEncoder`. 6. Using a
stratified train/test split to preserve the target-class distribution.

Preprocessing transformations should be fitted on the training data only
and then applied to validation and test data. This helps prevent data
leakage.

## 3. ANN Architecture and Training

The baseline ANN used a feed-forward architecture:

  Layer            Configuration
  ---------------- ----------------------------------
  Input            Preprocessed feature vector
  Hidden layer 1   64 neurons
  Hidden layer 2   32 neurons
  Output           1 neuron with sigmoid activation

The model used the Adam optimizer and binary cross-entropy loss. The
output sigmoid produces a probability between 0 and 1 for the positive
class (churn).

The training configuration specified a batch size of 32 and a maximum of
100 epochs, with early stopping using a patience of 10 epochs where
configured. The model was evaluated on held-out test data.

## 4. Model Performance

The following test metrics were reported during the experiment:

  Model                Accuracy   Precision   Recall   F1-score         ROC-AUC
  ------------------ ---------- ----------- -------- ---------- ---------------
  Baseline ANN           0.7932      0.6277   0.5455     0.5837          0.8316
  ANN with Dropout       0.7960      0.6364   0.5428     0.5859          0.8340
  ANN with L2            0.8010      0.6399   0.5749   0.6057\*   Not available

\* The L2 F1-score shown here is calculated from the reported rounded
precision (0.6399) and recall (0.5749). Replace it with the directly
reported F1-score if available. The L2 ROC-AUC was not available in the
recorded results and is intentionally left blank rather than estimated.

### Observations

-   **Baseline ANN:** The baseline achieved an accuracy of 0.7932 and
    ROC-AUC of 0.8316. Its recall of 0.5455 indicates that it identified
    a little over half of the actual churn customers at the selected
    classification threshold.
-   **Dropout:** Dropout slightly improved accuracy, precision,
    F1-score, and ROC-AUC compared with the baseline. Recall decreased
    slightly from 0.5455 to 0.5428. This suggests a small overall
    improvement, but not a substantial change.
-   **L2 regularization:** The reported accuracy, precision, and recall
    were higher than those of the baseline and Dropout models. The
    derived F1-score is approximately 0.6057. The ROC-AUC should be
    added from the original evaluation output before making a complete
    comparison.

These results suggest that regularization can improve generalization,
but its impact should be judged across multiple metrics rather than
accuracy alone.

## 5. Training and Validation Loss

The training and validation loss curves were plotted using the Keras
`history` object.

The curves help diagnose model behavior:

-   If both training and validation loss decrease, the model is learning
    useful patterns.
-   If training loss continues to decrease while validation loss starts
    increasing, the model may be overfitting.
-   If both losses remain high and stop improving, the model may be
    underfitting.
-   If the curves remain relatively close, the model may be generalizing
    more consistently.

The plot should be interpreted together with the final test metrics. A
low training loss by itself does not demonstrate good performance on
unseen data.

## 6. Reflection Questions

### 6.1 Why is feature scaling important for an ANN?

Neural networks are trained using gradient-based optimization. Numerical
features with very different ranges can lead to uneven updates and make
optimization less stable or slower. Standardizing numerical features
helps put them on comparable scales and can make training more
effective.

### 6.2 Why use one-hot encoding for categorical variables?

The ANN expects numerical input. One-hot encoding represents categories
as separate binary features without introducing an artificial ranking
between category labels.

### 6.3 What do the loss curves tell us?

Training loss measures error on the data used to update the model, while
validation loss measures performance on held-out validation data during
training. Comparing them helps identify whether the model is learning
patterns that generalize or beginning to overfit.

### 6.4 Did regularization improve performance?

The reported results show small improvements with Dropout over the
baseline in accuracy, precision, F1-score, and ROC-AUC, although recall
decreased slightly. The L2 model also showed higher reported accuracy,
precision, and recall than the baseline. Based on the available metrics,
L2 appears promising, but its complete comparison requires the original
ROC-AUC and directly reported F1-score.

### 6.5 Which metric is especially important for churn prediction?

Recall is important because a false negative means that a customer who
is actually likely to churn is not identified. Missing these customers
may mean losing an opportunity to offer retention support. Precision
also matters because contacting too many customers who are not going to
churn can waste retention resources. The appropriate balance depends on
the cost of false positives and false negatives.

### 6.6 Would accuracy alone be sufficient?

No. Churn datasets can have imbalanced classes, so a model can achieve
reasonable accuracy while missing many churn customers. Precision,
recall, F1-score, ROC-AUC, and the confusion matrix provide a more
complete view of performance.

### 6.7 How could the model be improved?

Potential next steps include: - Tune the classification threshold based
on the business cost of missed churners and unnecessary retention
outreach. - Tune the number of hidden layers, neurons, learning rate,
and batch size. - Compare regularization strengths for Dropout and L2. -
Use early stopping based on validation loss. - Compare the ANN with
Logistic Regression, Random Forest, and gradient-boosting models using
the same data split and evaluation metrics. - Use cross-validation or
repeated experiments to check whether small metric differences are
consistent.

## 7. Business Interpretation

A churn prediction model can help a telecommunications company identify
customers who may be at risk of leaving and prioritize retention
outreach. The model should not be selected based on accuracy alone. The
business should consider the number of churners detected, the number of
false alarms, the cost of contacting customers, and the value of
retaining a customer.

The classification threshold should be selected according to the
retention team's capacity and the relative cost of false negatives and
false positives. Model predictions should support, rather than
automatically determine, customer-retention decisions.

## 8. Limitations

-   The recorded results do not include the L2 model's ROC-AUC, so that
    comparison is incomplete.
-   The L2 F1-score in the table is calculated from rounded precision
    and recall; the original metric output should be used in the final
    report if available.
-   Small differences between model metrics may be due to random
    initialization or training variation. Repeated runs or
    cross-validation would help assess consistency.
-   The selected classification threshold affects precision and recall.
-   Performance on this test set does not guarantee the same performance
    on future customer populations.

## 9. Conclusion

Problem 13 demonstrated how to build and evaluate an ANN for binary
churn classification. The baseline model provided a useful benchmark,
while Dropout and L2 regularization offered ways to control overfitting
and improve generalization.

Based on the available results, Dropout produced a modest improvement
over the baseline, and the L2 model showed stronger reported accuracy,
precision, and recall. A final model choice should consider recall,
precision, F1-score, ROC-AUC, training behavior, and the business cost
of incorrect predictions---not accuracy alone.

Before submitting, add the L2 ROC-AUC and replace the derived L2
F1-score with the exact value from the notebook output if available.
