# Problem 9: Hyperparameter Tuning Reflection

In this experiment, I used GridSearchCV to tune the model's hyperparameters and improve its performance. The best cross-validation F1-score was 0.5848, and the final test F1-score was 0.5801. The grid search took approximately 22.6 seconds, which was reasonable for this experiment.

The tuned model achieved 80.27% accuracy and 83.73% ROC-AUC. However, its F1-score was lower than the baseline model's F1-score of 0.6136. This indicates that hyperparameter tuning did not improve the model's overall performance.

The model achieved 66.67% precision but only 51.34% recall. This means it correctly identified many of its predicted churn customers, but it still missed a significant number of actual churn customers.

One important takeaway is that hyperparameter tuning does not always guarantee better results. The best parameters depend on the dataset, evaluation metric, and model configuration. In future experiments, I would explore a more focused parameter grid, test different tree depths, and consider techniques to improve recall.

Overall, this experiment helped me understand how hyperparameter tuning works and why model evaluation should consider multiple metrics rather than accuracy alone.