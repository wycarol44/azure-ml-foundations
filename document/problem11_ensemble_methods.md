## 11.8 Analysis Questions

1\. Which ensemble achieved the highest F1-score?

Voting achieved the highest F1-score of 0.6157, making it the best overall model in terms of balancing precision and recall.

2\. Which achieved the highest recall?

Voting achieved the highest recall of 0.7433. It identified more actual churn customers than the other models.

3\. Which achieved the highest ROC-AUC?

Gradient Boosting achieved the highest ROC-AUC of 0.8453, indicating the best ability to distinguish between churn and non-churn customers.

4\. Did voting improve on its component models?

Yes. Voting achieved an F1-score of 0.6157, higher than Random Forest (0.6010), Gradient Boosting (0.5667), and Bagging (0.5757). Combining different models helped improve overall F1-score.

5\. Which method required the most training time?

Gradient Boosting required the most training time at 0.7218 seconds. Voting was the fastest at 0.3201 seconds.

6\. Was the improvement worth the additional complexity?

Yes, in this experiment, Voting achieved the highest F1-score and recall with relatively low training time. The improvement seems worthwhile, although the performance difference should also be evaluated on additional data.

7\. Which method would you recommend for a real churn-retention project?

I would recommend Voting because it achieved the highest F1-score and recall. For a churn-retention project, identifying customers who may leave is important so the company can contact them and offer support or incentives. Although Gradient Boosting achieved the highest ROC-AUC, Voting provided a better balance between precision and recall.

## 11.9 Model Performance Comparison

| Model             | Accuracy | Precision | Recall | F1     | ROC-AUC | Training time |
| ----------------- | -------- | --------- | ------ | ------ | ------- | ------------- |
| Voting            | 0.7537   | 0.5255    | 0.7433 | 0.6157 | 0.8385  | 0.3201s       |
| Gradient Boosting | 0.7970   | 0.6538    | 0.5000 | 0.5667 | 0.8453  | 0.7218s       |
| Random Forest     | 0.7615   | 0.5406    | 0.6765 | 0.6010 | 0.8289  | 0.3567s       |
| Bagging           | 0.7970   | 0.6467    | 0.5187 | 0.5757 | 0.8378  | 0.4498s       |

## 11.10 Reflection

In this experiment, I compared four ensemble learning methods: Voting, Gradient Boosting, Random Forest, and Bagging.

The results showed that Voting achieved the highest F1-score of 0.6157 and the highest recall of 0.7433. This means it provided a better balance between precision and recall and identified more actual churn customers.

Gradient Boosting achieved the highest ROC-AUC of 0.8453 and an accuracy of 79.70%. However, its recall was only 50%, meaning it missed many customers who actually churned.

I learned that different ensemble methods have different strengths. Combining multiple models through Voting can improve overall performance, while Gradient Boosting may be better at distinguishing between different classes.

Overall, I would choose Voting for this churn prediction task because identifying potential churn customers is especially important for customer retention. This experiment also showed me that accuracy alone is not enough to evaluate a model. We need to consider multiple metrics based on the business goal.