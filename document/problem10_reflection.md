## 10.6 Analysis Questions

1\. Which method found the higher CV F1-score?

Randomized Search achieved a higher CV F1-score of 0.6345, compared with Grid Search at 0.5848.

2\. Which method took longer?

Randomized Search took approximately 40.62 seconds, while Grid Search took 22.60 seconds. Therefore, Randomized Search took longer.

3\. Which method achieved better test performance?

Randomized Search achieved a higher test F1-score of 0.6282, compared with Grid Search at 0.5801. It also achieved higher recall, improving the identification of churn customers.

4\. Did randomized search explore more useful parameter combinations?

&#x20;  Yes. Randomized Search found a better combination of hyperparameters, including balanced class weights, unlimited tree depth, and 8 minimum samples per leaf. This resulted in better CV and test F1-scores.

5\. Would you use randomized search for a much larger parameter space?

&#x20;  Yes. Randomized Search is useful for larger parameter spaces because it can explore different combinations without testing every possible combination. It can save computational resources while finding promising parameters.

6\. How would you allocate a limited compute budget?

&#x20;  I would start with Randomized Search to explore a broad range of hyperparameters and identify promising combinations. Then, I would use a smaller Grid Search around the best parameters to fine-tune the model. I would focus on F1-score and recall rather than accuracy alone because identifying actual churn customers is important.

## Performance comparison

| Metric      | Grid Search | Randomized Search |
| ----------- | ----------- | ----------------- |
| CV F1       | 0.5848      | 0.6345            |
| Test F1     | 0.5801      | 0.6282            |
| Accuracy    | 80.27%      | 75.30%            |
| Precision   | 66.67%      | 52.31%            |
| Recall      | 51.34%      | 78.61%            |
| ROC-AUC     | 0.8373      | 0.8417            |
| Search time | 22.60 sec   | 40.62 sec         |

Overall reflection: Randomized Search produced a better F1-score and significantly improved recall, although it took more time and had lower accuracy. For this churn prediction task, I would prefer Randomized Search because missing actual churn customers can be costly.