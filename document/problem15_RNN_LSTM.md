# **Problem 15 — RNN and LSTM Sentiment Analysis**

## **1. Objective**

The objective of this lab was to build and compare two recurrent neural network models, Simple RNN and LSTM, for sentiment classification.

Using the IMDB movie review dataset, the models classify movie reviews as either Positive or Negative.

This lab helped me understand how recurrent neural networks process sequential text data and how LSTM addresses some limitations of traditional RNNs.

## **2. Dataset and Preprocessing**

**Dataset:** IMDB Movie Reviews Dataset (Keras)

The dataset contains 50,000 movie reviews labeled as positive or negative.

| Dataset    | Samples |
| ---------- | ------- |
| Training   | 20,000  |
| Validation | 5,000   |
| Test       | 25,000  |
| Total      | 50,000  |

Preprocessing configuration:

- Vocabulary size: 10,000 words
- Maximum sequence length: 200
- Reviews were converted into integer sequences.
- Sequences were padded or truncated to a fixed length of 200.
- The prepared dataset was saved as `data/imdb_sequences.npz`.

Data preparation completed successfully.

## **3. Model Architecture**

I implemented two models using TensorFlow/Keras.

**Simple RNN**

- Embedding layer: 64 dimensions
- SimpleRNN layer: 64 units
- Dropout: 0.3
- Dense output layer: 1 unit with sigmoid activation

**LSTM**

- Embedding layer: 64 dimensions
- LSTM layer: 64 units
- Dropout: 0.3
- Dense output layer: 1 unit with sigmoid activation

Both models used binary classification to predict positive or negative sentiment.

## **4. Local Smoke Test Results**

Before submitting the training jobs to Azure ML, I performed local smoke tests to verify that both models could execute successfully.

The tests used:

- 2,000 samples
- 1 training epoch
- A reduced dataset for quick validation

| Metric            | Simple RNN | LSTM     |
| ----------------- | ---------- | -------- |
| Test Loss         | 0.6886     | 0.6934   |
| Test Accuracy     | 53.10%     | 47.65%   |
| Positive-class F1 | 0.4010     | 0.6454   |
| Training Time     | 1.98 sec   | 3.71 sec |
| Epochs            | 1          | 1        |

### **Simple RNN Classification Report**

| Class    | Precision | Recall | F1-score |
| -------- | --------- | ------ | -------- |
| Negative | 0.54      | 0.71   | 0.61     |
| Positive | 0.51      | 0.33   | 0.40     |
| Accuracy |           |        | 0.53     |

The Simple RNN achieved 53.10% test accuracy. It performed better at identifying negative reviews than positive reviews.

### **LSTM Classification Report**

| Class    | Precision | Recall | F1-score |
| -------- | --------- | ------ | -------- |
| Negative | 0.00      | 0.00   | 0.00     |
| Positive | 0.48      | 1.00   | 0.65     |
| Accuracy |           |        | 0.48     |

The LSTM achieved 47.65% test accuracy. Although its positive-class F1-score was higher, the classification report showed that it predicted every test sample as Positive.

This indicates that the model had not learned to distinguish between the two sentiment classes effectively.

**Important:** These are preliminary local smoke-test results, not final model performance. Both models were trained for only one epoch on a small subset of the data.

## **5. Comparison and Observations**

The smoke tests showed that the LSTM took longer to train than the Simple RNN.

The LSTM did not outperform the Simple RNN in this experiment. However, this does not demonstrate that Simple RNN is generally better than LSTM.

Several factors may have affected the results:

1. **Limited training data:** Only 2,000 samples were used instead of the full training dataset.
2. **Insufficient training:** One epoch was not enough to establish whether either model could learn meaningful sentiment patterns.
3. **Model convergence:** The LSTM predicted every test sample as positive, suggesting that it had not learned a useful decision boundary.
4. **Training configuration:** Hyperparameters and model architecture may need further tuning.

The experiment demonstrated that accuracy alone is not sufficient to evaluate a classification model. Precision, recall, F1-score, and the distribution of predicted classes are also important.

## **6. Issues Encountered**

Several issues occurred during implementation and execution.

### **Issue 1: Deprecated Keras Argument**

Keras displayed a warning that `input_length` in the Embedding layer is deprecated.

**Solution:** Remove the `input_length` argument from the Embedding layer.

### **Issue 2: Model Summary Displayed Zero Parameters**

The model summary showed zero parameters and unbuilt layers before the model had been built.

**Observation:** This was related to the model not having received an input shape before calling `summary()`.

**Improvement:** Add an explicit Input layer or build the model before displaying its summary.

### **Issue 3: Comparison Script Could Not Find Metrics**

Running `compare_models.py` resulted in:

`FileNotFoundError: outputs/rnn/metrics.json`

However, the smoke-test results were saved in:

- `outputs/local_rnn_test/metrics.json`
- `outputs/local_lstm_test/metrics.json`

**Solution:** Update the comparison script to use the actual output directories or configure the training scripts to save results to the expected locations.

### **Issue 4: Azure ML Import Error**

The Azure job submission failed because of an incorrect import:

`ImportError: cannot import name 'Output' from 'azure.ai.ml.entities'`

**Solution:** Change the import statement to:

```
from azure.ai.ml import MLClient, command, Outputfrom azure.ai.ml.entities import Environment
```

The Azure ML job was not successfully submitted in this attempt. Therefore, the results in this reflection are from local execution only.

## **7. What I Learned**

Through this lab, I learned several important concepts:

- RNNs are designed to process sequential data and maintain information from previous time steps.
- LSTMs use memory cells and gating mechanisms to better preserve information over longer sequences.
- Text preprocessing, including tokenization, vocabulary selection, and padding, is an important part of NLP model development.
- LSTM models can require more computational resources than simple RNN models.
- A higher F1-score for one class does not necessarily indicate better overall classification performance.
- Model evaluation should include multiple metrics and examination of prediction distributions.

I also gained practical experience preparing a dataset, implementing neural network architectures, running local experiments, organizing model outputs, and troubleshooting Azure ML dependencies.

## **8. Improvements and Next Steps**

To improve the experiment, I would:

1. Fix the Azure ML import issue and successfully submit the training jobs.
2. Train both models using the full training dataset.
3. Increase the number of epochs and monitor validation loss and accuracy.
4. Compare training and validation curves to identify overfitting or underfitting.
5. Tune hyperparameters such as learning rate, batch size, embedding dimension, and hidden units.
6. Evaluate both models using accuracy, precision, recall, F1-score, and confusion matrices.
7. Investigate the LSTM's one-class prediction behavior and adjust the training configuration if necessary.

## **9. Overall Reflection**

This lab helped me understand the practical differences between Simple RNN and LSTM models for sentiment analysis.

One important takeaway was that a more advanced architecture does not automatically produce better results. Model performance depends on data size, training duration, hyperparameters, and whether the model successfully learns patterns from the data.

The local smoke tests were useful for validating the implementation and identifying problems before using cloud resources. Although the initial results were not strong, they provided a useful baseline and highlighted areas that need improvement.

My next step is to complete the Azure ML execution and compare the models using the full dataset and a more appropriate training configuration.

*Source note: The reported dataset sizes, model settings, smoke-test metrics, errors, and output paths above are based on the execution logs you provided. Azure training results are intentionally not reported because the job submission failed.*&#x20;
