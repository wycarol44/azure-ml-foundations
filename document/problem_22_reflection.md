# Problem 22 — Transformer Sentiment Analysis
## Results, Reflection, and Analysis

### 1. Experiment overview

This experiment compares a traditional machine-learning sentiment classifier with a pretrained transformer:

- **Baseline:** TF-IDF vectorization + Logistic Regression
- **Transformer:** DistilBERT fine-tuned for binary sentiment classification
- **Dataset:** GLUE SST-2 (Stanford Sentiment Treebank, binary sentiment)
- **Execution device:** CPU
- **Evaluation sample:** 500 sentences

The dataset was loaded successfully. It contains 67,349 training examples and 872 validation examples. For this experiment, the TF-IDF model was trained on a reproducible sample of 20,000 training sentences, and both models were evaluated on the same 500 validation sentences. Using the same test examples makes the model comparison more meaningful.

The first displayed training example was:

```text
Sentence: hide new secretions from the parental units
Label: 0 (Negative)
```

### 2. Model performance

The two models produced the following results:

| Metric | TF-IDF + Logistic Regression | DistilBERT |
|---|---:|---:|
| Accuracy | 0.772 (77.2%) | 0.908 (90.8%) |
| Precision (positive class) | 0.7410 | 0.8996 |
| Recall (positive class) | 0.8659 | 0.9272 |
| F1-score (positive class) | 0.7986 | 0.9132 |
| Number of evaluated examples | 500 | 500 |

DistilBERT outperformed the TF-IDF baseline on all four reported metrics. Its accuracy was **13.6 percentage points higher**, increasing from 77.2% to 90.8%. Its positive-class F1-score increased from 0.7986 to 0.9132, an improvement of 0.1146.

The classification reports also show a more balanced result for DistilBERT across negative and positive sentiment:

| Class and metric | TF-IDF | DistilBERT |
|---|---:|---:|
| Negative precision | 0.82 | 0.92 |
| Negative recall | 0.67 | 0.89 |
| Negative F1-score | 0.74 | 0.90 |
| Positive precision | 0.74 | 0.90 |
| Positive recall | 0.87 | 0.93 |
| Positive F1-score | 0.80 | 0.91 |

The TF-IDF model had relatively high recall for positive examples (0.87), but its negative recall was lower (0.67). This suggests that it missed a larger proportion of negative examples and tended to classify some negative sentences as positive. DistilBERT achieved higher precision and recall for both classes, indicating more consistent classification across the two sentiment categories.

### 3. Prediction disagreement analysis

The models disagreed on **112 of the 500 test sentences**, which represents **22.4%** of the evaluation sample. They therefore agreed on the remaining 388 sentences (77.6%).

A disagreement does not automatically mean that one model is wrong. Comparing each prediction with the ground-truth label is necessary to determine which model was correct.

The inspected examples show that both models sometimes made the correct prediction when the other model did not. For example:

- In *“the film tunes into a grief that could lead a man across centuries,”* the TF-IDF model predicted positive, matching the actual label, while DistilBERT predicted negative.
- In *“fresnadillo's dark and jolting images have a way of plying into your subconscious like the nightmare you had a week ago that won't go away,”* DistilBERT predicted positive, matching the actual label, while TF-IDF predicted negative.
- In *“the longer the movie goes, the worse it gets, but it's actually pretty good in the first few minutes,”* TF-IDF predicted negative, matching the actual label, while DistilBERT predicted positive.

These examples demonstrate that neither model is correct in every case. The full disagreement table is more useful than relying on a few selected examples because it allows each prediction to be checked against the ground truth.

#### Why do the models disagree?

**Context and word relationships.** TF-IDF represents text through word and n-gram frequencies. It can learn useful associations, but it does not build a contextual representation of the complete sentence. DistilBERT uses self-attention to represent words in relation to their surrounding context.

**Negation and contrast.** Phrases such as “not good,” “good at first, but worse later,” and “although the story is weak, the acting is excellent” require the model to interpret how parts of a sentence relate to one another. TF-IDF bigrams can capture some local patterns, but they do not provide the same contextual modeling as a transformer.

**Mixed sentiment and long sentences.** Several SST-2 examples contain both positive and negative descriptions. A sentence may praise the acting while criticizing the plot, or describe a film as disturbing while also recognizing its artistic value. The models may weigh these signals differently.

**Subjective or nuanced language.** Film reviews often use figurative language, irony, and subtle judgments. Even a transformer can be confidently wrong when the intended sentiment is ambiguous or when a sentence contains complex opinions.

#### Confidence is not the same as correctness

Some disagreement examples had high DistilBERT confidence scores even when its prediction did not match the ground-truth label. For example, the sentence about “two ships passing in the night” had a transformer confidence of approximately 0.9996, but the transformer predicted positive when the actual label was negative.

This is an important observation: a model's confidence score is not a guarantee that its prediction is correct. The score reflects the model's confidence in its selected class, and it may be poorly calibrated. Confidence should therefore be evaluated alongside actual prediction accuracy.

### 4. Inference latency and computational efficiency

Both models were evaluated on a CPU. The measured prediction times were:

| Measurement | TF-IDF | DistilBERT |
|---|---:|---:|
| Total prediction time | 0.0442 seconds | 3.369 seconds |
| Number of predictions | 500 | 500 |
| Average latency per sample | 0.088 ms | 6.738 ms |
| Throughput | 11,319.5 samples/second | 148.4 samples/second |

The TF-IDF model completed prediction for 500 sentences in approximately 0.044 seconds, while DistilBERT required approximately 3.369 seconds.

Based on these measurements:

- DistilBERT's average per-sample latency was approximately **76.6 times higher** than TF-IDF's.
- TF-IDF processed approximately **76 times as many samples per second** as DistilBERT in this run.
- Despite its higher computational cost, DistilBERT achieved substantially better classification performance.

The TF-IDF model's training time was approximately **1.001 seconds** on the 20,000-sentence training sample. The transformer was loaded from a pretrained checkpoint, so its original pretraining and fine-tuning costs were not part of this experiment's measured training time. The transformer timing represents inference after model loading and warm-up, not the cost of training a transformer from scratch.

The latency numbers are specific to this environment and implementation. Hardware, batch size, sentence length, software versions, and warm-up can affect the results. In particular, the transformer was run on CPU; GPU inference could improve its throughput. The timing comparison should therefore be interpreted as a measurement of this particular CPU setup rather than a universal performance ratio.

### 5. Overall trade-offs

| Consideration | TF-IDF + Logistic Regression | DistilBERT |
|---|---|---|
| Classification accuracy in this experiment | Lower | Higher |
| Contextual language understanding | Limited | Stronger |
| Training approach | Trained on the sampled dataset | Uses pretrained, fine-tuned weights |
| Inference speed on this CPU | Much faster | Slower |
| Computational requirements | Relatively low | Higher |
| Interpretability | Easier to inspect word features | More difficult to interpret |
| Suitable use cases | Lightweight classification and high-throughput processing | Context-sensitive sentiment analysis |

The experiment illustrates a trade-off between prediction quality and computational efficiency. TF-IDF is lightweight and fast, making it a practical baseline when low latency and low resource consumption are important. DistilBERT is more computationally expensive, but it performed substantially better on this sentiment classification sample.

### 6. Reflection

The most important finding from this lab is that a more sophisticated language representation can improve sentiment classification, but the improvement comes with a measurable computational cost.

The TF-IDF baseline achieved 77.2% accuracy and processed the 500 test sentences very quickly. This demonstrates that traditional machine-learning methods remain useful, especially when the task is relatively simple or when a system must process large volumes of text with limited computing resources.

DistilBERT achieved 90.8% accuracy and a positive-class F1-score of 0.9132. Its stronger performance is consistent with the advantage of contextual representations: the model can use relationships among words rather than relying primarily on their individual frequencies. This is particularly valuable for sentences involving contrast, negation, and nuanced descriptions.

The disagreement analysis was also valuable because aggregate metrics alone do not explain model behavior. The 112 disagreements show that the models interpret a meaningful portion of the test sentences differently. Reviewing examples against their actual labels reveals that the transformer is not infallible and that some TF-IDF predictions are correct when the transformer is wrong. The high confidence of some incorrect transformer predictions also highlights why confidence scores should not be treated as proof of correctness.

From an engineering perspective, the latency results matter. On this CPU, DistilBERT's average latency was approximately 76.6 times higher than TF-IDF's. A production system should therefore consider not only accuracy but also throughput, hardware cost, response-time requirements, and expected traffic. A transformer may be justified for applications where better sentiment understanding has significant value, while TF-IDF may be sufficient for applications where speed and simplicity are more important.

One limitation of this experiment is that the TF-IDF model was trained on 20,000 examples rather than the full 67,349-example training set. The evaluation used 500 examples sampled from the 872-example validation split. The results are therefore useful for a controlled comparison in this lab, but they should not be treated as definitive benchmark scores. A stronger experiment would train TF-IDF on the full training set, evaluate on the complete validation set, repeat latency measurements, and report median as well as average latency.

### 7. Conclusion

In this experiment, DistilBERT outperformed TF-IDF + Logistic Regression on the 500-sentence evaluation sample, achieving 90.8% accuracy compared with 77.2% for TF-IDF. It also achieved higher precision, recall, and F1-score for both sentiment classes.

However, DistilBERT required substantially more inference time on CPU. TF-IDF completed prediction approximately 76 times faster in this run.

The results demonstrate that transformer-based models can provide better contextual sentiment understanding, while traditional TF-IDF models remain attractive for their simplicity, speed, and low computational cost.

**The best model depends on the application's requirements.** If contextual accuracy is the priority, a pretrained transformer is a strong choice. If low latency, high throughput, and resource efficiency are more important, TF-IDF remains a useful and competitive baseline.

---

## Key results at a glance

- **Dataset:** GLUE SST-2
- **Training data available:** 67,349 examples
- **TF-IDF training sample:** 20,000 examples
- **Evaluation sample:** 500 examples
- **TF-IDF accuracy:** 77.2%
- **DistilBERT accuracy:** 90.8%
- **Accuracy improvement:** 13.6 percentage points
- **Prediction disagreements:** 112 (22.4%)
- **TF-IDF average latency:** 0.088 ms/sample
- **DistilBERT average latency:** 6.738 ms/sample
- **CPU inference speed ratio:** TF-IDF was approximately 76.6 times faster
