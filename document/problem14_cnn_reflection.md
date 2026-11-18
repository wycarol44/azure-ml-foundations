# Problem 14 — CNN Image Classification Reflection

## 1. Objective

The objective of this lab was to build and evaluate a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset.

The model classifies images into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

This lab helped me understand how CNNs learn visual features and how to evaluate a multi-class classification model.

## 2. Dataset and Approach

I used the CIFAR-10 dataset, which contains 60,000 small RGB images with a resolution of 32 × 32 pixels.

The workflow included:

1. Loading and preparing the CIFAR-10 dataset.
2. Splitting the training data into training and validation sets.
3. Normalizing image pixel values.
4. Building a CNN using convolutional layers, pooling layers, dense layers, and dropout.
5. Training the model and evaluating it on the test dataset.
6. Reviewing precision, recall, and F1-score for each class.

## 3. Evaluation Results

The model achieved **62% test accuracy**.

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Airplane | 0.71 | 0.68 | 0.69 |
| Automobile | 0.78 | 0.77 | 0.77 |
| Bird | 0.60 | 0.22 | 0.32 |
| Cat | 0.42 | 0.39 | 0.41 |
| Deer | 0.49 | 0.57 | 0.53 |
| Dog | 0.47 | 0.62 | 0.54 |
| Frog | 0.71 | 0.73 | 0.72 |
| Horse | 0.60 | 0.75 | 0.67 |
| Ship | 0.78 | 0.75 | 0.77 |
| Truck | 0.71 | 0.74 | 0.72 |
| **Macro Average** | **0.63** | **0.62** | **0.61** |
| **Weighted Average** | **0.63** | **0.62** | **0.61** |

Each class had 1,000 test images, so the dataset was balanced.

## 4. Analysis of Results

### Stronger Performance

The model performed best on automobiles and ships, both achieving an F1-score of 0.77. Frog and truck classification also performed relatively well.

These categories may have more distinctive visual patterns, making them easier for the CNN to recognize.

### Areas for Improvement

The weakest result was for birds, with an F1-score of 0.32. Although bird precision was 0.60, recall was only 0.22. This means the model correctly identified relatively few of the actual bird images.

The model also struggled with cats, deer, and dogs. These categories can have similar shapes, textures, poses, and backgrounds. The low image resolution may make distinguishing these objects more difficult.

The confusion matrix and misclassified images would help identify which categories are most frequently confused.

## 5. What I Learned

- Convolutional layers help models learn visual features such as edges, textures, and shapes.
- Pooling layers reduce feature-map dimensions and computational complexity.
- Normalization helps prepare image data for more stable training.
- Dropout can reduce overfitting by preventing excessive dependence on individual neurons.
- Accuracy alone is not enough to evaluate a classification model. Precision, recall, and F1-score provide more detailed insights.
- A balanced dataset helps explain why macro and weighted averages are relatively similar.

## 6. Potential Improvements

If I continued this experiment, I would:

1. Review the confusion matrix and misclassified images.
2. Experiment with stronger data augmentation, such as random cropping and horizontal flipping.
3. Tune the learning rate, batch size, and number of epochs.
4. Try a deeper CNN or a ResNet architecture.
5. Compare training and validation performance to identify overfitting or underfitting.
6. Evaluate improvements using the same held-out test dataset after model selection.

## 7. Overall Reflection

This lab provided practical experience with an end-to-end image classification workflow, from data preparation to CNN development and model evaluation.

The 62% test accuracy indicates that the model learned useful visual patterns, but its performance varied across classes. In particular, the low bird recall demonstrated why evaluating only overall accuracy can hide important weaknesses.

My main takeaway is that model development is an iterative process. Building a working model is only the first step. Analyzing classification errors, understanding model limitations, and making targeted improvements are equally important.
