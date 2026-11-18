# Problem 18 - Model Export & Deployment

## 1. Project Overview

This project demonstrates how to export a trained TensorFlow
model, package preprocessing logic, create an inference pipeline,
and deploy the model using Azure Machine Learning.

## 2. Model Information

- Architecture: MobileNetV2
- Framework: TensorFlow / Keras
- Pretrained weights: ImageNet
- Task: Binary image classification
- Dataset: Cats vs Dogs
- Input size: 160 x 160 RGB

## 3. Model Export

The final fine-tuned model was saved in Keras format.

File: artifacts/model.keras

## 4. Preprocessing

- Decode image
- Convert to RGB
- Resize to 160 x 160
- Apply MobileNetV2 preprocessing

## 5. Local Inference

Run:

python inference.py --image sample_images/cat.jpg

## 6. Local Testing

Run:

python test_inference.py

## 7. Azure Deployment

The model was registered in Azure Machine Learning
and deployed using a Managed Online Endpoint.

## 8. Endpoint Testing

Run:

python create_request.py

python test_endpoint.py

## 9. Results

Record actual local and cloud prediction results here.

## 10. Challenges

Document deployment issues, dependency problems,
preprocessing differences, and their solutions.