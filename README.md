# 🫁 Pneumonia Detection from Chest X-Rays

End-to-end medical AI system for detecting pneumonia using deep learning.

## 🔬 Features
- Transfer learning with ResNet50
- Class imbalance handling
- Medical-aware evaluation (AUC, Sensitivity)
- Flask REST API for inference
- Streamlit web UI
- Dockerized & Cloud deployable

## 🧠 Model
- Backbone: ResNet50 (ImageNet pretrained)
- Loss: BCEWithLogitsLoss
- Metrics: AUC, Recall, Sensitivity

## 🏗 Architecture
Streamlit UI → Flask API → Deep Learning Model

## 🚀 Run Locally
```bash
docker-compose up --build
