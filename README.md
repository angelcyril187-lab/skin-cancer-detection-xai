# Explainable Skin Cancer Detection using XAI

An AI-based skin lesion classification system that uses Deep Learning and Explainable AI (XAI) to classify skin lesions and provide visual and textual explanations for the prediction.

## Project Overview

This project uses a **ResNet50 deep learning model** to classify skin lesions into 7 types.

Unlike a traditional black-box model, the system provides multiple layers of explanation:

- Visual explanation using **Grad-CAM**
- Feature analysis using the **ABCDE rule**
- Natural-language explanation of the prediction

The goal is to make AI-based skin lesion classification more transparent and understandable.

## key Features

-  ResNet50-based skin lesion classification
-  Grad-CAM visual explanations
-  Integrated Gradients for pixel-level attribution
-  ABCDE feature analysis
-  Natural-language AI report
-  High, Medium and Low risk classification
-  Confusion matrix for model evaluation
-  Web-based interface for image analysis

## System Architecture


User
  ↓
Upload Skin Lesion Image
  ↓
Image Preprocessing
  ↓
ResNet50 Model
  ↓
Prediction
  ├── Risk Assessment
  ├── Grad-CAM / Explainability
  └── ABCDE Feature Analysis
          ↓
   Natural Language Report
          ↓
     Visual Dashboard

     Skin-Cancer-Detection-XAI/
│
├── app.js
├── confusion_matrix.png
├── evaluate.py
├── index.html
├── PROJECT_MASTER_GUIDE.md
├── requirements.txt
├── styles.css
└── train.py
