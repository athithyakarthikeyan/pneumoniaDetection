# Pneumonia Detection Using ResNet and Grad-CAM

## Overview
This project implements pneumonia detection using a fine-tuned ResNet-50 model trained on chest X-ray images. Grad-CAM is used to visualize important regions in the image that contribute to the model's prediction. The system includes a simple GUI for easy image uploads and analysis.

## Features
- Pneumonia classification using a ResNet-50 deep learning model.
- Grad-CAM heatmap visualization to highlight critical areas in X-rays.
- Transfer learning to improve accuracy with limited training data.
- Tkinter-based GUI for image upload and real-time predictions.

## Dataset
Kaggle Link : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
The dataset consists of labeled chest X-ray images categorized as:
- **Normal** – Healthy lungs.
- **Pneumonia** – Lungs with signs of infection.

## Installation
pip install torch torchvision numpy opencv-python matplotlib pillow tk

## Clone the Repository
git clone https://github.com/your-username/pneumonia-detection.git
cd pneumonia-detection

## Run the Application
python src.py

