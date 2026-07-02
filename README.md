# COVID-19 Chest X-ray Classification Project

This repository contains an AI for Medicine project for four-class chest X-ray classification using the COVID-19 Radiography Database.

## Classes

- COVID
- Lung Opacity
- Normal
- Viral Pneumonia

## Models

The project compares two transfer-learning models:

- MobileNetV2
- ResNet18

Both models are trained using a two-stage transfer-learning strategy:
1. Train the classifier head with the pretrained backbone frozen.
2. Fine-tune deeper layers with a smaller learning rate.

## Dataset

The dataset is the COVID-19 Radiography Database from Kaggle.

The full image dataset is not uploaded to this repository. The notebooks download the dataset automatically.

## Project status

This project is currently under development.
