# AI Image Detector – EfficientNet Transfer Learning

## Overview
Binary classification system that detects whether an image is AI-generated or real using EfficientNetB0 transfer learning.

## Features
- Transfer learning with EfficientNetB0
- Fine-tuning
- Early stopping
- Precision, Recall, AUC metrics
- Confusion matrix evaluation
- Modular ML pipeline (train/evaluate separation)

## Dataset Structure
data/
    train/
        real/
        fake/
    val/
        real/
        fake/
    test/
        real/
        fake/

## Training
python src/train.py --data_dir data --epochs 10

## Evaluation
python src/evaluate.py

## Tech Stack
- TensorFlow
- EfficientNet
- Scikit-learn
- Python
