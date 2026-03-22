# Improved-AI-Driven-Techniques-for-Precise-Fabric-Classification


![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A machine learning and deep learning project for **fabric classification** using **sparse reflectance map images**. This repository implements and compares **6 different classification programs**, ranging from traditional machine learning models to transfer learning-based convolutional neural networks.


## Overview

Fabric characterization is an important task in computer vision and material recognition. This project explores multiple approaches for recognizing fabric types from image data stored as **sparse reflectance maps**.

The repository includes:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Convolutional Neural Network (CNN)

The goal is to provide a reproducible framework for:

- fabric image classification
- model comparison
- performance evaluation
- experimentation with both classical machine learning and modern deep learning methods


## Dataset Source

The dataset used in this project is publicly available on Kaggle:

[Sparse Reflectance Maps for Fabric Characterization](https://www.kaggle.com/datasets/ritzz08/sparsereflectance-maps-for-fabric-characterization)



## Highlights

- Implements **6 classification programs**
- Supports both **traditional machine learning** and **deep learning**
- Includes **data visualization**, **model evaluation**, and **confusion matrix analysis**
- Uses **transfer learning** with:
  - **VGG16** for feature extraction
  - **MobileNetV2** for CNN classification
- Suitable for:
  - research experiments
  - benchmarking
  - coursework
  - future extension into material recognition pipelines

---

## Important Notes

### 1. Working Directory Path

Please ensure that the path/directory inside **every program cell** is set to:

```python
path = 'SparseReflectance-Maps-for-Fabric-Characterization'
```

### 2. Execution
All notebook cells should be executed sequentially, one at a time.

Running all cells simultaneously is not recommended, as several stages may require substantial memory and computational time, particularly for:

VGG16-based feature extraction
MobileNetV2-based CNN training
large image collections

## Repository Layout
```python
SparseReflectance-Maps-for-Fabric-Characterization/
├── data/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── program1_logistic_regression.ipynb
├── program2_decision_tree.ipynb
├── program3_random_forest.ipynb
├── program4_knn.ipynb
├── program5_svm.ipynb
├── program6_cnn.ipynb
└── README.md
```
The dataset should be organized by class, with each category stored in an individual subdirectory under data/.

## Experimental Methods
Logistic Regression
Used as a linear baseline for supervised fabric classification.

### Decision Tree
A tree-based classifier with standard preprocessing and parameter optimization.

### Random Forest
An ensemble-based approach for improved robustness and comparative evaluation.

### KNN with VGG16 Features
A hybrid pipeline combining deep feature extraction with instance-based classification.

### Support Vector Machine
A margin-based classifier for high-dimensional image representations.

### CNN with MobileNetV2
A transfer learning approach for end-to-end classification using a lightweight convolutional backbone.

## Method Summary
<img width="1264" height="843" alt="image" src="https://github.com/user-attachments/assets/577f7042-1915-4905-87ea-60bee056e1f7" />




## Running the Programs
Before execution, confirm:
```python
path = 'SparseReflectance-Maps-for-Fabric-Characterization'
```
Then run the notebooks individually and sequentially.

Example:
```python
python program1_logistic_regression.py
python program2_decision_tree.py
python program3_random_forest.py
python program4_knn.py
python program5_svm.py
python program6_cnn.py
```
If using Jupyter Notebook, execute each cell in order.

## Evaluation Outputs
The repository is intended to produce the following forms of experimental output:

- classification accuracy
- precision, recall, and F1-score
- confusion matrices
- training and validation curves
- comparative performance across methods

## Future Work
Possible extensions include:

- additional CNN architectures
- data augmentation
- larger-scale dataset evaluation
- explainable AI techniques such as Grad-CAM
- deployment of trained models
