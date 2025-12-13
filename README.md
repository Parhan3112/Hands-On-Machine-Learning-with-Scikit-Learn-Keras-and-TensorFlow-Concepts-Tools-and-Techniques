# Midterm Exam: Deep Learning Projects

## Student Identification
* **Name:** [INSERT YOUR NAME HERE]
* **Class:** [INSERT YOUR CLASS HERE]
* **NIM:** [INSERT YOUR NIM HERE]

---

## Repository Purpose
This repository serves as a submission for the Deep Learning Midterm Exam. It contains two distinct Deep Neural Network (DNN) implementations using **PyTorch**, demonstrating proficiency in handling both **Binary Classification** (imbalanced data) and **Regression** tasks.

The projects focus on the end-to-end machine learning pipeline, including data preprocessing, model architecture design, training loop implementation, and performance evaluation.

---

## Project Overview

### 1. Fraud Detection (Classification)
* **File:** midterm_dl_1.ipynb
* **Objective:** Detect fraudulent transactions in a financial dataset.
* **Challenge:** The dataset contains highly imbalanced classes and a mix of numerical and categorical features requiring robust preprocessing and weighted loss functions.

### 2. Song Release Year Prediction (Regression)
* **File:** midterm_dl_2.ipynb
* **Objective:** Predict the release year of a song based on extracted audio features.
* **Challenge:** Accurately mapping abstract audio features to a continuous time variable (Year).

---

## Models and Matrix Results

### Project 1: Fraud Detection (Binary Classification)

**Model Architecture**
* **Type:** Feed-Forward Neural Network (Multilayer Perceptron).
* **Structure:** 3 Hidden Layers (512 -> 256 -> 64 neurons).
* **Techniques:**
    * **Batch Normalization:** Applied to stabilize training.
    * **Dropout (0.2 - 0.3):** Applied to prevent overfitting.
    * **Weighted Loss:** `BCEWithLogitsLoss` was used with a calculated positive weight to handle class imbalance.

**Evaluation Metrics (10 Epochs)**
| Metric | Result | Analysis |
| :--- | :--- | :--- |
| **Validation ROC-AUC** | **0.8972** | The model distinguishes between fraud and non-fraud transactions with approximately 90% effectiveness. |
| **Training Loss** | **0.7723** | The model showed steady convergence without significant signs of overfitting. |

### Project 2: Song Year Prediction (Regression)

**Model Architecture**
* **Type:** Deep Neural Network (DNN) for Regression.
* **Structure:** 3 Hidden Layers (128 -> 64 -> 32 neurons).
* **Techniques:** ReLU activation functions and Dropout layers (0.2).
* **Optimizer:** Adam (`lr=0.001`).

**Evaluation Metrics (200 Epochs)**
| Metric | Result | Analysis |
| :--- | :--- | :--- |
| **MSE (Mean Squared Error)** | **75.79** | The average squared difference between estimated values and the actual value. |
| **RMSE (Root MSE)** | **8.71** | On average, the prediction deviates by approximately 8.7 years from the actual release year. |
| **MAE (Mean Absolute Error)**| **6.07** | The average absolute difference is around 6 years. |
| **R² Score** | **0.3632** | The model explains approximately 36% of the variance in the dataset. |

---

## How to Navigate

To run these notebooks, it is recommended to use **Google Colab** or a local Jupyter environment with GPU support (e.g., NVIDIA T4, L4, or A100).

### Dependencies
Ensure the following Python libraries are installed:
* torch
* pandas
* numpy
* scikit-learn
* matplotlib

### Execution Steps
1. **midterm_dl_1.ipynb (Fraud Detection):**
    * Mount Google Drive (datasets are expected in `/content/drive/MyDrive/Dataset_MLDL/`).
    * Run the preprocessing cells to handle missing values (`-1` for numeric, `unknown` for categorical) and apply Standard Scaling.
    * Execute the training loop.
    * The notebook generates a `submission.csv` file upon completion.

2. **midterm_dl_2.ipynb (Song Prediction):**
    * Mount Google Drive.
    * Run the cleaning and scaling cells (SimpleImputer and StandardScaler).
    * Execute the training loop for 200 epochs.
    * Review the Loss Plot and final Evaluation Metrics printed at the bottom of the notebook.