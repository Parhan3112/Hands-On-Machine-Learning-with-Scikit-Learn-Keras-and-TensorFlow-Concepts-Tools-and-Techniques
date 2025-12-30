# TUGAS  (ENRICHMENT FOR MACHINE LEARNING AND DEEP LEARNING CLASSES) - INDIVIDUAL TASK

## Student Identification
* **Name:** [Muhammad Farhan]
* **Class:** [TK-46-01]
* **NIM:** [1103220187]
* 

# Chapter 3: Classification 📊

**Book:** Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow

Welcome to the notebook for **Chapter 3**. Unlike the previous chapter which focused on Regression (predicting values), this chapter dives deep into **Classification** (predicting classes/categories). The primary dataset used for exploration is the famous **MNIST** dataset (handwritten digits).



---

## 🎯 Chapter Summary & Key Objectives

This notebook covers the end-to-end process of building and evaluating classification systems. The key topics include:

* **Binary Classification:** Distinguishing between two classes (e.g., Is this digit a 5 or not?).
* **Performance Measures:** Going beyond simple accuracy, especially for skewed datasets.
    * Confusion Matrix (TP, TN, FP, FN).
    * Precision, Recall, and F1 Score.
    * ROC Curve & AUC Score.
* **Multiclass Classification:** Handling more than two classes (Digits 0-9).
* **Error Analysis:** Visualizing errors to improve the model.
* **Multilabel & Multioutput Classification:** Predicting multiple classes for a single instance.

---

## 🧠 Theoretical Deep Dive

This notebook provides AI-assisted theoretical explanations for complex concepts found in classification tasks.

### A. The Precision/Recall Trade-off
Why can't we have 100% Precision and 100% Recall simultaneously?
* **Precision:** Focuses on quality. "Of all the items I picked, how many are valid?"
* **Recall:** Focuses on quantity. "Of all valid items existing, how many did I find?"

> **Analogy:** Imagine a basket of apples (Positive) and stones (Negative). To ensure you *only* pick apples (High Precision), you might leave some behind (Low Recall). To ensure you get *every* apple (High Recall), you might accidentally pick up some stones (Low Precision). This balance is controlled by the **Decision Threshold**.

### B. Choosing the Right Curve: ROC vs PR
* **ROC Curve (Receiver Operating Characteristic):** Plots True Positive Rate vs False Positive Rate. Best for balanced datasets.
* **PR Curve (Precision-Recall):** Best used when the **positive class is rare** (e.g., fraud detection) or when False Positives are more critical than False Negatives.



### C. Multiclass Strategies: OvO vs OvR
* **OvR (One-versus-the-Rest):** Trains N classifiers (e.g., "Is this 5 or not?"). Efficient for most algorithms.
* **OvO (One-versus-One):** Trains N*(N-1)/2 classifiers (e.g., "Is this 5 or 3?"). Useful for algorithms that don't scale well with large datasets (like SVM).



---

## 💻 Code Implementation & Workflow

The notebook is structured into logical steps using **Python**, **NumPy**, **Matplotlib**, and **Scikit-Learn**.

### 1. Data Loading & Binary Classification
We load the MNIST dataset using `fetch_openml` and split it into training and testing sets. We start by building a simple **"5-Detector"** using the **SGDClassifier** (Stochastic Gradient Descent).

### 2. Performance Evaluation
We implement robust evaluation metrics beyond simple accuracy:
* **Confusion Matrix:** Visualizing True/False Positives and Negatives.
* **Precision & Recall Calculation:** Using `precision_score` and `recall_score`.
* **ROC Curve Visualization:** Plotting the curve to analyze the classifier's performance at different thresholds.



### 3. Multiclass Classification
We scale up the problem to classify all digits (0-9) using **Support Vector Machine (SVC)**. The notebook demonstrates how Scikit-Learn automatically selects between OvO and OvR strategies.

### 4. Error Analysis
We visualize the **Confusion Matrix** using `matplotlib` to identify patterns in the model's mistakes (e.g., confusing the digit '5' with '3').

---

## 🛠️ Libraries Used

* **NumPy:** For array manipulation.
* **Matplotlib:** For data visualization (digits and charts).
* **Scikit-Learn:**
    * `datasets`: To fetch MNIST.
    * `linear_model`: SGDClassifier.
    * `svm`: SVC.
    * `model_selection`: Cross-validation tools.
    * `metrics`: Confusion matrix, ROC, AUC, etc.

---

### Ready to explore Classification? Open the notebook to begin! 🚀

