# Handwritten Digit Recognition (Classic ML)

This project is a classic Machine Learning implementation for classifying handwritten digits from the MNIST dataset. It uses Scikit-learn, Pandas, and two different classification models: **Logistic Regression** and **Support Vector Machine (SVM)**.

This repository corresponds to the "Machine Learning Classification Project" listed on my CV, demonstrating the fundamentals of an ML pipeline.

## 🚀 Project Overview

The script performs the following steps:
1.  **Loads** the MNIST dataset (70,000 images) using `fetch_openml` and `pandas`.
2.  **Prepares** the data by creating a smaller, more manageable sample (5,000 for training, 1,000 for testing).
3.  **Scales** the pixel data using `StandardScaler` for better model performance.
4.  **Trains** a Logistic Regression model on the sample.
5.  **Trains** a Support Vector Machine (SVM) model on the same sample.
6.  **Evaluates** both models and prints their accuracy on the test set.

## 🛠️ Technologies Used

* **Python 3**
* **Scikit-learn (sklearn):** For the ML models (SVM, LogisticRegression), data loading, and processing.
* **Pandas:** For data manipulation (loading the data as a DataFrame).
* **NumPy:** (Used implicitly by Scikit-learn and Pandas)

## 🏁 How to Run

### 1. Prerequisites

Make sure you have Python 3 installed on your system.

### 2. Clone the Repository

```bash
git clone [https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git)
cd YOUR-REPOSITORY-NAME
