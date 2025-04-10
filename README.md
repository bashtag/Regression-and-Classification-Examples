# Regression-and-Classification-Examples

This repository contains examples of regression and classification tasks using the UCI datasets. In this work, we apply three prominent machine learning techniques—K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Decision Trees (DT)—to two datasets from the UCI repository:

- **Breast Cancer Wisconsin Diagnostic Dataset (ID: 17):** Used for classification.
- **Bike Sharing Dataset (ID: 275):** Used for regression.

The code and experiments presented here were developed as part of Dr. Yakup GENÇ's CSE455 Machine Learning Course.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Techniques and Implementation Details](#techniques-and-implementation-details)
  - [KNN for Classification](#knn-for-classification)
  - [KNN for Regression](#knn-for-regression)
  - [SVM for Classification](#svm-for-classification)
  - [SVM for Regression](#svm-for-regression)
  - [Decision Trees for Classification](#decision-trees-for-classification)
  - [Decision Trees for Regression](#decision-trees-for-regression)
- [Evaluation and Comparison](#evaluation-and-comparison)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

This repository demonstrates different approaches to regression and classification using popular machine learning techniques. The experiments are divided into several parts:
- **KNN-based models:** Implemented from scratch for both classification (using Euclidean distance) and regression (using Manhattan distance).
- **SVM-based models:** Leveraging scikit-learn's `SVC` for classification (with probability estimation, ROC curve, and threshold adjustment) and `SVR` for regression.
- **Decision Tree models:** Utilizing scikit-learn's `DecisionTreeClassifier` and `DecisionTreeRegressor`. In addition, decision rules are extracted to illustrate the interpretability of decision trees.

---

## Datasets

- **Breast Cancer Wisconsin Diagnostic Dataset (ID: 17):** 
  - **Task:** Classification (Malignant vs. Benign)
  - **Label Conversion:** "M" (malignant) is labeled as 1 and "B" (benign) as 0.

- **Bike Sharing Dataset (ID: 275):**
  - **Task:** Regression (Predicting the target variable related to bike sharing)
  - **Feature Engineering:** The date feature is transformed into a numerical value representing days since the start date.

---

## Techniques and Implementation Details

### KNN for Classification

- **Approach:**  
  - Implemented a KNN classifier from scratch using the Euclidean distance.
  - Performed k-fold cross-validation to generate confusion matrices and compute average accuracy.
  
- **Key Results:**  
  - **Average Accuracy:** Approximately 93.44%  
  - **Runtime:** A balance between computation time and accuracy; faster than SVM yet slightly slower than Decision Trees.

---

### KNN for Regression

- **Approach:**  
  - Implemented a KNN regressor from scratch using Manhattan distance.
  - Evaluated performance using Mean Squared Error (MSE) via k-fold cross-validation on both a random sample (5000 points) and the full dataset.
  
- **Key Results:**  
  - **Sample (5000 points):** Average MSE ~14428.0347, Runtime ~103.37 seconds.
  - **Full Dataset:** Average MSE ~7256.6588, with a noticeably increased runtime.
  
- **Observation:**  
  - KNN regression is computationally expensive due to the need to compute distances between each test and all training samples.

---

### SVM for Classification

- **Approach:**  
  - Used scikit-learn's `SVC` with a linear kernel.
  - Added probability estimation for generating ROC curves and applied Fawcett's (Youden's J statistic) method to determine the optimal decision threshold.
  - Evaluated using k-fold cross-validation with confusion matrices and accuracy metrics.
  
- **Key Results:**  
  - **Average Accuracy:** Approximately 95.21%.
  - **Optimal Accuracy (after threshold adjustment):** Improved to ~95.74%.
  - **Runtime:** Approximately 22.57 seconds, which is higher than KNN and Decision Trees.

---

### SVM for Regression

- **Approach:**  
  - Employed scikit-learn's `SVR` with a linear kernel.
  - Evaluated using Mean Squared Error (MSE) through k-fold cross-validation on both a random sample and the full dataset.
  
- **Key Results:**  
  - **Sample (5000 points):** Average MSE ~22333.5858, Runtime ~33.69 seconds.
  - **Full Dataset:** Average MSE ~22041.1762, Runtime ~386.54 seconds.
  
- **Observation:**  
  - SVM regression offers improved efficiency over KNN regression by not requiring storage and computation over the entire dataset for every prediction, though at a slight compromise in accuracy.

---

### Decision Trees for Classification

- **Approach:**  
  - Utilized scikit-learn's `DecisionTreeClassifier` with experiments on different pruning strategies, including:
    - **Pruning Strategy 1:** Limiting `max_depth` to 5.
    - **Pruning Strategy 2:** Using `min_samples_split` of 10.
    - **No Pruning:** For full complexity.
  - Extracted and printed decision rules for interpretability.
  
- **Key Results:**  
  - Comparable accuracies across pruned and unpruned trees.
  - **Runtime:** Extremely fast (in milliseconds), making DT very efficient for classification tasks.
  
- **Observation:**  
  - Pruning effectively reduces overfitting with little loss in accuracy, while also providing a clear, interpretable set of decision rules.

---

### Decision Trees for Regression

- **Approach:**  
  - Applied scikit-learn's `DecisionTreeRegressor` for regression tasks.
  - Evaluated using k-fold cross-validation with MSE as the metric.
  - Also extracted decision rules similar to the classification case.
  
- **Key Results:**  
  - **Performance:** Produced the best MSE scores with significantly lower runtimes (sub-second) compared to SVM regression.
  
- **Observation:**  
  - Decision Tree regression is not only computationally efficient but also provides interpretable models through rule extraction.

---

## Evaluation and Comparison

- **Accuracy (Classification):**  
  - **KNN:** ~93.44%  
  - **SVM:** ~95.21% (up to 95.74% with optimal threshold)  
  - **Decision Trees:** Comparable accuracy with extremely fast runtimes.

- **Mean Squared Error (Regression):**  
  - **KNN Regression:** Higher MSE with high computational cost (especially on full datasets).
  - **SVM Regression:** Moderate MSE with improved runtime over KNN.
  - **Decision Tree Regression:** Best MSE scores with the fastest runtime, making it ideal for large-scale tasks.

- **Computational Efficiency:**  
  - **Decision Trees:** Fastest among the evaluated methods for both classification and regression.
  - **SVM:** Generally more accurate but with increased computation time.
  - **KNN:** Simple implementation but can be computationally intensive, especially for regression on large datasets.

---

## Dependencies

The code in this repository requires the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- A custom module `ucimlrepo` for fetching UCI datasets

You can install the required packages using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

### Clone the Repository:
```bash
git clone https://github.com/yourusername/Regression-and-Classification-Examples.git
cd Regression-and-Classification-Examples
```

### Fetch the Datasets:
The datasets are automatically fetched using the custom `ucimlrepo` module.

### Run the Code:
Execute the corresponding Python scripts or Jupyter notebooks to run experiments on classification and regression tasks.

### Visualizations:
The experiments generate confusion matrices, ROC curves, and extracted decision rules which can be visualized directly.

---

## Acknowledgments
This work is provided by **Dr. Yakup GENÇ** as part of his **CSE455 Machine Learning Course**.  
Special thanks to all contributors and instructors who have provided guidance and support during the development of these examples.

```
This repository is intended for educational purposes and to facilitate understanding of machine learning techniques in regression and classification tasks.