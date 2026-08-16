# Feature Selection and Hyperparameter Optimization (Grid Search)

## Objective
Apply feature selection techniques and systematic hyperparameter tuning to improve the performance of classification models.

## Content

### 1. Sequential Feature Selection
- Dataset: [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Using scikit-learn's `SequentialFeatureSelector` to automatically select the most relevant variables

### 2. Grid Search — HR Data (Decision Tree)
- Dataset: [HR Analytics](https://www.kaggle.com/datasets/raminhuseyn/hr-analytics-data-set)
- `GridSearchCV` to find the optimal hyperparameters for a decision tree
- Result: final model with **98% accuracy** and **91% recall** on the minority class

### 3. Grid Search — SVM
- Same HR dataset, comparing performance with **Support Vector Machine (SVM)**

### 4. Fraud Model with Optimization
- Building a fraud detection model prioritizing high recall (capturing the maximum number of frauds) while maintaining a low cost of false positives

## Technologies
`Python` `scikit-learn` `pandas`

## How to Run
```bash
pip install scikit-learn pandas kaggle
jupyter notebook "grid search - dane HR - lh.ipynb"
jupyter notebook "feature selection.ipynb"
