# AI-Driven-Toxicity-Prediction


## Overview
This project utilizes machine learning models to predict chemical toxicity based on molecular features. The dataset used is the **Tox21 dataset**, a widely used benchmark dataset for toxicity prediction.

## Features
- **Dataset:** Tox21 dataset from DeepChem
- **Models Used:** Random Forest, XGBoost
- **Model Evaluation:** ROC curves, SHAP analysis
- **Feature Importance Analysis:** SHAP values for model interpretability

## Installation
To run this project, install the required dependencies:
```bash
pip install deepchem rdkit-pypi shap xgboost scikit-learn matplotlib
```

## Procedures
1. **Load Dataset:** Fetch the Tox21 dataset using DeepChem.
2. **Data Preprocessing:** Extract features, labels, and molecular representations.
3. **Train-Test Split:** Split the dataset into training and testing sets.
4. **Model Training:** Train **Random Forest** and **XGBoost** classifiers.
5. **Model Evaluation:** Compute the ROC AUC score and plot ROC curves.
6. **SHAP Analysis:** Use SHAP to explain model predictions and feature importance.

## Graph Analysis
- **ROC Curve:** Evaluates model performance by showing the trade-off between sensitivity and specificity.
- **SHAP Summary Plot:** Displays feature importance and how features influence predictions.

## SHAP Results Interpretation
- The SHAP summary plot highlights the most influential features in predicting toxicity.
- A bar plot of SHAP values provides a ranking of feature importance.
- These insights help in understanding how different molecular features contribute to toxicity predictions.

## Summary
This project leverages AI techniques to predict toxicity levels based on molecular features. The combination of **DeepChem, machine learning models, and SHAP explainability** provides a comprehensive approach to toxicity prediction.

## Future Improvements
- Experiment with additional models such as neural networks.
- Enhance feature engineering for improved accuracy.
- Deploy the model as an API for real-world applications.

## Authors
Nnaemeka G. Anyadike

## License
This project is open-source and available under the MIT License.



