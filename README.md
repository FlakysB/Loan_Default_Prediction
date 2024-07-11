# Loan Default Prediction Analysis
## Overview
Loan default prediction is a critical problem in the financial industry, as it helps lenders or banks identify borrowers who are
likely to default on their loans. By accurately predicting loan defaults, lenders can adjust loan terms, reserve additional funds to
cover potential losses, or even deny loans to high-risk borrowers.

## Objective
This analysis aims to develop a machine learning-based solution to predict loan defaults based on 
various features such as age, income, loan amount, credit score, employment status, and more.

## Dataset Description
The dataset used in this project consists of 255,347 records and 18 columns. The target feature, Default, 
indicates whether a customer has defaulted on their loan. The objective is to build a machine learning model to predict loan defaults in advance.

## Approach
### Initial Exploratory Data Analysis (EDA)
Initially, all models were run using all columns in the dataset. When the F1 and ROC AUC scores were not satisfactory,
feature importance analysis was conducted to drop columns that did not add predictive significance to the models.

## Model Selection and Hyperparameter Tuning
Five models were initially used to predict the target feature:

Random Forest Classifier (RFC)

Gradient Boosting Classifier (GBC)

Extreme Gradient Boosting Classifier (XGB)

Support Vector Machine (SVM)

K-Nearest Neighbors Classifier (KNC)

The best three models were selected based on their ROC AUC score, F1 score, and accuracy:

Random Forest Classifier (RFC)

Gradient Boosting Classifier (GBC)

Extreme Gradient Boosting Classifier (XGB)

Hyperparameter tuning using Random Search was then performed to optimize these models. However, the performance improvements
were not significant. The best-performing model was further validated using a validation set.

## Findings
Feature Importance: Some features did not contribute significantly to the predictive power of the models and were dropped.
Model Performance: The selected models (RFC, GBC, and XGB) were fine-tuned using hyperparameters, but the performance gains were marginal.
Validation: The best model was validated using a separate validation set to verify its performance. Further fine-tuning of hyperparameters may still be possible to achieve better results.

# Conclusion
The best model was Xgb with the following results
Classification Report (resampled):
              precision    recall  f1-score   support

           0       0.92      0.86      0.89     33855
           1       0.28      0.41      0.33      4448

    accuracy                           0.81     38303
    
   macro avg       0.60      0.64      0.61     38303
   
weighted avg       0.84      0.81      0.82     38303

True Negatives: 29175

False Positives: 4680

False Negatives: 2618

True Positives: 1830

ROC AUC Score: 0.7319983877762655


### Validation set result is:
lassification Report (resampled):
              precision    recall  f1-score   support

           0       0.92      0.86      0.89     33854
           1       0.28      0.41      0.33      4448

    accuracy                           0.81     38302
   macro avg       0.60      0.64      0.61     38302
weighted avg       0.84      0.81      0.82     38302

True Negatives: 29207

False Positives: 4647

False Negatives: 2635

True Positives: 1813

ROC AUC Score: 0.7252125896464845
