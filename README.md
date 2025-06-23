# Fraud Detection System – Code Flow Documentation

Project Overview:
================================

This project builds a web-based fraud detection tool using Streamlit and a Random Forest Classifier. The model is trained on financial transaction data and includes real-time prediction capabilities, model evaluation metrics, and interactive data inputs.
------------------------------
System Flow Diagram (Text Representation)

 ┌──────────────────────┐
 │  Load Libraries      │
 └────────┬─────────────┘
          │
 ┌────────▼─────────────┐
 │  Streamlit Setup     │
 └────────┬─────────────┘
          │
 ┌────────▼──────────────────────────┐
 │ Load + Preprocess Dataset         │
 │ - Handle missing values           │
 │ - Scale amount using RobustScaler │
 │ - Encode categorical features     │
 │ - Engineer new features           │
 └────────┬──────────────────────────┘
          │
 ┌────────▼──────────────────────────────┐
 │ Train Model (Random Forest)           │
 │ - Sample data                         │
 │ - Scale features with StandardScaler  │
 │ - Apply SMOTE for class balancing     │
 │ - Train Random Forest Classifier      │
 └────────┬──────────────────────────────┘
          │
 ┌────────▼──────────────┐
 │  UI Components        │
 │  - Dataset preview    │
 │  - Metrics summary    │
 │  - Sample predictions │
 │  - Custom prediction  │
 │  - Model evaluation   │
 └────────┬──────────────┘
          │
 ┌────────▼──────────────┐
 │   Output Predictions  │
 └───────────────────────┘

 ------------------------------

Code Flow in Structured Pointers
1. Library Imports and UI Setup
Imports required libraries: pandas, numpy, scikit-learn, imblearn, streamlit, matplotlib, seaborn.

Configures the Streamlit app layout and sidebar.

2. Data Loading and Preprocessing
Reads transaction data from Fraud.csv.

Handles missing values in nameDest, oldbalanceDest, and newbalanceDest.

Scales the amount column using RobustScaler.

Encodes type using categorical codes.

Engineers new features:

balanceChangeOrig = oldbalanceOrg - newbalanceOrig

balanceChangeDest = newbalanceDest - oldbalanceDest

Drops unnecessary columns such as nameOrig, nameDest, and original amount.

3. Model Training Pipeline
Uses 5% of the dataset to optimize speed.

Splits the data into training and testing sets.

Scales features using StandardScaler.

Applies SMOTE to balance imbalanced class distribution.

Trains a Random Forest Classifier with controlled depth and estimators.

4. Dataset Preview and Fraud Metrics
Displays the top 10 records from the preprocessed dataset.

Computes:

Total fraudulent transactions.

Overall fraud rate.

5. Sample Predictions
Manually defines two sample transaction scenarios.

Processes and scales the input using previously fitted scalers.

Predicts fraud probability and displays formatted output.

6. Custom User Input Prediction
Interactive form using Streamlit widgets.

Users input transaction details like step, type, balances.

Data is transformed and scaled.

Model returns the predicted fraud probability.

Results are shown along with the structured input.

7. Model Evaluation and Visualization
Optionally displays detailed evaluation via expanders:

Confusion Matrix

ROC Curve

Precision-Recall Curve

Feature Importance (Bar Chart)

Class Distribution (Legit vs Fraud)
