# Fraud Detection System App - Full Guide


Part 1: How to Use the Fraud Detection App
------------------------------------------

Step 1: Open the Streamlit App
- Run the fraud_detection_App.py file using Streamlit
- Command: streamlit run fraud_detection_App.py
- The app will open in your default browser

Step 2: Explore Dataset
- Scroll down to see the top 10 rows from the dataset
- Useful for understanding the structure and fields

Step 3: Check Summary Statistics
- Displays:
  - Total fraudulent transactions
  - Overall fraud rate in percentage

Step 4: View Sample Predictions
- Shows two sample transactions
- Automatically calculates and shows fraud probabilities

Step 5: Make a Custom Prediction
- Enter values in the form:
  - Step
  - Transaction Type (TRANSFER, CASH_OUT, etc.)
  - Old/New balances for sender and receiver
- Click Predict to view fraud probability

Step 6: Analyze Model Evaluation
- Expand the "Show Model Evaluation" section
- Includes:
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
  - Feature Importance
  - Class Distribution

Part 2: How to Run or Deploy the App
------------------------------------

Step 1: Install Dependencies
- Run: pip install -r requirements.txt

Step 2: Run the App Locally
- Run: streamlit run fraud_detection_App.py

Step 3: Deploy the App (Optional)
- Use platforms like Streamlit Cloud
- Make sure Fraud.csv is in the same folder

Part 3: Workflow Diagram (Code Flow)
------------------------------------

[fraud_detection_App.py]
↓
Streamlit UI Setup
↓
load_data_and_amount_scaler() → Loads and preprocesses CSV data
↓
train_model() → Trains Random Forest model after sampling, scaling, and SMOTE
↓
Model and scalers returned
↓
UI Renders:
  - Dataset preview
  - Fraud metrics
  - Sample predictions
  - Custom prediction form
  - Evaluation plots

Part 4: Function Descriptions
-----------------------------

1. load_data_and_amount_scaler()
- Reads data from Fraud.csv
- Fills missing values
- Scales amount using RobustScaler
- Adds balanceChangeOrig and balanceChangeDest features
- Encodes type using numerical labels
- Drops unnecessary columns

2. train_model()
- Uses 5% sampled data
- Splits data into train and test sets
- Scales features using StandardScaler
- Applies SMOTE for class balancing
- Trains RandomForestClassifier
- Returns model, scalers, feature names, and test set

3. Sample Prediction Block
- Creates two example transactions
- Transforms and scales them
- Predicts fraud probability using the trained model
- Displays results in a table

4. Custom Input Form
- Uses Streamlit form to accept transaction data
- Calculates balance changes
- Scales and formats features
- Predicts and displays fraud probability

5. Evaluation Visualizations
- Confusion Matrix: True/False Positives and Negatives
- ROC Curve: AUC vs FPR
- Precision-Recall Curve: Important for imbalanced classes
- Feature Importance: Bar chart of top 10 influential features
- Class Distribution: Graph of fraud vs legit transaction count

Part 5: Key Components and Files
--------------------------------

File: fraud_detection_App.py
- Main application logic

File: Fraud.csv
- Dataset used for training and predictions

Libraries Used:
- streamlit
- pandas
- numpy
- sklearn
- imblearn
- matplotlib
- seaborn


