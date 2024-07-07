# Credit Fraud Detection

## Introduction
The Credit Fraud Detection project is a Python-based application designed to identify fraudulent transactions using machine learning techniques. The project preprocesses the data, addresses class imbalances, and applies various classification algorithms to accurately predict fraud.

## How It Works
### Workflow Steps
1. **Data Loading**: Load the credit card transaction dataset.
2. **Data Preprocessing**: Clean and scale the dataset.
3. **Class Imbalance Handling**: Use techniques like SMOTE (Synthetic Minority Over-sampling Technique) and NearMiss to balance the dataset.
4. **Model Training and Evaluation**: Train multiple classification models and evaluate their performance.

## Dependencies and Installation
To install the Credit Fraud Detection project:

1. Clone the repository:
    ```sh
    git clone https://github.com/Shubham246763/Credit-Fraud-Detection.git
    cd Credit-Fraud-Detection
    ```


## Usage
1. **Run the Jupyter Notebook**:
    Open and run `Credit_Fraud_Detection.ipynb` to see the step-by-step implementation of the fraud detection process.

2. **Use the Python Script**:
    - Load and preprocess the dataset:
        ```python
        import pandas as pd
        df = pd.read_csv('creditcard.csv')
        df.fillna(df.mean(), inplace=True)
        from sklearn.preprocessing import RobustScaler
        rob_scaler = RobustScaler()
        df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
        df.drop(['Time', 'Amount'], axis=1, inplace=True)
        df.insert(0, 'scaled_amount', df.pop('scaled_amount'))
        df.insert(1, 'scaled_time', df.pop('scaled_time'))
        ```

    - Train and evaluate classifiers:
        ```python
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report

        X = df.drop('Class', axis=1)
        y = df['Class'].astype(int)
        sss = StratifiedKFold(n_splits=5)

        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            clf = LogisticRegression().fit(X_train, y_train)
            predictions = clf.predict(X_test)
            print(classification_report(y_test, predictions))
        ```

## File Descriptions
- **Credit_Fraud_Detection.ipynb**: Jupyter notebook containing the detailed implementation of the credit fraud detection process.
- **README.md**: This readme file.
- **credit_risk.py**: Python script for preprocessing the data and training the models.

