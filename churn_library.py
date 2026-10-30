'''
This module contains functions for performing churn prediction analysis.
Author: Phumiphat Charoentananuwat
Date created: 2026-10-24
'''

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report, RocCurveDisplay

EDA_DIR = "./images/eda"
RESULTS_DIR = "./images/results"
MODELS_DIR = "./models"
DATA_PATH = "./data/bank_data.csv"


def create_output_directories():
    """
    Create output directories used by the project.

    input:
            None
    output:
            None
    """
    os.makedirs(EDA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def import_data(pth):
    """
    Return a dataframe for the csv found at pth.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """

    try:
        df = pd.read_csv(pth)

        # Convert the 'Attrition_Flag' column to a binary 'Churn' column
        df["Churn"] = df["Attrition_Flag"].apply(
                lambda val: 0 if val == "Existing Customer" else 1
        )

        return df
    except FileNotFoundError as err:
        print(f"File not found: {pth}")
        raise err


def save_fig(pth):
    """
    Save the current figure to the specified path.

    input:
        pth: a path to save the figure
    output:
        None
    """
    plt.savefig(pth, bbox_inches="tight")
    plt.close()


def perform_eda(df):
    """
    Perform EDA on df and save figures.

    input:
            df: pandas dataframe
    output:
            None
    """
    create_output_directories()

    # Churn distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x="Churn", data=df)
    plt.title("Churn Distribution")
    save_fig(os.path.join(EDA_DIR, "churn_distribution.png"))

    # Age distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df["Customer_Age"], bins=30, kde=True)
    plt.title("Customer Age Distribution")
    save_fig(os.path.join(EDA_DIR, "age_distribution.png"))

    # Marital status distribution
    plt.figure(figsize=(8, 6))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title("Marital Status Distribution")
    save_fig(os.path.join(EDA_DIR, "marital_status_distribution.png"))
    
    # Total transaction distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df["Total_Trans_Ct"], bins=30, kde=True)
    plt.title("Total Transaction Distribution")
    save_fig(os.path.join(EDA_DIR, "total_transaction_distribution.png"))

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    save_fig(os.path.join(EDA_DIR, "correlation_heatmap.png"))


def encoder_helper(df, category_lst, response):
    """
    Encode categorical features.

    input:
            df: pandas dataframe
            category_lst: list of categorical columns
            response: response column name
    output:
            df: updated dataframe
    """

    df = df.copy()
    for category in category_lst:
        # Create a new column for the mean of the response variable for each category
        mean_response = df.groupby(category)[response].mean()
        new_col_name = f"{category}_{response}"
        df[new_col_name] = df[category].map(mean_response)
    return df
    


def perform_feature_engineering(df, response):
    """
    Split dataset into train and test sets.

    input:
              df: pandas dataframe
              response: response column name
    output:
              x_train, x_test, y_train, y_test
    """
    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn',
    ]
    y = df[response]
    X = df[keep_cols]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return x_train, x_test, y_train, y_test


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    Save classification reports as images.

    input:
            predictions and labels
    output:
            None
    """
    create_output_directories()
    
    # Report for Random Forest
    plt.figure(figsize=(8, 6))
    plt.text(0.01, 1.25, "Random Forest Train", fontsize=10, fontweight="bold")
    plt.text(0.01, 0.05, classification_report(y_train, y_train_preds_rf), fontsize=10)
    plt.text(0.01, 0.6, "Random Forest Test", fontsize=10, fontweight="bold")
    plt.text(0.01, 0.7, classification_report(y_test, y_test_preds_rf), fontsize=10)
    plt.axis("off")
    save_fig(os.path.join(RESULTS_DIR, "random_forest_classification_report.png"))

    # Report for Logistic Regression
    plt.figure(figsize=(8, 6))
    plt.text(0.01, 1.25, "Logistic Regression Train", fontsize=10, fontweight="bold")
    plt.text(0.01, 0.05, classification_report(y_train, y_train_preds_lr), fontsize=10)
    plt.text(0.01, 0.6, "Logistic Regression Test", fontsize=10, fontweight="bold")
    plt.text(0.01, 0.7, classification_report(y_test, y_test_preds_lr), fontsize=10)
    plt.axis("off")
    save_fig(os.path.join(RESULTS_DIR, "logistic_regression_classification_report.png"))


def feature_importance_plot(model, x_data, output_pth):
    """
    Save feature importance plot.

    input:
            model, x_data, output path
    output:
            None
    """
    
    # For Random Forest, we can use the feature_importances_ attribute
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [x_data.columns[i] for i in indices]

        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance")
        plt.bar(range(x_data.shape[1]), importances[indices])
        plt.xticks(range(x_data.shape[1]), names, rotation=90)
        save_fig(output_pth)

def roc_curve_plot(lr_model, rf_model, x_test, y_test, output_pth):
    """
    Save ROC curve plot.

    input:
            y_test, y_test_preds_lr, y_test_preds_rf
    output:
            None
    """
    create_output_directories()
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Logistic Regression ROC Curve
    RocCurveDisplay.from_estimator(
        lr_model, 
        x_test, 
        y_test, 
        name="Logistic Regression",
        ax=ax
    )

    # Random Forest ROC Curve
    RocCurveDisplay.from_estimator(
        rf_model, 
        x_test, 
        y_test, 
        name="Random Forest",
        ax=ax
    )
    
    plt.title("ROC Curve Comparison")
    save_fig(os.path.join(RESULTS_DIR, "roc_curve_comparison.png"))


def train_models(x_train, x_test, y_train, y_test):
    """
    Train models and save outputs.

    input:
            train/test data
    output:
            None
    """

    create_output_directories()
    
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = Pipeline([
            ("scaler", StandardScaler()), 
            ("model", LogisticRegression())
        ])

    param_dist = {
        'n_estimators': [200, 300],
        'max_features': ['sqrt'],
        'max_depth': [5, 8, 10],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'criterion': ['gini']
    }

    cv_rfc = RandomizedSearchCV(
        estimator=rfc,
        param_distributions=param_dist,
        n_iter=12,
        cv=3,
        random_state=42,
        n_jobs=-1,
        error_score='raise'
    )

    # Train the models
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    # Save the trained models
    joblib.dump(cv_rfc.best_estimator_, os.path.join(MODELS_DIR, "rfc_model.pkl"))
    joblib.dump(lrc, os.path.join(MODELS_DIR, "logistic_model.pkl"))
    
    y_train_preds_rf = cv_rfc.predict(x_train)
    y_test_preds_rf = cv_rfc.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Generate classification reports
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )

    # Feature importance plot for Random Forest
    feature_importance_plot(
        cv_rfc.best_estimator_, 
        x_train, 
        os.path.join(RESULTS_DIR, "feature_importance.png")
    )

    # ROC Curve Plot
    roc_curve_plot(
        lrc, 
        cv_rfc.best_estimator_, 
        x_test, y_test, 
        os.path.join(RESULTS_DIR, "roc_curve_comparison.png")
    )

if __name__ == "__main__":
    create_output_directories()

    df = import_data(DATA_PATH)

    perform_eda(df)

    category_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]

    df = encoder_helper(df, category_columns, "Churn")
    x_train, x_test, y_train, y_test = perform_feature_engineering(df, "Churn")
    train_models(x_train, x_test, y_train, y_test)
