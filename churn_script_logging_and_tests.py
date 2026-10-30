'''
This module contains tests for the churn prediction analysis functions.
Author: Phumiphat Charoentananuwat
Date created: 2026-10-24
'''

import logging
import os

import churn_library as cls

LOGS_DIR = "./logs"
LOG_FILE = os.path.join(LOGS_DIR, "churn_library.log")
DATA_PATH = "./data/bank_data.csv"


categorical_columns = [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
]

os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def test_import(import_data):
    """Test data import."""
    try:
        df = import_data(DATA_PATH)
        logging.info("Data imported successfully.")

    except FileNotFoundError as err:
        logging.error("File not found.")

        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    except AssertionError as err:
        logging.error("Dataframe is empty.")
        raise err


def test_eda(perform_eda):
    """Test EDA."""
    try:
        df = cls.import_data(DATA_PATH)
        perform_eda(df)

        assert os.path.exists(os.path.join(cls.EDA_DIR, "churn_distribution.png"))  
        logging.info("EDA performed successfully.")

    except Exception as err:
        logging.error("Failed to perform EDA.")
        raise err


def test_encoder_helper(encoder_helper):
    """Test encoding."""
    try:
        df = cls.import_data(DATA_PATH)
        df = cls.encoder_helper(df, categorical_columns, "Churn")

        assert all(col in df.columns for col in [f"{col}_Churn" for col in categorical_columns])
        logging.info("Encoding performed successfully.")

    except Exception as err:
        logging.error("Failed to perform encoding.")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """Test feature engineering."""
    try:
        df = cls.import_data(DATA_PATH)
        df = cls.encoder_helper(df, categorical_columns, "Churn")

        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(df, "Churn")

        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0

        logging.info("Feature engineering performed successfully.")

    except Exception as err:
        logging.error("Failed to perform feature engineering.")
        raise err


def test_train_models(train_models):
    """Test model training."""
    try:
        df = cls.import_data(DATA_PATH)
        df = cls.encoder_helper(df, categorical_columns, "Churn")

        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(df, "Churn")
        cls.train_models(x_train, x_test, y_train, y_test)

        logging.info("Models trained successfully.")

    except Exception as err:
        logging.error("Failed to train models.")    
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)

    print("Tests completed. Check logs for details.")
