# TODO:
# Add a module-level docstring describing:
# - Purpose of this file
# - Author
# - Date created
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

# TODO:
# configure logging to write INFO and ERROR messages
# to a .log file inside the ./logs directory

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

        # TODO: add logging for success
        logging.info("Data imported successfully.")

    except FileNotFoundError as err:

        # TODO: add logging for file not found
        logging.error("File not found.")

        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    except AssertionError as err:

        # TODO: add logging for failure
        logging.error("Dataframe is empty.")
        raise err


def test_eda(perform_eda):
    """Test EDA."""
    try:
        df = cls.import_data(DATA_PATH)
        perform_eda(df)

        # TODO:
        # assert output files exist
        assert os.path.exists(os.path.join(cls.EDA_DIR, "churn_distribution.png"))  
        # TODO: logging success
        logging.info("EDA performed successfully.")
    except Exception as err:

        # TODO: logging failure
        logging.error("Failed to perform EDA.")

        raise err


def test_encoder_helper(encoder_helper):
    """Test encoding."""
    try:
        df = cls.import_data(DATA_PATH)

        # TODO:
        # create response column
        # call encoder_helper
        # assert encoded columns exist

        df = cls.encoder_helper(df, categorical_columns, "Churn")
        assert all(col in df.columns for col in [f"{col}_Churn" for col in categorical_columns])
        # TODO: logging success
        logging.info("Encoding performed successfully.")
    except Exception as err:

        # TODO: logging failure
        logging.error("Failed to perform encoding.")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """Test feature engineering."""
    try:
        df = cls.import_data(DATA_PATH)

        # TODO:
        # prepare data
        # call function
        # assert outputs
        df = cls.encoder_helper(df, categorical_columns, "Churn")
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(df, "Churn")
        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0

        # TODO: logging success
        logging.info("Feature engineering performed successfully.")
    except Exception as err:

        # TODO: logging failure
        logging.error("Failed to perform feature engineering.")
        raise err


def test_train_models(train_models):
    """Test model training."""
    try:
        df = cls.import_data(DATA_PATH)

        # TODO:
        # prepare data
        # call train_models
        # assert model files + images exist
        df = cls.encoder_helper(df, categorical_columns, "Churn")
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(df, "Churn")
        cls.train_models(x_train, x_test, y_train, y_test)
        # TODO: logging success
        logging.info("Models trained successfully.")

    except Exception as err:

        # TODO: logging failure
        logging.error("Failed to train models.")    
        raise err


if __name__ == "__main__":
    # TODO: ensure logs directory exists

    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)

    print("Tests completed. Check logs for details.")
