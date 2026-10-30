# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree (Udacity)
- Repository: [github.com/NPC1399/predict-customer-chrun](https://github.com/NPC1399/predict-customer-chrun) (GitHub)

---

## Project Description

Credit card companies lose revenue when customers close their accounts. This project
identifies credit card customers who are likely to churn, so the bank can reach out to
them before they leave.

The project refactors an exploratory notebook (`churn_notebook.ipynb`) into a modular,
tested, and logged Python library that follows clean code principles (PEP 8, Pylint).

The machine learning pipeline:

1. **Import data**: loads the bank customer CSV and creates a binary `Churn` target
   from `Attrition_Flag` (1 = attrited customer, 0 = existing customer).
2. **Exploratory data analysis**: saves distribution plots and a correlation heatmap.
3. **Encode categorical features**: replaces each categorical column with the mean churn
   rate of its category (target encoding), e.g. `Gender` → `Gender_Churn`.
4. **Feature engineering**: selects the numeric and encoded features, then performs a
   stratified 70/30 train/test split.
5. **Train and evaluate models**: trains two classifiers, saves them, and saves
   classification reports, feature importances, and ROC curves.

Models used:

- **Random Forest**, tuned with `RandomizedSearchCV` (3-fold cross-validation)
- **Logistic Regression**, in a `Pipeline` with `StandardScaler`

---

## Files and Data Description

```
.
├── churn_library.py                  # Library of pipeline functions
├── churn_script_logging_and_tests.py # Tests and logging
├── churn_notebook.ipynb              # Original exploratory notebook
├── requirements.txt                  # Python dependencies
├── data/
│   └── bank_data.csv                 # Dataset
├── images/
│   ├── eda/                          # EDA plots
│   └── results/                      # Model evaluation plots
├── models/                           # Saved models (.pkl)
└── logs/                             # Test log file
```

### Main Files

- `churn_library.py`  
  The production version of the notebook. It contains one function per pipeline step
  (`import_data`, `perform_eda`, `encoder_helper`, `perform_feature_engineering`,
  `train_models`) plus helper functions for saving plots and reports. Running it executes
  the full pipeline end to end.

- `churn_script_logging_and_tests.py`  
  Unit tests for each main function in `churn_library.py`. Each test receives the function
  under test as an argument, checks its output (dataframe shape, encoded columns,
  train/test splits, saved images and models), and logs the result to
  `logs/churn_library.log`.

- `churn_notebook.ipynb`  
  The original exploratory notebook that the library was refactored from. It is kept
  for reference only and is not needed to run the pipeline.

---

### Data

- `data/bank_data.csv`  
  10,127 credit card customers with 22 columns.

  - **Target**: `Attrition_Flag` (`Existing Customer` / `Attrited Customer`), converted
    to the binary `Churn` column. About 16% of customers churned, so the classes are
    imbalanced, which is why the train/test split is stratified.
  - **Categorical features**: `Gender`, `Education_Level`, `Marital_Status`,
    `Income_Category`, `Card_Category`
  - **Numeric features**: customer demographics (`Customer_Age`, `Dependent_count`),
    relationship with the bank (`Months_on_book`, `Total_Relationship_Count`,
    `Months_Inactive_12_mon`, `Contacts_Count_12_mon`), and credit card usage
    (`Credit_Limit`, `Total_Revolving_Bal`, `Avg_Open_To_Buy`, `Total_Trans_Amt`,
    `Total_Trans_Ct`, `Total_Amt_Chng_Q4_Q1`, `Total_Ct_Chng_Q4_Q1`,
    `Avg_Utilization_Ratio`)
  - **Dropped**: `CLIENTNUM` (customer ID) and the unnamed index column

---

### Output Directories

After running the project, outputs will be saved to:

- EDA images → `images/eda/`
- Model results → `images/results/`
- Models → `models/`
- Logs → `logs/churn_library.log`

The directories are created automatically if they do not exist.

---

## Running the Files

### 0. Install Dependencies

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 1. Run the Pipeline

```bash
python churn_library.py
```

The script loads `data/bank_data.csv`, saves EDA plots to `images/eda/`, encodes the
categorical features, splits the data, trains the Random Forest and Logistic Regression
models, saves both models to `models/`, and saves evaluation plots to `images/results/`.
Training takes a few minutes because of the hyperparameter search.

---

### 2. Run Tests and Logging

```bash
python churn_script_logging_and_tests.py
```

The script tests each main function of `churn_library.py` in pipeline order:

| Test | What it checks |
|---|---|
| `test_import` | The CSV loads and the dataframe is not empty |
| `test_eda` | All five EDA images are saved |
| `test_encoder_helper` | An encoded `<category>_Churn` column exists for each categorical column |
| `test_perform_feature_engineering` | The train and test sets are not empty |
| `test_train_models` | Both models and all four result images are saved |

Each test writes an `INFO` message to `logs/churn_library.log` when it passes and an
`ERROR` message when it fails, then re-raises the error so a failure stops the script.
Check the log file to see the result of each test.

To check code quality:

```bash
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```

---

## Expected Outputs

- Models:
  - `models/rfc_model.pkl`: best Random Forest from the hyperparameter search
  - `models/logistic_model.pkl`: Logistic Regression pipeline (scaler + model)

- EDA images (`images/eda/`):
  - `churn_distribution.png`: churn vs. non-churn counts
  - `age_distribution.png`: customer age histogram
  - `marital_status_distribution.png`: share of each marital status
  - `total_transaction_distribution.png`: total transaction count distribution
  - `correlation_heatmap.png`: correlation between numeric features

- Model results (`images/results/`):
  - `random_forest_classification_report.png`: train and test classification reports
  - `logistic_regression_classification_report.png`: train and test classification reports
  - `feature_importance.png`: Random Forest feature importances
  - `roc_curve_comparison.png`: ROC curves of both models on the test set

- Logs:
  - `logs/churn_library.log`: pass/fail messages from the test script

---

## Notes

- **Logging**: only the test script writes logs, and it appends to
  `logs/churn_library.log` on every run. Delete the file to start a clean log.
- **Plots**: matplotlib uses a non-interactive backend, so the scripts run on servers
  without a display.
- **Target encoding**: category churn rates are computed on the full dataset before the
  train/test split, as in the original notebook. This leaks a small amount of test
  information into the features; in production the encoding should be fit on the
  training set only.
- **Class imbalance**: only about 16% of customers churned, so accuracy alone is
  misleading. Use recall, F1-score, and the ROC curves to compare the models.
- **Reproducibility**: the train/test split, the Random Forest, and the hyperparameter
  search all use `random_state=42`.
