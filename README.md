# Fraud Detection for E-Commerce and Bank Transactions

A machine learning project for **Adey Innovations Inc.** focused on detecting fraudulent e-commerce purchases and bank credit card transactions.

The project explores transaction behavior, feature engineering, imbalanced classification, model evaluation, and explainable machine learning to support better fraud detection decisions.

## Business Problem

Fraud detection involves balancing two important risks:

* **False positives:** legitimate transactions incorrectly flagged as fraudulent, creating customer friction.
* **False negatives:** fraudulent transactions incorrectly classified as legitimate, potentially causing financial losses.

The goal is to evaluate machine learning approaches that distinguish fraudulent from legitimate transactions while considering class imbalance, model performance, and business impact.

## Datasets

The project uses three datasets:

* `Fraud_Data.csv` — e-commerce transactions with fraud labels.
* `IpAddress_to_Country.csv` — IP address ranges mapped to countries for geolocation analysis.
* `creditcard.csv` — bank credit card transactions with fraud labels.

**Data availability:** Dataset files are not tracked in Git. Place the required files in the appropriate local data directories before running the analysis.

## Current Project Structure

```text
fraud_detection_for_ecom_and_bank_transactions/
├── .github/
│   └── workflows/
├── notebooks/
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering-fraudData.ipynb
|   ├── feature-engineering-creditCard.ipynb
│   ├── modeling_creditcard.ipynb
|   ├── modeling_fraud.ipynb
│   └── shap-explainability.ipynb
├── scripts/
├── src/
|   ├── modeling_utils.py
├── results/
|   ├── creditcard_results.csv
|   ├── frau_results.csv
├── tests/
|   ├── test_data_cleaning.py
├── .gitignore
├── README.md
└── requirements.txt
```

*The tree above describes the intended organization; adjust individual filenames and folders to match the current repository.*

## Setup and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Avicci-b/fraud_detection_for_ecom_and_bank_transactions.git
cd fraud_detection_for_ecom_and_bank_transactions
```

### 2. Create and activate a virtual environment

On Linux or WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the datasets

Obtain the required datasets from their authorized sources and place them in the locations expected by the notebooks. Check the notebook paths before running the analysis.

### 5. Run the analysis

Open the project in Jupyter Notebook or VS Code and run the notebooks in the appropriate order, beginning with exploratory data analysis and preprocessing.

## Key Findings — Task 1

### E-commerce Transactions

The dataset has an observed fraud rate of approximately **9.36%**.

* **Device sharing:** `device_shared_count` showed a strong relationship with fraud in the analysis. The observed fraud rate increased substantially for devices associated with many user accounts.
* **Time since signup:** Fraudulent purchases had a median time since signup of approximately one second, compared with approximately 60 days for legitimate purchases.
* **Geolocation:** Some countries showed elevated observed fraud rates relative to the overall baseline. These patterns require contextual investigation and should not be interpreted as evidence that a country's users are inherently fraudulent.
* **Purchase value and age:** These features showed no meaningful difference between fraud and legitimate transactions in the initial analysis.

### Bank Credit Card Transactions

The dataset has an observed fraud prevalence of approximately **0.17%**, making class imbalance a major modeling challenge.

* **Transaction amount:** The amount distribution was strongly right-skewed; a log transformation was explored to reduce skewness.
* **Fraud amount patterns:** Fraudulent transactions showed a different distribution of transaction amounts from legitimate transactions, including a lower median but a higher mean in the initial analysis.
* **PCA-transformed features:** Features V1–V28 are transformed variables, so their individual meanings are not directly interpretable from their names alone.
* **Resampling considerations:** Oversampling can change the class distribution used during training. Synthetic examples must be distinguished from real observations, and evaluation should be performed on data that retains the original distribution.

## Key findings on Task 2: modeling approach
 
- **Models:** Logistic Regression (baseline) and LightGBM (ensemble), per dataset.
- **Imbalance handling:** two routes, compared directly: SMOTE on the training data only, and class weighting on the original training data. The test set is never resampled.
- **Evaluation:** one shared `evaluate_model()` function for every model and both datasets, so numbers are comparable. All results below are on the held-out test set at a 0.5 threshold, except PR-AUC and ROC-AUC, which are threshold-free.
## Results
 
### E-commerce (`Fraud_Data.csv`)
 
| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| LightGBM (class-weight) | 0.7072 | 0.8372 | 0.5434 | 0.6943 | 0.6097 |
| LogReg (class-weight) | 0.6604 | 0.8358 | 0.6193 | 0.6346 | 0.6269 |
| LightGBM (SMOTE)* | 0.7080 | 0.8397 | 0.9740 | 0.5297 | 0.6862 |
| LogReg (SMOTE)* | 0.6307 | 0.8117 | 0.7508 | 0.5781 | 0.6532 |
 
\* Trained on a SMOTE set containing invalid synthetic rows (see Limitations). Kept as a documented experiment, not as primary results.
 
### Credit card (`creditcard.csv`)
 
| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| LightGBM (SMOTE) | 0.8144 | 0.9655 | 0.8000 | 0.8000 | 0.8000 |
| LogReg (SMOTE) | 0.6911 | 0.9635 | 0.1341 | 0.8526 | 0.2318 |
| LightGBM (class-weight, sqrt ratio) | 0.3586 | 0.8678 | 0.4861 | 0.7368 | 0.5858 |
| LightGBM (class-weight, full ratio) | 0.0812 | 0.8579 | 0.1026 | 0.7368 | 0.1802 |
 
## Findings
 
**E-commerce**
- LightGBM leads the clean logistic regression by about 0.05 PR-AUC (0.707 vs 0.660). The lead is moderate, not dominant, and ROC-AUC is essentially tied.
- SMOTE gave LightGBM no PR-AUC benefit over class weighting (0.708 vs 0.707) and hurt Logistic Regression (0.631 vs 0.660). The different precision and recall of the LightGBM arms reflect where the 0.5 threshold lands on each model's probability scale, not different ranking quality.
- **About half of the fraud is invisible to the current features.** Fraud caught by LightGBM has extreme `device_shared_count` and a signup-to-purchase time of seconds. Missed fraud matches legitimate transactions on every feature. The ceiling is feature-limited, not model-limited.
- Logistic Regression assigns almost no weight to `time_since_signup_hours` (a near-zero coefficient) despite it being a strong signal, because fraud sits at a cliff at the low extreme while legitimate users spread across the whole range. A straight line cannot represent that; tree splits can.
**Credit card**
- LightGBM with SMOTE is the clear winner on PR-AUC (0.814): 76 of 95 frauds caught with 19 false alarms.
- ROC-AUC is almost identical for LightGBM and Logistic Regression (0.9655 vs 0.9635), yet at 0.5 the logistic regression raises 523 false alarms to catch 81 frauds. PR-AUC exposes a difference ROC-AUC hides.
- Class weighting performed poorly as configured, and the full-ratio arm is unstable (PR-AUC measured 0.057 in one run and 0.081 in another with identical settings). With only about 380 real fraud rows in training, heavily weighted trees overfit them. SMOTE appears to give the trees a smoother fraud region to learn, despite over 99.8% of the resampled fraud class being synthetic.

## Recommendation
 
| Dataset | Recommended model | Why |
|---|---|---|
| E-commerce | LightGBM, class-weight | Best clean PR-AUC; trained on valid data. Final threshold should be set from the cost of missed fraud versus false alarms (not yet done). |
| Credit card | LightGBM, SMOTE | Best PR-AUC by a wide margin with balanced precision and recall at 0.5. |
## Project Progress

* [x] **Task 1 — Data Analysis and Preprocessing:** Completed
* [x] **Task 2 — Model Building and Training:** Completed
* [ ] **Task 3 — Model Explainability:** Not started

The modeling stage focuses on comparing classification approaches using suitable evaluation metrics for imbalanced data. The explainability stage will investigate important features and individual model predictions.

## Limitations

These issues were found during modeling and are documented rather than hidden.
 
1. **Invalid SMOTE rows in the e-commerce data.** SMOTE was applied directly to one-hot columns and the result was cast back to booleans. About 15,578 resampled rows carry two countries at once, plus 1,269 with two browsers and 474 with two sources, which is impossible for a real transaction. The SMOTE arms inflated dummy-variable coefficients and are not used as primary results. Class weighting is primary for this dataset. A proper fix (SMOTE-NC before one-hot encoding) was not implemented.
2. **Scale mismatch in the saved credit card arrays.** The saved resampled training arrays had `Time`, `hour_of_day` and `Amount_log` standardized while the saved train and test arrays were raw. Models trained on the resampled array and tested on the raw test set produced invalid results: logistic regression predicted no fraud at all (PR-AUC 0.002), and LightGBM scored 0.702 instead of 0.814. The modeling notebook now rebuilds SMOTE from the raw training arrays, and the saved resampled pickle is not used for credit card modeling.
3. **Class weights must be scaled to the imbalance.** The same `scale_pos_weight` strategy that worked at a 1:10 imbalance (e-commerce) failed at 1:600 (credit card). A square-root heuristic was chosen in advance, not tuned on the test set, and it still underperformed.
4. **Single train/test split, no confidence intervals.** With only 95 fraud cases in the credit card test set, one fraud shifts recall by about 1 percentage point. Differences between close models should not be over-read.
5. **Threshold fixed at 0.5 and no hyperparameter tuning.** Operating points are not tuned to a cost model. Any tuning should use a validation split from the training data, never the test set.
6. **Not run:** a class-weight Logistic Regression arm on the credit card data, so the credit card baseline was only trained with SMOTE.

## Next steps
 
- Threshold selection from an explicit cost ratio, on a validation split.
- Bootstrap confidence intervals for PR-AUC.
- Task 3: SHAP explanations for the recommended models.
- E-commerce feature ideas: number of users per IP address, short signup-to-purchase flags, and user-level aggregates.
- Proper SMOTE-NC for the e-commerce data to settle the SMOTE-versus-class-weight question.
- * Use feature importance and SHAP to investigate model behavior.
* Assess the limitations of synthetic oversampling and avoid evaluating on resampled test data.


## Author

**Biniyam Mitiku**
Information Science Student | Addis Ababa University

GitHub: [@Avicci-b](https://github.com/biniyam-mitiku)
