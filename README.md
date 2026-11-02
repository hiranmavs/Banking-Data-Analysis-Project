# Banking System Management – Data Analysis using Python

## Objective
Analyse a bank's customers, accounts, transactions and loans to support decisions on
customer targeting, fraud control and credit risk.

## Libraries
| Library | Use |
|---|---|
| pandas / numpy | data generation, cleaning, aggregation |
| matplotlib / seaborn | visualisation (10 charts) |
| scikit-learn | K-Means, Isolation Forest, Random Forest |

## Project Structure
```
bank_project/
├── data_generator.py   # builds 4 synthetic CSV tables
├── analysis.py         # cleaning, EDA, ML, report generation
├── requirements.txt
├── data/               # customers, accounts, transactions, loans
└── outputs/            # charts/, analysis_report.md, flagged_transactions.csv
```

## Methodology
1. **Data generation** – 2,000 customers, ~2,500 accounts, 60,000 transactions, 700 loans (synthetic, seeded).
2. **Data cleaning** – missing channel values imputed by mode; duplicates removed; date features derived.
3. **EDA** – demographics, account mix, monthly volume trend, channel/type usage, correlation heatmap.
4. **Customer segmentation** – K-Means (k=4, chosen via elbow method) on age, income, credit score, balance, activity.
5. **Fraud detection** – Isolation Forest on log-amount and hour of day (unsupervised anomaly detection).
6. **Loan default prediction** – Random Forest classifier; metrics: precision, recall, F1, ROC-AUC (~0.85).

## How to Run
```
pip install -r requirements.txt
python analysis.py
```
To use real data, replace the CSVs in `data/` and skip `build_all()` in `analysis.py`.

## Key Findings (from the sample run)
- UPI is the most common transaction type and Mobile the dominant channel.
- Credit score is the strongest predictor of loan default.
- ~0.3% of transactions are flagged as anomalous for manual review.
- Four customer segments emerge, including a small high-income, high-balance premium group.

# Banking System Management – Analysis Report

## 1. Data Cleaning

Missing 'channel' values filled with mode: 300. Records: customers=2000, accounts=2478, transactions=60000, loans=700.

## 2. Exploratory Data Analysis

Total deposits: ₹205,852,904

Most used channel: Mobile; most common transaction: UPI

City with highest deposits: Kolkata (₹42,518,582)

Dormant/Closed accounts: 18.8%

## 3. Customer Segmentation (K-Means)

|   segment |   age |    annual_income |   credit_score |   total_balance |   txn_count |   customers |
|----------:|------:|-----------------:|---------------:|----------------:|------------:|------------:|
|         0 |    47 | 430115           |            682 |          134567 |          49 |         377 |
|         1 |    61 | 440307           |            682 |           63252 |          24 |         678 |
|         2 |    31 | 467351           |            671 |           68159 |          25 |         737 |
|         3 |    48 |      1.23612e+06 |            683 |          298093 |          32 |         208 |

## 4. Fraud / Anomaly Detection (Isolation Forest)

Flagged 176 suspicious transactions out of 60000. Top 50 saved to outputs/flagged_transactions.csv

## 5. Loan Default Prediction (Random Forest)

Overall default rate: 32.7%

ROC-AUC: 0.857

```
              precision    recall  f1-score   support

           0       0.88      0.86      0.87       118
           1       0.72      0.75      0.74        57

    accuracy                           0.82       175
   macro avg       0.80      0.81      0.80       175
weighted avg       0.83      0.82      0.82       175

```
Most important feature: credit_score
