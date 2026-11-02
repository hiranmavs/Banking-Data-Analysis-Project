"""Step 1: Generate a realistic synthetic banking dataset (customers, accounts, transactions, loans)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)


def generate_customers(n=2000):
    cities = ["Kolkata", "Siliguri", "Delhi", "Mumbai", "Bengaluru", "Chennai", "Hyderabad", "Pune"]
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
        "age": rng.integers(18, 75, n),
        "gender": rng.choice(["Male", "Female"], n),
        "city": rng.choice(cities, n, p=[.2, .08, .15, .17, .13, .09, .1, .08]),
        "occupation": rng.choice(["Salaried", "Self-Employed", "Student", "Retired", "Business"], n,
                                 p=[.4, .2, .1, .1, .2]),
        "annual_income": rng.lognormal(13, 0.6, n).round(-2),
        "credit_score": np.clip(rng.normal(680, 80, n), 300, 900).astype(int),
        "join_date": pd.to_datetime("2018-01-01") + pd.to_timedelta(rng.integers(0, 2500, n), unit="D"),
    })
    return df


def generate_accounts(customers):
    rows = []
    for cid, income in zip(customers.customer_id, customers.annual_income):
        for _ in range(rng.choice([1, 2], p=[.75, .25])):
            rows.append({
                "account_id": f"A{len(rows) + 1:06d}",
                "customer_id": cid,
                "account_type": rng.choice(["Savings", "Current", "Fixed Deposit"], p=[.6, .2, .2]),
                "balance": round(abs(rng.normal(income * 0.15, income * 0.1)), 2),
                "status": rng.choice(["Active", "Dormant", "Closed"], p=[.82, .12, .06]),
            })
    return pd.DataFrame(rows)


def generate_transactions(accounts, n=60000):
    acc = accounts.sample(n, replace=True, random_state=1).reset_index(drop=True)
    ttype = rng.choice(["Deposit", "Withdrawal", "Transfer", "Bill Payment", "UPI", "ATM"], n,
                       p=[.15, .12, .15, .13, .35, .10])
    amount = np.round(rng.lognormal(7.2, 1.1, n), 2)
    # inject a few anomalous large transactions (potential fraud)
    idx = rng.choice(n, 150, replace=False)
    amount[idx] = amount[idx] * rng.uniform(15, 40, 150)
    df = pd.DataFrame({
        "txn_id": [f"T{i:07d}" for i in range(1, n + 1)],
        "account_id": acc.account_id,
        "txn_type": ttype,
        "amount": amount,
        "txn_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 700 * 24 * 60, n), unit="m"),
        "channel": rng.choice(["Mobile", "Net Banking", "Branch", "ATM"], n, p=[.45, .25, .15, .15]),
    })
    # add some missing values for the cleaning step
    df.loc[rng.choice(n, 300, replace=False), "channel"] = np.nan
    return df


def generate_loans(customers):
    sub = customers.sample(frac=0.35, random_state=3)
    loan_amt = (sub.annual_income * rng.uniform(0.5, 4, len(sub))).round(-3)
    # default probability depends on credit score and income burden
    p_default = 1 / (1 + np.exp((sub.credit_score - 600) / 40 - (loan_amt / sub.annual_income) * 0.4))
    return pd.DataFrame({
        "loan_id": [f"L{i:05d}" for i in range(1, len(sub) + 1)],
        "customer_id": sub.customer_id.values,
        "loan_type": rng.choice(["Home", "Personal", "Auto", "Education"], len(sub)),
        "loan_amount": loan_amt.values,
        "interest_rate": rng.uniform(7, 16, len(sub)).round(2),
        "tenure_months": rng.choice([12, 24, 36, 60, 120, 240], len(sub)),
        "defaulted": (rng.random(len(sub)) < p_default.values).astype(int),
    })


def build_all(save=True):
    customers = generate_customers()
    accounts = generate_accounts(customers)
    transactions = generate_transactions(accounts)
    loans = generate_loans(customers)
    if save:
        customers.to_csv("data/customers.csv", index=False)
        accounts.to_csv("data/accounts.csv", index=False)
        transactions.to_csv("data/transactions.csv", index=False)
        loans.to_csv("data/loans.csv", index=False)
    return customers, accounts, transactions, loans


if __name__ == "__main__":
    build_all()
    print("Datasets saved in /data")
