"""Step 2-6: Cleaning, EDA, customer segmentation, fraud detection, loan default prediction."""
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import data_generator as dg
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="viridis")
CH = "outputs/charts/"
report = []


def log(title, text=""):
    print(f"\n=== {title} ===\n{text}")
    report.append(f"## {title}\n\n{text}\n")


def save(name):
    plt.tight_layout()
    plt.savefig(CH + name, dpi=130)
    plt.close()


# ---------- 1. LOAD ----------
customers, accounts, txns, loans = dg.build_all()

# ---------- 2. CLEANING ----------
missing_before = int(txns.channel.isna().sum())
txns["channel"] = txns["channel"].fillna(txns["channel"].mode()[0])
txns = txns.drop_duplicates("txn_id")
txns["month"] = txns.txn_date.dt.to_period("M").astype(str)
txns["hour"] = txns.txn_date.dt.hour
log("1. Data Cleaning", f"Missing 'channel' values filled with mode: {missing_before}. "
    f"Records: customers={len(customers)}, accounts={len(accounts)}, "
    f"transactions={len(txns)}, loans={len(loans)}.")

# ---------- 3. EDA ----------
# 3a. Customer demographics
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
sns.histplot(customers.age, bins=20, kde=True, ax=ax[0]); ax[0].set_title("Customer Age Distribution")
customers.city.value_counts().plot.barh(ax=ax[1], color="teal"); ax[1].set_title("Customers by City")
sns.histplot(np.log10(customers.annual_income), bins=25, ax=ax[2]); ax[2].set_title("Annual Income (log10)")
save("01_customer_demographics.png")

# 3b. Account overview
merged = accounts.merge(customers, on="customer_id")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
accounts.account_type.value_counts().plot.pie(autopct="%1.1f%%", ax=ax[0]); ax[0].set_ylabel("")
ax[0].set_title("Account Type Share")
sns.boxplot(data=accounts[accounts.balance < accounts.balance.quantile(.95)],
            x="account_type", y="balance", ax=ax[1]); ax[1].set_title("Balance by Account Type")
save("02_accounts_overview.png")

# 3c. Transaction trends
monthly = txns.groupby("month").agg(volume=("amount", "sum"), count=("txn_id", "count")).reset_index()
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(monthly.month, monthly.volume / 1e6, marker="o", color="darkblue")
ax.set_title("Monthly Transaction Volume (Millions)"); ax.set_ylabel("₹ Million")
plt.xticks(rotation=60)
save("03_monthly_volume.png")

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
txns.txn_type.value_counts().plot.bar(ax=ax[0], color="coral"); ax[0].set_title("Transactions by Type")
txns.channel.value_counts().plot.bar(ax=ax[1], color="slateblue"); ax[1].set_title("Transactions by Channel")
save("04_txn_type_channel.png")

# 3d. Correlation heatmap
corr = customers.set_index("customer_id")[["age", "annual_income", "credit_score"]].join(
    accounts.groupby("customer_id").balance.sum().rename("total_balance")).corr(numeric_only=True)
plt.figure(figsize=(6, 5)); sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix"); save("05_correlation.png")

top_city = merged.groupby("city").balance.sum().sort_values(ascending=False)
log("2. Exploratory Data Analysis",
    f"Total deposits: ₹{accounts.balance.sum():,.0f}\n\n"
    f"Most used channel: {txns.channel.mode()[0]}; most common transaction: {txns.txn_type.mode()[0]}\n\n"
    f"City with highest deposits: {top_city.index[0]} (₹{top_city.iloc[0]:,.0f})\n\n"
    f"Dormant/Closed accounts: {(accounts.status != 'Active').mean():.1%}")

# ---------- 4. CUSTOMER SEGMENTATION (K-Means) ----------
cust_feat = customers.set_index("customer_id")[["age", "annual_income", "credit_score"]].join(
    accounts.groupby("customer_id").balance.sum().rename("total_balance")).fillna(0)
cust_feat["txn_count"] = txns.merge(accounts[["account_id", "customer_id"]], on="account_id") \
    .groupby("customer_id").size()
cust_feat = cust_feat.fillna(0)
X = StandardScaler().fit_transform(cust_feat)
inertia = [KMeans(k, n_init=10, random_state=0).fit(X).inertia_ for k in range(1, 9)]
plt.figure(figsize=(6, 4)); plt.plot(range(1, 9), inertia, "o-"); plt.title("Elbow Method")
plt.xlabel("k"); plt.ylabel("Inertia"); save("06_elbow.png")

cust_feat["segment"] = KMeans(4, n_init=10, random_state=0).fit_predict(X)
seg = cust_feat.groupby("segment").mean().round(0)
seg["customers"] = cust_feat.segment.value_counts().sort_index()
plt.figure(figsize=(8, 5.5))
sns.scatterplot(data=cust_feat, x="annual_income", y="total_balance", hue="segment", palette="Set1", alpha=.6)
plt.xscale("log"); plt.yscale("symlog"); plt.title("Customer Segments (K-Means, k=4)")
save("07_segments.png")
log("3. Customer Segmentation (K-Means)", seg.to_markdown())

# ---------- 5. FRAUD / ANOMALY DETECTION ----------
t = txns.copy()
t["log_amt"] = np.log1p(t.amount)
iso = IsolationForest(contamination=0.003, random_state=0)
t["anomaly"] = iso.fit_predict(t[["log_amt", "hour"]]) == -1
plt.figure(figsize=(9, 5))
sns.scatterplot(data=t.sample(8000, random_state=0), x="hour", y="log_amt", hue="anomaly",
                palette={False: "lightgrey", True: "red"}, alpha=.7)
plt.title("Anomalous Transactions (Isolation Forest)"); save("08_anomalies.png")
flagged = t[t.anomaly].sort_values("amount", ascending=False)
flagged.head(50)[["txn_id", "account_id", "txn_type", "amount", "txn_date", "channel"]] \
    .to_csv("outputs/flagged_transactions.csv", index=False)
log("4. Fraud / Anomaly Detection (Isolation Forest)",
    f"Flagged {len(flagged)} suspicious transactions out of {len(t)}. "
    f"Top 50 saved to outputs/flagged_transactions.csv")

# ---------- 6. LOAN DEFAULT PREDICTION ----------
L = loans.merge(customers, on="customer_id")
L["loan_to_income"] = L.loan_amount / L.annual_income
feats = ["age", "annual_income", "credit_score", "loan_amount", "interest_rate",
         "tenure_months", "loan_to_income"]
Xtr, Xte, ytr, yte = train_test_split(L[feats], L.defaulted, test_size=.25, stratify=L.defaulted, random_state=0)
rf = RandomForestClassifier(300, class_weight="balanced", random_state=0).fit(Xtr, ytr)
auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
imp = pd.Series(rf.feature_importances_, index=feats).sort_values()
plt.figure(figsize=(7, 4.5)); imp.plot.barh(color="darkgreen"); plt.title("Loan Default – Feature Importance")
save("09_feature_importance.png")
rate = L.groupby("loan_type").defaulted.mean().sort_values()
plt.figure(figsize=(6, 4)); (rate * 100).plot.bar(color="firebrick"); plt.ylabel("Default %")
plt.title("Default Rate by Loan Type"); save("10_default_by_type.png")
log("5. Loan Default Prediction (Random Forest)",
    f"Overall default rate: {L.defaulted.mean():.1%}\n\nROC-AUC: {auc:.3f}\n\n"
    f"```\n{classification_report(yte, rf.predict(Xte))}\n```\n"
    f"Most important feature: {imp.index[-1]}")

# ---------- 7. SAVE REPORT ----------
with open("outputs/analysis_report.md", "w") as f:
    f.write("# Banking System Management – Analysis Report\n\n" + "\n".join(report))
print("\nDone. Charts → outputs/charts/, report → outputs/analysis_report.md")
