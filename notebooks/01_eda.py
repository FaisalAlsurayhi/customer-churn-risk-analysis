# %% [markdown]
# # 01 - Exploratory Data Analysis
#
# **Goal:** Understand who churns and where the signal lives before training anything.
# By the end of this notebook you should be able to answer:
# - What's the baseline churn rate?
# - Which contract types and tenure brackets churn most?
# - Where does the business already have a story before any modeling?

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)

PROJECT_ROOT = Path.cwd() if Path.cwd().name != "notebooks" else Path.cwd().parent
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"
VISUALS_DIR = PROJECT_ROOT / "visuals"
VISUALS_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## Load data

# %%
df = pd.read_csv(DATA_PATH)
print(f"Rows: {len(df):,}")
print(f"Columns: {df.shape[1]}")
df.head()

# %%
# Quick health check
print(df.dtypes)
print(f"\nMissing values: {df.isnull().sum().sum()}")

# %% [markdown]
# ## Cleaning: TotalCharges
#
# `TotalCharges` is loaded as object - it has whitespace strings for brand-new customers
# whose tenure is 0. Convert to numeric and check the damage.

# %%
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(f"NaN TotalCharges after numeric conversion: {df['TotalCharges'].isnull().sum()}")
print(f"All of these have tenure = 0: {(df.loc[df['TotalCharges'].isnull(),'tenure']==0).all()}")

# %%
# 11 brand-new customers, drop them rather than impute. They have no churn history yet.
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)
print(f"After cleaning: {len(df):,} rows")
print(f"Overall churn rate: {df['Churn_bin'].mean():.1%}")

# %% [markdown]
# ## 1. Churn distribution
#
# Baseline first. About 1 in 4 customers churn. That's our reference point - any segment
# above this rate is overrepresented in churn.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

churn_counts = df["Churn"].value_counts()
axes[0].bar(churn_counts.index, churn_counts.values, color=["#2E86AB", "#E63946"])
axes[0].set_title("Churn distribution")
axes[0].set_ylabel("Customers")
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 50, f"{v:,}\n({v/len(df):.1%})", ha="center", fontsize=10)

contract_churn = df.groupby("Contract")["Churn_bin"].mean().sort_values(ascending=False)
axes[1].bar(contract_churn.index, contract_churn.values, color="#E63946")
axes[1].set_title("Churn rate by contract type")
axes[1].set_ylabel("Churn rate")
axes[1].axhline(df["Churn_bin"].mean(), color="black", linestyle="--", linewidth=1,
                label=f"Baseline {df['Churn_bin'].mean():.1%}")
for i, v in enumerate(contract_churn.values):
    axes[1].text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=10)
axes[1].legend()

plt.tight_layout()
plt.savefig(VISUALS_DIR / "01_churn_overview.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Read:** Month-to-month customers churn at 42.7% versus 2.8% on two-year contracts.
# That's a 15x gap. Whatever comes out of the model later, contract type is the single
# biggest lever the business already controls.

# %% [markdown]
# ## 2. Tenure
#
# Customers don't churn uniformly across their lifecycle. Most leave in the first year.

# %%
df["tenure_bracket"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                               labels=["0-12mo", "13-24mo", "25-48mo", "49-72mo"])
tenure_churn = df.groupby("tenure_bracket", observed=True)["Churn_bin"].mean()
print(tenure_churn.round(3))

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(tenure_churn.index.astype(str), tenure_churn.values, color="#E63946")
ax.axhline(df["Churn_bin"].mean(), color="black", linestyle="--", linewidth=1,
           label=f"Baseline {df['Churn_bin'].mean():.1%}")
for i, v in enumerate(tenure_churn.values):
    ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=10)
ax.set_title("Churn rate by tenure bracket")
ax.set_ylabel("Churn rate")
ax.legend()
plt.tight_layout()
plt.savefig(VISUALS_DIR / "02_churn_by_tenure.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Read:** First-year customers churn at 47%, dropping to 9.5% by year four. Retention
# spend should be front-loaded - the first 12 months are where customers decide whether
# they're staying or shopping around.

# %% [markdown]
# ## 3. Monthly charges
#
# Higher bills -> more churn? Not as cleanly as you'd think. The distribution shows
# something more useful.

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(df[df["Churn"] == "No"]["MonthlyCharges"], bins=30, alpha=0.55, label="Stayed", color="#2E86AB")
ax.hist(df[df["Churn"] == "Yes"]["MonthlyCharges"], bins=30, alpha=0.55, label="Churned", color="#E63946")
ax.set_xlabel("Monthly charges (USD)")
ax.set_ylabel("Customers")
ax.set_title("Monthly charges distribution by churn outcome")
ax.legend()
plt.tight_layout()
plt.savefig(VISUALS_DIR / "03_charges_by_churn.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Read:** Stayers cluster at the low end (around \$20-\$30 - the phone-only customers).
# Churners are concentrated in the \$70-\$100 band, which lines up with fiber-optic
# internet customers. The "high charges, less than a year of tenure, month-to-month
# contract" customer is going to come back as the headline segment in the modeling work.

# %% [markdown]
# ## What we walk into modeling with
#
# - 7,032 customers, 26.6% baseline churn
# - Contract type is the dominant signal: 42.7% (MtM) vs 2.8% (2-year)
# - First-year customers are the riskiest cohort by a wide margin
# - Charges show a bimodal stay/churn distribution that points toward fiber-optic services
#
# Next: encode features, train logistic regression and random forest, see whether the
# model adds anything on top of what we can already see.
