# %% [markdown]
# # 04 - Segments and Recommendations
#
# **Goal:** Turn the model output into something a retention team could use.
# Three things have to land:
# 1. Which features are driving churn predictions
# 2. Where the customers sit on a risk spectrum
# 3. A specific recommendation with numbers attached

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

sns.set_style("whitegrid")
np.random.seed(42)

PROJECT_ROOT = Path.cwd() if Path.cwd().name != "notebooks" else Path.cwd().parent
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"
VISUALS_DIR = PROJECT_ROOT / "visuals"

# %% [markdown]
# ## Re-fit models so we have access to coefficients and importances
#
# (In a real pipeline these would be pickled artifacts from notebook 02.)

# %%
df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)

X = pd.get_dummies(df.drop(columns=["customerID", "Churn", "Churn_bin"]), drop_first=True)
y = df["Churn_bin"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=2000, random_state=42).fit(X_train_s, y_train)
rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10,
                             random_state=42, n_jobs=-1).fit(X_train, y_train)

lr_proba = lr.predict_proba(X_test_s)[:, 1]

# %% [markdown]
# ## 1. Feature importance - what is the model using?

# %%
importances = pd.DataFrame({
    "feature": X.columns,
    "rf_importance": rf.feature_importances_,
    "lr_coef": lr.coef_[0],
}).sort_values("rf_importance", ascending=False)

print("Top 15 by RF importance:")
print(importances.head(15).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(10, 7))
top15 = importances.head(15).iloc[::-1]
colors = ["#E63946" if c > 0 else "#2E86AB" for c in top15["lr_coef"]]
ax.barh(top15["feature"], top15["rf_importance"], color=colors)
ax.set_xlabel("Random Forest feature importance")
ax.set_title("Top 15 features driving churn predictions\n"
             "(red = increases churn, blue = decreases churn - direction from LR)")
plt.tight_layout()
plt.savefig(VISUALS_DIR / "06_feature_importance.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# **What's pulling people out the door:**
# - `InternetService_Fiber optic` - fiber customers churn more, even though they pay more
# - `TotalCharges` - interesting: high *total* charges correlates with churn, while high
#   *monthly* charges interact with tenure (long-tenured customers naturally accumulate higher totals)
# - `StreamingTV`, `StreamingMovies`, `MultipleLines` - add-on services correlate with leavers
# - `PaymentMethod_Electronic check` - this one is consistent across studies of this dataset.
#   Electronic-check payers churn more. The mechanism is debatable (less invested customer?
#   higher friction at billing time?) but the signal is strong.
#
# **What's keeping people:**
# - `tenure` (largest negative coefficient by far) - every additional month on the books
#   reduces churn risk meaningfully
# - `Contract_Two year` and `Contract_One year` - locking customers in works exactly as expected
# - `OnlineSecurity`, `TechSupport` - sticky add-ons. Customers who use these services
#   churn less, probably because they're more integrated into the product

# %% [markdown]
# ## 2. Risk segments
#
# Cut the test set into four risk bands using predicted probability. This is the table
# the retention team could use for prioritization.

# %%
test_df = X_test.copy()
test_df["churn_actual"] = y_test.values
test_df["churn_proba"] = lr_proba
test_df["risk_segment"] = pd.cut(test_df["churn_proba"],
                                   bins=[0, 0.2, 0.5, 0.8, 1.0],
                                   labels=["Low", "Medium", "High", "Critical"])

seg_summary = test_df.groupby("risk_segment", observed=True).agg(
    customers=("churn_actual", "count"),
    actual_churn_rate=("churn_actual", "mean"),
    avg_predicted_proba=("churn_proba", "mean"),
).round(3)
seg_summary["share_of_base"] = (seg_summary["customers"] / seg_summary["customers"].sum()).round(3)
print(seg_summary)

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(seg_summary.index.astype(str), seg_summary["actual_churn_rate"], color="#E63946")
ax.axhline(y_test.mean(), color="black", linestyle="--", linewidth=1,
           label=f"Baseline {y_test.mean():.1%}")
for i, (rate, n) in enumerate(zip(seg_summary["actual_churn_rate"], seg_summary["customers"])):
    ax.text(i, rate + 0.02, f"{rate:.1%}\n(n={n})", ha="center", fontsize=10)
ax.set_title("Actual churn rate by predicted risk segment (test set)")
ax.set_ylabel("Churn rate")
ax.legend()
plt.tight_layout()
plt.savefig(VISUALS_DIR / "07_risk_segments.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. The headline segment
#
# Predicted-probability bands are useful for triage. But a team also needs a
# segment they can describe in one sentence. The model points
# to the same combination over and over: **month-to-month + high charges + low tenure.**

# %%
high_charges_threshold = df["MonthlyCharges"].quantile(0.66)
seg_mask = ((df["Contract"] == "Month-to-month") &
            (df["MonthlyCharges"] >= high_charges_threshold) &
            (df["tenure"] < 12))

seg_size = seg_mask.sum()
seg_share = seg_mask.mean()
seg_churn_rate = df.loc[seg_mask, "Churn_bin"].mean()
non_seg_churn_rate = df.loc[~seg_mask, "Churn_bin"].mean()
seg_churners = df.loc[seg_mask, "Churn_bin"].sum()
total_churners = df["Churn_bin"].sum()

print(f"Segment definition: month-to-month + monthly charges >= ${high_charges_threshold:.2f} + tenure < 12 months")
print(f"Segment size:           {seg_size:,} customers ({seg_share:.1%} of base)")
print(f"Churn rate in segment:  {seg_churn_rate:.1%}")
print(f"Churn rate elsewhere:   {non_seg_churn_rate:.1%}")
print(f"Lift over baseline:     {seg_churn_rate / df['Churn_bin'].mean():.1f}x")
print(f"Share of all churners:  {seg_churners/total_churners:.1%} of total churn comes from {seg_share:.1%} of base")

# %% [markdown]
# ## Recommendation
#
# **Action 1 - Targeted retention offers for the headline segment.** The 5% of customers
# on month-to-month contracts paying \$83+ per month with under a year of tenure churn at
# ~75%. They generate ~14% of all churn. A retention offer - discount, bill credit, or
# contract upgrade incentive - is worth testing here before broad discounting. The 0.08 model threshold should
# trigger this workflow automatically each month.
#
# **Action 2 - Push month-to-month customers toward annual contracts.** Two-year contract
# customers churn at 2.8%, fifteen times less than month-to-month. This is the single
# biggest structural lever in the dataset. Even a small shift from month-to-month
# to annual contracts could matter more than another round of model tuning.
#
# **Action 3 - Investigate fiber-optic experience.** Fiber customers pay more and churn
# more. That's the inverse of what the business should expect from premium service. The
# model can't tell us *why* (price comparison shopping? service quality? competitor
# offers?), but this is where I would ask follow-up questions.
#
# **Action 4 - Do not use the model with a 0.5 threshold by default.** The threshold sweep in
# notebook 03 showed the default loses money on this customer base. The optimal threshold
# moves with retention economics - get the offer cost, churn cost, and success rate
# numbers from the business, then set the threshold there.

# %% [markdown]
# ## Limitations
#
# - Public dataset, not a real customer base. Cost numbers in notebook 03 are illustrative.
# - No price/promotion history. We can see *that* fiber churns more; we can't see whether
#   price-comparison events drive it.
# - Cross-sectional snapshot. A real production system would update predictions monthly
#   and incorporate behavior change over time.
# - "Churn" is binary in the data. Some customers downgrade rather than leave; the model
#   can't distinguish.
