# %% [markdown]
# # 02 - Preprocessing and Modeling
#
# **Goal:** Train a logistic regression baseline and a random forest, then compare
# whether the more complex model is actually worth using.
#
# Logistic regression first because it's interpretable and gives the business a coefficient
# story. Random forest is the cross-check: if it clearly beats LR, there may be
# non-linear patterns worth exploring.

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
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             precision_score, recall_score, f1_score, confusion_matrix)

sns.set_style("whitegrid")
np.random.seed(42)

PROJECT_ROOT = Path.cwd() if Path.cwd().name != "notebooks" else Path.cwd().parent
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"
VISUALS_DIR = PROJECT_ROOT / "visuals"

# %% [markdown]
# ## Load and clean

# %%
df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)

print(f"Rows: {len(df):,}")
print(f"Churn rate: {df['Churn_bin'].mean():.4f}")

# %% [markdown]
# ## Feature matrix
#
# One-hot encode every categorical and drop the first level to avoid collinearity.
# Drop `customerID` (unique key, no signal) and the raw `Churn` strings.

# %%
features_to_drop = ["customerID", "Churn", "Churn_bin"]
X = df.drop(columns=features_to_drop)
y = df["Churn_bin"]

X = pd.get_dummies(X, drop_first=True)
print(f"Feature matrix: {X.shape}")
print(f"Features: {X.columns.tolist()}")

# %% [markdown]
# ## Train/test split
#
# 80/20, stratified on the target so churn rate is preserved in both sets.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape}, churn rate {y_train.mean():.3f}")
print(f"Test:  {X_test.shape}, churn rate {y_test.mean():.3f}")

# %% [markdown]
# ## Logistic regression baseline
#
# Scale features (LR needs it for coefficient comparability and convergence).

# %%
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=2000, random_state=42)
lr.fit(X_train_s, y_train)

lr_proba = lr.predict_proba(X_test_s)[:, 1]
lr_pred = (lr_proba >= 0.5).astype(int)

print("=== Logistic Regression (default 0.5 threshold) ===")
print(f"AUC:       {roc_auc_score(y_test, lr_proba):.4f}")
print(f"Accuracy:  {(lr_pred == y_test).mean():.4f}")
print(f"Precision: {precision_score(y_test, lr_pred):.4f}")
print(f"Recall:    {recall_score(y_test, lr_pred):.4f}")
print(f"F1:        {f1_score(y_test, lr_pred):.4f}")

# %% [markdown]
# ## Random forest
#
# 300 trees, shallow leaves to avoid overfitting on a 5,600-row training set.

# %%
rf = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=10,
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

rf_proba = rf.predict_proba(X_test)[:, 1]
rf_pred = (rf_proba >= 0.5).astype(int)

print("=== Random Forest (default 0.5 threshold) ===")
print(f"AUC:       {roc_auc_score(y_test, rf_proba):.4f}")
print(f"Accuracy:  {(rf_pred == y_test).mean():.4f}")
print(f"Precision: {precision_score(y_test, rf_pred):.4f}")
print(f"Recall:    {recall_score(y_test, rf_pred):.4f}")
print(f"F1:        {f1_score(y_test, rf_pred):.4f}")

# %% [markdown]
# ## Model comparison
#
# AUCs are within 0.005 of each other. The interesting result is that LR has *higher*
# recall and F1 at the default threshold. With this dataset, the simpler model carries
# its weight. The random forest is not finding enough extra signal to change the
# recommendation.
#
# This is where I would push back on choosing the more complex model just because it
# looks more advanced. LR is easier to explain, and the performance tradeoff is tiny.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# ROC curves
ax = axes[0]
for name, proba, color in [("Logistic Regression", lr_proba, "#2E86AB"),
                            ("Random Forest", rf_proba, "#E63946")]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2, color=color)
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves")
ax.legend(loc="lower right")

# PR curves
ax = axes[1]
for name, proba, color in [("Logistic Regression", lr_proba, "#2E86AB"),
                            ("Random Forest", rf_proba, "#E63946")]:
    p, r, _ = precision_recall_curve(y_test, proba)
    ax.plot(r, p, label=name, linewidth=2, color=color)
ax.axhline(y_test.mean(), color="gray", linestyle="--", linewidth=1,
           label=f"Baseline ({y_test.mean():.1%})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-recall curves")
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(VISUALS_DIR / "04_roc_pr_curves.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Verdict
#
# Use logistic regression as the main model. Keep RF as a comparison model during
# retraining; if the gap gets wider later, the customer mix may have changed.
#
# Next notebook: the default 0.5 threshold is almost certainly wrong for this business.
# Find the threshold that maximizes expected dollar value, not F1.

# %% [markdown]
# ## Save predictions for downstream notebooks

# %%
np.save(PROJECT_ROOT / "data" / "lr_proba.npy", lr_proba)
np.save(PROJECT_ROOT / "data" / "rf_proba.npy", rf_proba)
np.save(PROJECT_ROOT / "data" / "y_test.npy", y_test.values)
print("Saved test predictions for use in 03 and 04.")
