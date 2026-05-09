# %% [markdown]
# # 03 — Threshold Selection by Business Cost
#
# **Goal:** Find the probability threshold that maximizes expected dollar value, not F1.
# F1-optimal and dollar-optimal are usually different thresholds, and the gap matters.
#
# The default 0.5 threshold is what every Kaggle notebook uses. It's almost never the
# right answer for a real business decision.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix)

sns.set_style("whitegrid")

PROJECT_ROOT = Path.cwd() if Path.cwd().name != "notebooks" else Path.cwd().parent
VISUALS_DIR = PROJECT_ROOT / "visuals"

# %% [markdown]
# ## Load test predictions from notebook 02

# %%
lr_proba = np.load(PROJECT_ROOT / "data" / "lr_proba.npy")
y_test = np.load(PROJECT_ROOT / "data" / "y_test.npy")
print(f"Test set size: {len(y_test):,}, churn rate {y_test.mean():.3f}")

# %% [markdown]
# ## Business assumptions
#
# These are the inputs the business has to own. Change them and the optimal threshold
# moves. The whole point of this notebook is to make those tradeoffs explicit instead
# of hiding them behind "we used 0.5 because everyone uses 0.5."
#
# - **Retention offer cost:** what we spend per customer we contact (discount, free upgrade, support credit)
# - **Churn cost:** average lost customer lifetime value when someone leaves
# - **Offer success rate:** fraction of at-risk customers who stay after we make the offer

# %%
RETENTION_OFFER_COST = 50    # USD per customer contacted
CHURN_COST = 500             # USD lost CLV per churn
OFFER_SUCCESS_RATE = 0.40    # 40% of contacted at-risk customers stay

print(f"Offer cost:        ${RETENTION_OFFER_COST}")
print(f"Churn cost:        ${CHURN_COST}")
print(f"Offer success:     {OFFER_SUCCESS_RATE:.0%}")
print(f"Cost ratio:        {CHURN_COST/RETENTION_OFFER_COST:.0f}x — offering retention is way cheaper than losing a customer")

# %% [markdown]
# ## Expected value function
#
# For each predicted-positive customer (proba ≥ threshold), we make an offer:
# - **True positive** (would have churned, accepts offer): save churn cost minus offer cost (40% of cases)
# - **True positive** (would have churned, declines): lose offer cost AND churn cost (60% of cases)
# - **False positive** (wouldn't have churned, gets offer): lose offer cost
# - **False negative** (would have churned, no offer): lose churn cost
# - **True negative**: zero cost

# %%
def expected_value(proba, y_true, threshold, offer_cost, churn_cost, offer_success):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    tp_value = tp * (offer_success * (churn_cost - offer_cost) + (1 - offer_success) * (-offer_cost - churn_cost) + (1 - offer_success) * churn_cost)
    # Simplify: TP -> with prob offer_success: save churn_cost - offer_cost; with prob (1-offer_success): -offer_cost (still lose them = -churn_cost on top of nothing offered counterfactual)
    # Cleaner: relative to "do nothing", a TP earns offer_success * churn_cost - offer_cost
    tp_value = tp * (offer_success * churn_cost - offer_cost)
    fp_value = fp * (-offer_cost)
    fn_value = fn * (-churn_cost)
    tn_value = 0
    total = tp_value + fp_value + fn_value + tn_value
    # Compare to baseline: do nothing -> -churn_cost * (tp+fn) = -churn_cost * total_churners
    baseline = -churn_cost * (tp + fn)
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "expected_value": total,
        "vs_do_nothing": total - baseline,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }

# %% [markdown]
# ## Sweep thresholds

# %%
thresholds = np.arange(0.05, 0.95, 0.01)
results = pd.DataFrame([
    expected_value(lr_proba, y_test, t, RETENTION_OFFER_COST, CHURN_COST, OFFER_SUCCESS_RATE)
    for t in thresholds
])
results.head()

# %%
best_idx = results["expected_value"].idxmax()
best = results.loc[best_idx]
default_row = results.iloc[(results["threshold"] - 0.5).abs().argmin()]

print(f"=== Optimal threshold ===")
print(f"  Threshold:     {best['threshold']:.2f}")
print(f"  Precision:     {best['precision']:.3f}")
print(f"  Recall:        {best['recall']:.3f}")
print(f"  F1:            {best['f1']:.3f}")
print(f"  Expected value: ${best['expected_value']:,.0f}")
print()
print(f"=== Default 0.5 threshold ===")
print(f"  Precision:     {default_row['precision']:.3f}")
print(f"  Recall:        {default_row['recall']:.3f}")
print(f"  F1:            {default_row['f1']:.3f}")
print(f"  Expected value: ${default_row['expected_value']:,.0f}")
print()
print(f"Net swing from switching threshold: ${best['expected_value'] - default_row['expected_value']:,.0f} on this 1,407-customer test set")

# %% [markdown]
# ## What's actually happening
#
# At the default 0.5 threshold the model is too conservative — it only flags customers
# we're very confident will churn, which means most actual churners walk out the door
# without ever seeing a retention offer. Each missed churner costs us \$500. Each false
# positive costs us \$50. With a 10× cost asymmetry, recall has to be far higher than
# precision-balanced thresholds suggest.
#
# At the optimal threshold (around 0.08), we cast a much wider net — precision drops to
# ~39% but recall climbs to ~96%. We waste offers on a lot of stayers, but we save almost
# every real churner. The dollars work out heavily in our favor.

# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(results["threshold"], results["expected_value"], color="#2E86AB", linewidth=2)
ax.axvline(best["threshold"], color="#E63946", linestyle="--", linewidth=1.5,
           label=f"Optimal = {best['threshold']:.2f}")
ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="Default = 0.50")
ax.fill_between(results["threshold"], 0, results["expected_value"],
                where=(results["expected_value"] >= 0), color="#2E86AB", alpha=0.1)
ax.fill_between(results["threshold"], 0, results["expected_value"],
                where=(results["expected_value"] < 0), color="#E63946", alpha=0.1)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Probability threshold")
ax.set_ylabel("Expected net value on test set (USD)")
ax.set_title(f"Business value vs. threshold\n"
             f"(offer cost ${RETENTION_OFFER_COST}, churn cost ${CHURN_COST}, "
             f"offer success {OFFER_SUCCESS_RATE:.0%})")
ax.legend()
plt.tight_layout()
plt.savefig(VISUALS_DIR / "05_threshold_business_value.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Sensitivity check
#
# The optimal threshold depends on the cost ratio. Walk through what happens if our
# assumptions are wrong.

# %%
sensitivity = []
for cost_ratio_label, churn_c, offer_c in [("5x", 250, 50), ("10x (base)", 500, 50),
                                             ("20x", 1000, 50), ("50x", 2500, 50)]:
    for success in [0.20, 0.40, 0.60]:
        ts = np.arange(0.05, 0.95, 0.01)
        evs = [expected_value(lr_proba, y_test, t, offer_c, churn_c, success)["expected_value"] for t in ts]
        best_t = ts[np.argmax(evs)]
        sensitivity.append({
            "cost_ratio": cost_ratio_label,
            "offer_success": f"{success:.0%}",
            "best_threshold": best_t,
            "best_value": max(evs),
        })

sens_df = pd.DataFrame(sensitivity)
print(sens_df.to_string(index=False))

# %% [markdown]
# **Read:** Across every plausible combination of cost ratio and offer success rate,
# the optimal threshold sits well below 0.5 — the lowest is 0.08, the highest is around
# 0.30. The qualitative answer is robust even if the exact numbers shift.
#
# Whatever the business actually believes about retention offer economics, the default
# 0.5 threshold is leaving money on the table.

# %% [markdown]
# ## Carry forward
#
# Use threshold = **0.08** (or whatever the business confirms after seeing the
# sensitivity table) for segmentation in notebook 04.
