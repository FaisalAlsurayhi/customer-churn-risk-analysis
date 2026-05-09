# Customer Churn Risk Analysis

> **Business question:** Which customers are most likely to churn, and what should the company do about it?

## TL;DR

A logistic regression on the IBM Telco Customer Churn dataset (~7,000 customers) reaches an **AUC of 0.836**, nearly matching the random forest's 0.841 while staying easier to explain. With realistic retention economics — \$50 to make an offer, \$500 average lost CLV per churn, 40% offer success — the **default 0.5 prediction threshold actually loses money**. The threshold that maximizes expected value sits down at **~0.08**, where recall jumps to 96% at the cost of precision.

The model also points clearly at one segment the retention team should be working: **month-to-month customers paying over \$83/month with under a year of tenure churn at 75% versus a 24% baseline** — 5% of the customer base produces 14% of all churn.

## Visuals

![Churn distribution and contract types](visuals/01_churn_overview.png)

![Churn rate by tenure](visuals/02_churn_by_tenure.png)

![ROC and PR curves](visuals/04_roc_pr_curves.png)

![Business value vs. threshold](visuals/05_threshold_business_value.png)

![Top 15 features](visuals/06_feature_importance.png)

![Risk segments](visuals/07_risk_segments.png)

## Dataset

- **Source:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 rows (7,032 after dropping 11 brand-new customers with no tenure history)
- **Target:** `Churn` (Yes / No), 26.6% baseline rate
- **Features:** 19 customer attributes — contract type, tenure, monthly/total charges, internet service type, payment method, demographic flags, add-on service flags

## Approach

- Cleaned `TotalCharges` (loaded as object due to whitespace strings for tenure-zero customers), one-hot encoded categoricals.
- 80/20 stratified train-test split.
- Trained a logistic regression baseline and a tuned random forest (300 trees, max depth 10, min leaf 10).
- Evaluated with AUC, precision, recall, F1, ROC and PR curves.
- Built a business-cost model (offer cost / churn cost / offer success rate) and swept thresholds to find the dollar-optimal cutoff, not the F1-optimal one.
- Translated probabilities into four risk segments and identified an interpretable headline segment for the retention team.

## Key Findings

- **Logistic regression matches random forest on every metric that matters.** AUC 0.836 vs. 0.841, and LR has higher recall and F1 at the default threshold. Ship the simpler model — there are no non-linearities here that the forest is uniquely catching.
- **Contract type is the single largest signal.** Month-to-month customers churn at 42.7%; two-year contract customers churn at 2.8%. That's a 15× gap, and it dwarfs anything the model adds on top.
- **Tenure is the second-largest.** First-year customers churn at 47.7%; year-four customers churn at 9.5%. Retention spend should be front-loaded.
- **The default 0.5 threshold loses money on this customer base.** With \$50 offers and \$500 churn cost, the dollar-optimal threshold is around 0.08 — the model has to flag aggressively because missing a churner is 10× more expensive than a wasted offer. The sensitivity table in `03_threshold_business_cost.py` shows the optimum stays well below 0.5 across every plausible cost ratio.
- **Headline segment for the retention team:** month-to-month + monthly charges ≥ \$83 + tenure < 12 months. **355 customers (5% of base), 75.2% churn rate, 14.3% of all churn.**
- **Fiber-optic internet is a churn signal, not a stickiness signal.** Fiber customers pay more *and* churn more. The model can't tell us why; the business should investigate.

## Recommendations

1. Run the model monthly with the threshold set by the business's actual retention economics, not 0.5. The threshold sweep in notebook 03 is the artifact to share with finance.
2. Target the headline segment with retention offers before any broad discounting — this is where dollar-per-effort is highest.
3. Push month-to-month customers toward annual contracts. Even modest conversion shifts the company-wide churn number more than any model improvement.
4. Open a separate investigation into fiber-optic customer experience. Premium-priced services should not have above-average churn.

## Limitations

- Public dataset, not a real customer base. The cost numbers in notebook 03 are illustrative and need to be validated against actual retention economics.
- No price or promotion history. We can see *that* fiber churns more; we can't see whether competitor pricing or service incidents are driving it.
- Cross-sectional snapshot. A production system would update predictions monthly as customer behavior evolves.
- Churn is binary in the data. Some customers downgrade rather than leave, and the model can't distinguish the two.

## Reproducibility

```bash
# Clone the repo
git clone https://github.com/FaisalAlsurayhi/customer-churn-risk-analysis.git
cd customer-churn-risk-analysis

# Create virtual environment
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Dataset is included in data/telco_churn.csv
# Run notebooks in order: 01 -> 02 -> 03 -> 04
python notebooks/01_eda.py
python notebooks/02_modeling.py
python notebooks/03_threshold_business_cost.py
python notebooks/04_segments_and_recommendations.py
```

The `.py` files are written in jupyter percent format (`# %%` cells). Open them in VS Code with the Python extension, or convert to `.ipynb` with `jupytext --to ipynb notebooks/*.py`.

---

*Built by Faisal Alsurayhi as part of a data analyst portfolio.*
