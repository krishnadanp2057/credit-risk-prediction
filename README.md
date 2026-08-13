# CreditGuard — Explainable Loan Default Risk Prediction

Predicts the probability that a loan applicant will default, using 2M+ real
LendingClub loan records (2007–2018). Built to go beyond a standard sklearn
tutorial: handles severe class imbalance, compares four models honestly
(catching a metric that was misleading on its face), explains every
prediction with SHAP, and ships as an interactive risk-scoring web app.

**[Notebook](notebooks/01_eda_and_modeling.ipynb) · [App source](src/app.py)**

---

## Why this project exists

Most student "credit risk" projects stop at `accuracy_score` on a clean,
balanced toy dataset. This one is scoped the way the problem is actually
framed in industry:

- **Real, messy data** — 2.26M raw LendingClub records, filtered down to
  ~1.35M loans with a known outcome, using only features known *at
  application time* (no post-issuance data leakage).
- **Severe class imbalance** — defaults are ~20% of loans, handled with
  SMOTE rather than ignored.
- **Honest model comparison** — four models evaluated on precision,
  recall, F1, and ROC-AUC, not just accuracy, because a model that
  predicts "no default" for everyone would already score ~80% accuracy
  without learning anything.
- **A caught methodology bug, not a hidden one** — the model with the
  best ROC-AUC turned out to be nearly useless in practice (6.5% recall).
  That finding, and the fix, is documented below and in the notebook.
- **Explainability** — every prediction ships with a SHAP-based
  breakdown of what drove it, not just a single number.
- **A business-impact estimate** — the confusion matrix is translated
  into an approximate dollar figure, connecting the model's errors to
  what they'd actually cost a lender.

---

## Results

| Model | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| Logistic Regression (baseline / L2) | 0.553 | 0.258 | 0.249 | 0.253 |
| Logistic Regression (L1 / Lasso) | 0.702 | 0.065 | 0.523 | 0.115 |
| **Random Forest (selected)** | **0.693** | **0.368** | **0.370** | **0.369** |

**Selected model:** Random Forest (300 trees, max depth 8,
`class_weight='balanced'`), evaluated at the default 0.5 threshold.

**Cross-validated ROC-AUC:** ~0.69 (3-fold, SMOTE re-applied inside each
fold to avoid synthetic-sample leakage across folds — see *Methodology
notes* below), consistent with the held-out test score.

### Why Random Forest, not the higher-AUC model

Logistic Regression with L1 regularization scored a *higher* ROC-AUC
(0.702 vs. 0.693) — but at the standard 0.5 decision threshold, its
recall for actual defaults was only **6.5%**. It ranked applicants
correctly in relative order (hence the decent AUC), but its predicted
probabilities were skewed too low to cross the 0.5 cutoff for most true
positives — meaning it would catch roughly 1 in 15 real defaults in
production. Random Forest, despite a marginally lower AUC, catches
**36.8%** of actual defaults at the same threshold.

This is the central finding of the project: **ROC-AUC measures ranking
ability across all thresholds, but a deployed model uses one fixed
cutoff — a high AUC with collapsed recall at that cutoff is a trap.**
For a lender, a missed default is more costly than a false alarm, so
recall matters more than a marginal AUC difference here.

### Business impact (estimated)

At the selected threshold, the confusion matrix on the test set
translates to approximately:

- **True positives (defaults correctly flagged):** *[fill in from your
  notebook's Section 18 output]*
- **False negatives (defaults missed):** *[fill in]*
- **Estimated capital at risk from missed defaults:** *[fill in —
  `false negatives × average loan amount`]*

This is a rough, order-of-magnitude estimate (average loan amount ×
error counts), not a calibrated financial model — but it's enough to
show the model catching a meaningful share of at-risk capital that a
naive "approve everyone" baseline would miss entirely.

### Top SHAP drivers

*[Fill in the top 3 features from your SHAP summary plot — likely
`grade`/`sub_grade`, `dti`, and `revol_util` based on sanity checks: a
Grade A / 10+ years applicant scored ~0.2% risk vs. a Grade G / <1 year
applicant at ~65%.]*

---

## Methodology notes (for anyone reviewing this)

**Data leakage caught and fixed:** the one-hot encoding step initially
included dummy columns derived from `loan_status` — the same column the
target was built from. Training on these let the model read the answer
instead of learning real risk patterns. They're explicitly dropped
before the train/test split (see notebook Section 6).

**Cross-validation leakage caught and fixed:** an early cross-validation
run scored ROC-AUC ≈ 0.90 — far higher than the honest test-set score of
0.693. The cause: SMOTE had already been applied *before* the CV split,
so synthetic minority-class samples in a validation fold could be
interpolated from real samples sitting in a different training fold.
The fix uses an `imblearn` `Pipeline` so SMOTE is refit fresh inside each
fold, using only that fold's training data (notebook Section 15).

**Dropped columns:** `issue_d` (loan issue date) was one-hot encoded into
~200 near-useless month/year dummy columns and dropped as noise rather
than signal.

---

## Project structure

```
credit-risk-prediction/
├── data/                       # raw dataset (gitignored — see Setup)
├── notebooks/
│   └── 01_eda_and_modeling.ipynb   # full pipeline: EDA → leakage fix →
│                                    # SMOTE → model comparison → confusion
│                                    # matrices → ROC/PR curves → threshold
│                                    # tuning → cross-validation → SHAP
├── src/
│   ├── preprocessing.py        # reusable feature engineering functions
│   └── app.py                  # Streamlit deployment app
├── model.pkl                   # trained Random Forest (generated by notebook)
├── background.pkl              # SHAP background sample (generated by notebook)
├── requirements.txt
└── README.md
```

## Setup

**1. Get the data**

Download the [LendingClub loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
from Kaggle. Only the `accepted_2007_to_2018Q4.csv` file is used — the
rejected-applications file has no outcome label and isn't relevant to
default prediction. Place the CSV in `data/archive/`.

**2. Environment**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Run the notebook**

```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```
Run top to bottom. This trains the model, evaluates it, and saves
`model.pkl` and `background.pkl` for the app.

**4. Run the app**

```bash
streamlit run src/app.py
```

---

## Tech stack

Python, Pandas, Scikit-learn, imbalanced-learn (SMOTE), SHAP, Streamlit,
Plotly, Matplotlib/Seaborn.

## Limitations

- **Selection bias:** trained only on *approved* LendingClub applicants —
  the model has no information about applicants who were rejected
  outright, so it can't speak to that population.
- **Not calibrated for deployment:** the business-impact figure above is
  a rough estimate, not a validated financial model — real deployment
  would need calibrated probabilities and an actual cost matrix for
  false negatives vs. false positives.
- **Threshold left at default (0.5):** a production system would likely
  tune this against real cost estimates rather than the standard cutoff
  used here.

## What I'd do with more time

- Cost-sensitive threshold selection using real false-negative /
  false-positive cost estimates
- Hyperparameter tuning via `GridSearchCV`
- Compare against XGBoost / LightGBM
- Probability calibration (Platt scaling / isotonic regression)
