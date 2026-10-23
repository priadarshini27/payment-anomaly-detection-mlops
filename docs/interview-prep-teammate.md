# Interview Prep — EDA, Modeling & Evaluation

Your prep for the Payment Anomaly Detection project. You implemented the
EDA, feature engineering, model training and evaluation, and co-refactored
the codebase. Your teammate (Poushik) built the drift monitoring. Go deep on
section 3, and know section 5 well enough to answer at a high level.

All numbers come from the measured runs in [02-dataset-and-eda.md](02-dataset-and-eda.md),
[06-modeling.md](06-modeling.md) and [07-evaluation-metrics.md](07-evaluation-metrics.md).

---

## 1. Your resume bullets

**UPI Transaction Anomaly Detection with Drift Monitoring (Team Project)**
- Implemented the EDA, feature engineering, model training and evaluation
  for an unsupervised Isolation Forest fraud-detection system on the PaySim
  dataset. Set contamination to the empirical fraud rate and used a
  time-ordered 80/20 split, so the model was evaluated on later
  transactions it never saw in training.
- Built an evaluation framework for heavily imbalanced data (<0.2% fraud)
  around PR-AUC, ROC-AUC and precision/recall@k at the top 0.5% of
  transactions, sized to manual-review capacity. Co-refactored the codebase
  into a modular MLOps pipeline.

---

## 2. Your pitch

### 30 seconds

> We built an end-to-end fraud-detection pipeline on PaySim, a
> 6.3-million-row synthetic payments dataset where only 0.13% of
> transactions are fraud. I did the EDA, turned its findings into features,
> trained an unsupervised Isolation Forest, and built the evaluation. I used
> a time-based split, so the model is tested on transactions from after its
> training period, and metrics that are meaningful at a 0.13% fraud rate. My
> teammate built drift monitoring that retrains the model when the data
> shifts.

### 2 minutes: walk the pipeline, then zoom in on your part

1. **EDA** on all 6.3M rows. Every cleaning rule and feature traces back to a
   specific finding.
2. **Features:** 12 features, including balance residuals, an account-drain
   flag, log amount, a merchant flag, one-hot type and velocity.
3. **Training:** Isolation Forest on the earliest 80% of rows by time; the
   fraud label is never a feature.
4. **Evaluation** on the later 20%: PR-AUC, ROC-AUC, precision/recall at the
   top 0.5%, a confusion matrix by type, and a train vs held-out check for
   overfitting. Results are stored per run in `model_runs`.
5. Then drift monitoring (your teammate's part) compares new data with your
   training data and retrains when needed.

### Numbers to memorize

| Fact | Value |
|---|---|
| Dataset | PaySim, 6,362,620 rows, 744 hourly steps (30 days) |
| Fraud rate | 0.129% (8,213 rows) |
| Fraud by type | `TRANSFER` 0.769%, `CASH_OUT` 0.183%, all other types 0% |
| Account drain | 97.7% of frauds vs 23.8% of legitimate transactions |
| Split | Train steps 1–355 (5,113,881 rows, 0.077% fraud) · held-out steps 356–743 (1,248,723 rows, 0.339% fraud) |
| Model | `IsolationForest(n_estimators=200, contamination=0.0013, max_samples=256, max_features=1.0, random_state=42)` |
| ROC-AUC | **0.854** held-out vs 0.851 train (gap −0.003) |
| PR-AUC | 0.028 held-out (lift about 8.2× over the base rate) |
| Precision / recall @ top 0.5% | 0.039 / 0.058 |
| Frauds caught (held-out) | 246, all `TRANSFER`; 0 of 2,112 `CASH_OUT` |
| Tests | 35 pytest tests |

---

## 3. Your deep dive: EDA, modeling and evaluation

### EDA: each finding led to a decision
Notebook: [`eda/eda.ipynb`](../eda/eda.ipynb).

| Finding | Decision |
|---|---|
| 0 missing values, 0 duplicates, 0 negative values | No imputation; the checks stay in as safety nets for future data |
| 16 zero-amount rows, **all fraud** | Dropped as invalid records. A deliberate trade-off: it loses 16 of 8,213 frauds |
| Fraud only in `TRANSFER` and `CASH_OUT` | One-hot `type`; evaluation broken down by type |
| **97.7%** of frauds drain the sender's account vs **23.8%** of legitimate transactions | `origZeroedOut` feature, one of the strongest signals |
| Balances often don't reconcile (old − amount ≠ new) | Treated as signal, not dirty data: `balanceDeltaOrig` and `balanceDeltaDest` |
| `amount` heavily right-skewed | `logAmount = log1p(amount)` |
| Every `PAYMENT` goes to an `M…` merchant account, and merchants have no balance | `isMerchantDest` (explains why `balanceDeltaDest` behaves differently for merchants) |
| PaySim's own `isFlaggedFraud` rule fires 16 times | Not used: 100% precision but misses 8,197 of 8,213 frauds |
| No strong hour-of-day pattern | No time features |

Good line: *"I didn't run a generic cleaning checklist. Every rule in the
pipeline is there because the EDA found a reason for it."*

### Features (12)
`logAmount`, `balanceDeltaOrig`, `balanceDeltaDest`, `origZeroedOut`,
`isMerchantDest`, five `type_*` one-hots, `txnVelocity1h`, `txnVelocity24h`.
Defined once in `preprocessing/feature_engineering.py` and used for
training, scoring and drift, so there's no training/serving skew.
**Excluded:** `isFraud` and `isFlaggedFraud` (label leakage) and account IDs.

### Why Isolation Forest
- **How it works:** each tree repeatedly picks a random feature and a random
  split value until each point is isolated. Anomalies are few and
  different, so they get isolated in fewer splits. The score is the average
  path length across trees, normalized by the expected path length for that
  sample size.
- **Why unsupervised:** with 0.13% positives, a supervised model memorizes
  the simulator's specific fraud patterns. Real fraud changes and labels
  arrive weeks late. An unsupervised model can be retrained without labels.
- **Why not LOF, One-Class SVM or an autoencoder:** LOF and OCSVM scale badly
  to 6M rows; an autoencoder needs much more tuning. Isolation Forest is
  linear-time and a strong baseline.

### Hyperparameters, with reasons
| Parameter | Value | Why |
|---|---|---|
| `n_estimators` | 200 | Averages path lengths over more trees, so scores vary less than with the default 100 |
| `contamination` | 0.0013 | Matches the real fraud rate, so the threshold flags the expected fraction. Ranking metrics don't depend on it |
| `max_samples` | 256 | Short isolation paths, fast on 5M rows, hard to overfit |
| `max_features` | 1.0 | Only 12 features, so no need to subsample them |
| `random_state` | 42 | Reproducible across training, evaluation and retraining |

`StandardScaler` is fitted on the training partition only. A tree model
doesn't strictly need scaling, but it keeps the scoring pipeline consistent.

### Time-based split (likely question)
- **Time, not random:** a deployed model only scores the future. A random
  split leaks later behaviour into training, and because velocity features
  use each account's history, rows sharing history would land on both sides.
- **Whole steps:** each simulated hour falls entirely on one side.
- **By row count, not step count:** transaction volume collapses after
  about step 400 while fraud keeps arriving, so "80% of steps" would not be
  80% of the data.

### Evaluation design
- **Not accuracy:** predicting "not fraud" for everything scores 99.87%.
- **PR-AUC is primary:** its random baseline is the fraud rate, so it
  measures how well the model ranks the rare class. ROC-AUC can look good
  just because negatives dominate.
- **Precision/recall @ top 0.5%:** models a fraud team that can review a
  fixed number of alerts per day.
- **Confusion matrix by type:** shows whether errors land in
  `TRANSFER`/`CASH_OUT` (expected) or elsewhere (a sign a feature is wrong).
- **The overfitting check uses ROC-AUC, not PR-AUC.** This is a subtle point
  worth raising yourself. The held-out fraud rate is 4.4× higher (0.339% vs
  0.077%), so raw PR-AUC rises even if the model ranks no better. ROC-AUC
  doesn't depend on the base rate, so a ROC-AUC gap is a real difference.
  The check warns if the gap is more than 0.05.
- Every run's metrics go into `model_runs`, so model versions can be
  compared with one SQL query.

### Results and how to frame them honestly
- **No overfitting:** ROC-AUC 0.854 held-out vs 0.851 train. That's expected,
  because Isolation Forest never sees labels and uses 256-row trees.
- **Low absolute numbers, explained:** PR-AUC 0.028 is about 8× random, and
  precision@0.5% (3.9%) is about 11.6× the base rate. This is an unsupervised
  baseline, not a finished fraud system.
- **Main weakness: `CASH_OUT`.** All 246 caught frauds are `TRANSFER`s, and
  **0 of 2,112 `CASH_OUT` frauds** were flagged. The current features don't
  separate fraudulent cash-outs from normal ones. That's missing signal, not
  overfitting, so the fix is better features, not more regularization.
- **Fixes:** a feature linking a `TRANSFER` to the `CASH_OUT` that follows it
  (PaySim fraud is usually a transfer, then a cash-out), destination-account
  history, and amount compared with the account's usual amount.
- The velocity features turned out to be nearly constant, because PaySim
  account IDs almost never repeat. Be ready to say so.

### Questions to expect
- **Why set contamination to the fraud rate?** It only sets the threshold.
  Ranking metrics are unchanged.
- **How do you tune an unsupervised model without labels?** A small labeled
  validation set, or business constraints such as alert volume. Here the
  labels were only used to evaluate, never to tune on the held-out window.
- **Why drop 16 real fraud rows?** Zero-amount transactions aren't valid
  financial records. It costs 0.2% of frauds, and the choice is documented.
- **Would supervised do better?** Almost certainly on PaySim. In production,
  you'd use the anomaly score as a feature in a supervised model (hybrid).
- **What does the score threshold (-0.0326) mean?** The anomaly score is
  the negated `decision_function`, so higher means more anomalous. The
  threshold is the score of the transaction at the top-0.5% mark: everything
  at or above it gets flagged. It comes from review capacity, not from
  sklearn's default cutoff at 0.

---

## 4. Shared project questions

**Is this really UPI data?** No, say so plainly. PaySim is a synthetic
mobile-money dataset calibrated on real logs. It's a public stand-in for
UPI-style transfers and cash-outs, because real payment logs aren't public.

**What makes it "MLOps"?** Independently runnable stages; one shared feature
definition; versioned runs in `model_runs`; drift and retrain decisions
logged; tests; a dashboard. What's missing: CI, containers, a model
registry, a serving API, scheduling.

---

## 5. Your teammate's half (know this at summary level)

- **Drift monitoring** compares a 200k-row sample of your training data with
  a recent window (steps 600–744), feature by feature.
- **PSI** (Population Stability Index) measures how much each feature's
  distribution has shifted, using decile buckets. Bands: < 0.10 none,
  0.10–0.25 moderate, > 0.25 significant.
- **The KS test** checks whether two samples come from the same
  distribution (p < 0.05 means drift). It's very sensitive at large sample
  sizes, so it's logged as a warning and not used as the trigger.
- **Retrain trigger:** 3 or more features with PSI > 0.25, **or** PR-AUC
  dropping more than 15%. When triggered, it re-runs your `train.py` and
  `evaluate.py`.
- **Result:** no significant drift, no retrain. That agrees with your
  finding that held-out ROC-AUC didn't drop.

**How it connects to your work:** your training partition is the drift
reference window, and your `model_runs` metrics are what the PR-AUC trigger
compares against.

---

## 6. The refactor (joint work)

- Packages per stage (`preprocessing`, `database`, `sql`, `models`, `drift`,
  `dashboard`), each runnable on its own.
- One feature list (`MODEL_FEATURE_COLUMNS`) for training, scoring and drift.
- SQLite tables `transactions`, `model_runs`, `drift_metrics` and
  `retrain_log`; velocity computed with SQL window functions.
- 35 pytest tests: cleaning rules, feature math, drift and the
  train/held-out split (no step on both sides).

**"Who did what?"** Answer the same way Poushik will: *"I did the EDA,
features, training and evaluation. My teammate built the drift monitoring
and retraining loop. We refactored the pipeline together."*

---

## 7. Tough questions

- **"Your precision is 4%. Is this model useful?"** Not yet as a standalone
  alerting system. It finds about 12× more fraud per reviewed alert than
  random review, with no labels. It's a baseline, and the evaluation already
  shows exactly where to improve (`CASH_OUT`).
- **"Isn't ROC-AUC 0.85 good?"** It's decent ranking, but at a 0.13% fraud
  rate ROC-AUC flatters the model, which is why PR-AUC is the primary metric.
- **"How do you know you didn't leak the test set?"** The scaler, the model
  and the drift reference never see held-out steps; a test checks no step is
  on both sides; the label is never a feature.
- **"What would you change first?"** `CASH_OUT` features, then try a hybrid
  supervised model on top of the anomaly score.
