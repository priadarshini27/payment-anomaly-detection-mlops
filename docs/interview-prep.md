# Interview Prep — Payment Anomaly Detection with Drift Monitoring

Prep notes for both contributors. Each person should be able to give the
shared pitch, go deep on their own half, and give a short, accurate
answer about the other half.

| Area | Owner |
|---|---|
| Drift monitoring (PSI, KS test, retraining trigger) | Poushik |
| EDA, feature engineering, model training and evaluation | Teammate |
| Refactor into a modular MLOps pipeline, tests, docs | Both |

All numbers below come from the measured runs recorded in
[06-modeling.md](06-modeling.md), [07-evaluation-metrics.md](07-evaluation-metrics.md)
and [08-drift-monitoring.md](08-drift-monitoring.md). Don't quote numbers
that aren't in those files.

---

## 1. Resume bullets

**Poushik**
- Built the drift monitoring pipeline for an unsupervised Isolation Forest
  fraud-detection system on the PaySim dataset, tracking feature drift with
  Population Stability Index (PSI) and Kolmogorov-Smirnov tests.
- Co-refactored the codebase into a modular MLOps pipeline with a teammate,
  who implemented the EDA, model training and evaluation.

**Teammate**
- Implemented the EDA, feature engineering, model training and evaluation
  for an unsupervised Isolation Forest fraud-detection system on the PaySim
  dataset, using a time-ordered 80/20 split so the model was evaluated on
  later transactions it never saw in training.
- Built an evaluation framework for heavily imbalanced data (<0.2% fraud)
  around PR-AUC, ROC-AUC and precision/recall@k at the top 0.5% of
  transactions; co-refactored the codebase into a modular MLOps pipeline.

---

## 2. The pitch

### 30 seconds

> We built an end-to-end fraud-detection pipeline on PaySim, a 6.3-million-row
> synthetic mobile-money dataset where only 0.13% of transactions are fraud.
> Because labels that rare are unreliable to learn from, we used an
> unsupervised Isolation Forest and kept the fraud label for evaluation only.
> We didn't stop at training: the system checks whether incoming data has
> drifted from the training data using PSI and the KS test, and retrains
> automatically when it has. Everything is stored in SQLite and shown in a
> Streamlit dashboard.

### 2 minutes (walk the pipeline)

1. **Ingest and clean.** Read the CSV in 500k-row chunks, validate the schema, drop invalid rows.
2. **Engineer features.** 12 features: balance residuals, an account-drain flag, log amount, a merchant flag, one-hot transaction type, and transaction velocity.
3. **Store.** SQLite tables `transactions`, `model_runs`, `drift_metrics` and `retrain_log`. Velocity features are computed with SQL window functions.
4. **Train.** Isolation Forest on the earliest 80% of rows by time (steps 1–355).
5. **Evaluate.** On the held-out later 20%: PR-AUC, ROC-AUC, precision/recall at the top 0.5%, and a confusion matrix by transaction type.
6. **Monitor drift.** PSI and KS per feature, comparing a training reference window with a recent window.
7. **Retrain.** Triggered when 3 or more features show significant PSI drift, or PR-AUC drops more than 15%. Every check is logged.
8. **Dashboard.** Six Streamlit views: overview, fraud analytics, transactions, model performance, drift, and system health.

### Numbers to memorize

| Fact | Value |
|---|---|
| Dataset | PaySim, 6,362,620 rows, 744 hourly steps (30 days) |
| Fraud rate | 0.129% (8,213 rows), only in `TRANSFER` and `CASH_OUT` |
| Features | 12 |
| Split | Train steps 1–355 (5.11M rows, 0.077% fraud); held-out steps 356–743 (1.25M rows, 0.339% fraud) |
| Model | `IsolationForest(n_estimators=200, contamination=0.0013, max_samples='auto' (256), random_state=42)` |
| Held-out ROC-AUC | **0.854** (train 0.851, gap −0.003: no overfitting) |
| Held-out PR-AUC | 0.028 (about 8.2× the base rate) |
| Precision / recall @ top 0.5% | 0.039 / 0.058 (precision about 11.6× the 0.339% base rate) |
| Drift result (train vs steps 600–744) | PSI < 0.10 on all features; KS flags 7 of 12; no retrain triggered |
| Tests | 35 pytest tests (cleaning, features, drift, train split) |

---

## 3. Shared questions (both should answer these)

**Why unsupervised instead of a classifier like XGBoost?**
With 0.13% positives, a supervised model mostly learns the exact fraud
patterns the simulator injected. In real payments, fraud patterns change
and labels arrive late (chargebacks take weeks). An unsupervised model
flags anything unusual, needs no labels to retrain, and that makes
drift-triggered retraining practical. The honest trade-off: a supervised
model would almost certainly score higher on PaySim itself. A strong answer
mentions that a hybrid (anomaly score as a feature in a supervised model)
is the usual production approach.

**How does Isolation Forest work?**
Each tree picks a random feature and a random split value, recursively,
until each point is isolated. Anomalies are few and different, so they get
isolated in fewer splits. The anomaly score comes from the average path
length across trees, normalized by the expected path length for a sample
of that size. A shorter path means more anomalous. Each tree sees only 256
sampled rows, so it is fast and hard to overfit.

**Why not accuracy?**
A model that predicts "not fraud" for everything scores 99.87% accuracy and
catches nothing.

**Why PaySim? Is this really UPI data?**
No, and say so plainly. PaySim is a synthetic mobile-money dataset
calibrated on real transaction logs from an African provider. We used it as
a public stand-in for UPI-style peer-to-peer payments (transfers,
cash-outs, merchant payments), because real payment logs aren't publicly
available.

**What does "MLOps" mean in this project?**
Separate, independently runnable stages; a single shared feature definition
used at both training and scoring time; versioned runs in `model_runs`
with hyperparameters and metrics; drift metrics and retrain decisions
logged for audit; a test suite; and a dashboard over live tables. What's
missing: no CI, no containerization, no model registry or serving API, and
it runs as batch jobs.

**What would you do next?**
Fix the two drift limitations in section 4; add features for `CASH_OUT`
fraud (section 5); score the live window to make the PR-AUC trigger real;
add CI and a scheduler (cron or Airflow) for the drift check; try a hybrid
supervised model.

---

## 4. Poushik — drift monitoring deep dive

### What I built
- [`drift/psi.py`](../drift/psi.py): PSI per feature.
- [`drift/ks_test.py`](../drift/ks_test.py): two-sample KS test per feature.
- [`drift/retrain.py`](../drift/retrain.py): loads the reference and current
  windows, runs both tests on all 12 features, writes `drift_metrics`,
  applies the dual trigger, retrains and re-evaluates if triggered, and logs
  every decision to `retrain_log`.

### PSI: be able to write this on a whiteboard

```
PSI = Σ over buckets (actual% − expected%) × ln(actual% / expected%)
```

- Bucket edges are **deciles of the reference** distribution. The outer
  edges are set to ±∞, so current values outside the training range still
  land in a bucket.
- Duplicate edges are collapsed (features with many zeros).
- Empty buckets are replaced with ε = 1e-6 to avoid log(0) and divide-by-zero.
- Bands: **< 0.10** none, **0.10–0.25** moderate, **> 0.25** significant.
  These are the standard credit-risk monitoring thresholds.
- It is a symmetric form of KL divergence, which is why both terms appear.

### KS test
- `scipy.stats.ks_2samp`: the statistic is the maximum vertical gap between
  the two empirical CDFs. Drift is flagged when p < 0.05.
- It catches shifts inside a single PSI bucket that PSI would miss.

### Why use both? (likely question)
They fail differently. KS has a p-value, so at large sample sizes it
reports statistically significant differences that are practically tiny.
Our result shows this: PSI under 0.10 on every feature, but KS flagged 7
of 12 with p-values as low as 1e-101. (Some of those PSI zeros are also
limitation #2 below, so be precise about this if pushed.) If KS alone were the trigger, the
model would retrain on noise every time. So the **trigger uses PSI**
(magnitude), and KS is logged as an early-warning signal.

### The retraining trigger
Retrain if **either**:
- (a) **3 or more features** exceed PSI 0.25 at the same time. Requiring
  several features avoids retraining on one noisy feature; or
- (b) **PR-AUC drops more than 15%** relative to the deployed model. This
  catches concept drift, where the inputs look the same but the relation
  between inputs and fraud changes. PSI can't detect that.

**Data drift vs concept drift:** PSI and KS only see P(X). Concept drift is
a change in P(y | X), and only shows up in performance metrics. That's why
there are two triggers.

### Reference window
A 200,000-row random sample of the **training partition only**, saved at
training time. The current window defaults to steps 600–744, which lies
entirely in the held-out period. So the check compares what the model
learned from with data it has never seen, which is the comparison that
matters in production.

### Result
No feature reached PSI 0.10 and retraining wasn't triggered. That agrees
with the evaluation: held-out ROC-AUC didn't drop, so "no drift, no
retrain" was the correct decision.

### Known limitations: raise these yourself before an interviewer finds them

1. **The PR-AUC trigger never fires.** `check_and_retrain()` passes the
   stored `pr_auc` as both the current and the baseline value, so the drop
   is always 0. The fix: score the current window with the deployed model,
   then compare with the deployed run's held-out PR-AUC. Watch for the
   base-rate effect: PR-AUC rises when the fraud rate rises, so normalize
   by base rate (lift) or use ROC-AUC.
2. **PSI can't see shifts in binary features.** For a 0/1 feature, the
   decile edges collapse to {0, 1}. Setting the outer edges to ±∞ then
   leaves a single bucket, so PSI is always 0, even for a shift from 5% to
   80% ones. 7 of the 12 features are binary (`origZeroedOut`,
   `isMerchantDest` and the five `type_*` columns), so only 5 features can
   ever count toward the "3+ features" trigger. KS still catches these
   shifts, which is why `origZeroedOut` and `type_CASH_OUT` show PSI 0 but
   KS flags them. The fix: compute PSI for binary and categorical features
   on value frequencies instead of quantile buckets.
3. **The velocity features are nearly constant.** PaySim originator IDs
   almost never repeat, so `txnVelocity1h/24h` carry very little signal.
   That's why their KS statistic is about 0.
4. **The drift check is batch and manual.** In production it would run on a
   schedule over a sliding window and alert.

### Questions to expect
- *Why deciles?* Equal-frequency buckets give each bucket about 10% of the
  reference, so no bucket is empty and the expected% values are stable.
  Equal-width bins break on skewed features like `amount`.
- *Is PSI symmetric?* Yes, swapping reference and current gives the same value.
- *Why not Wasserstein or Jensen-Shannon?* Both are valid. PSI was chosen for
  its interpretable bands from credit risk. JS is bounded and symmetric and
  handles categorical features well, so it would be a reasonable upgrade.
- *How do you pick the window size?* Large enough for stable bucket
  proportions, small enough to react quickly. It's a trade-off. Here it
  follows the simulation's step range.
- *What if drift fires but performance is fine?* That's why the trigger
  needs 3 or more features. In production, a human would review moderate
  drift before acting.

---

## 5. Teammate — EDA, modeling and evaluation deep dive

### EDA findings (from [`eda/eda.ipynb`](../eda/eda.ipynb))
| Finding | What it led to |
|---|---|
| 0 missing values, 0 duplicates, 0 negative values | Checks kept as safety nets, no imputation |
| 16 zero-amount rows, **all fraud** | Dropped as invalid records, a documented trade-off (16 of 8,213 frauds) |
| Fraud only in `TRANSFER` (0.77%) and `CASH_OUT` (0.18%) | One-hot `type`; evaluation broken down by type |
| **97.7%** of frauds drain the origin account vs **23.8%** of legitimate transactions | `origZeroedOut` feature |
| Balances don't reconcile (old − amount ≠ new) | Treated as signal, not noise: `balanceDeltaOrig/Dest` |
| `amount` heavily right-skewed | `logAmount = log1p(amount)` |
| Every `PAYMENT` goes to an `M…` merchant account | `isMerchantDest` |
| PaySim's own `isFlaggedFraud` rule fires 16 times | Not used: 100% precision but almost no recall |

### Modeling choices
- **Label never used as a feature.** `isFraud` is used only in `evaluate.py`.
  `isFlaggedFraud` and ID columns are excluded to prevent leakage.
- **`StandardScaler`** fitted on the training partition only.
- **Hyperparameters:**
  - `n_estimators=200` stabilizes the averaged path lengths.
  - `contamination=0.0013` matches the real fraud rate instead of `'auto'`.
  - `max_samples=256` keeps it fast on 5M rows.
  - `max_features=1.0`, because 12 features is small.
  - `random_state=42` for reproducibility.
- **Time-based split, not random.** A deployed model only scores the
  future. A random split leaks later behaviour, and since velocity features
  use account history, rows sharing history would end up on both sides. The
  split is on whole steps and by **row count**: volume collapses after
  about step 400 while fraud keeps arriving.

### Evaluation choices
- **PR-AUC is the primary metric.** Its random baseline is the fraud rate,
  so it reflects performance on the rare class. ROC-AUC looks good when
  negatives dominate.
- **Precision/recall @ top 0.5%** models a fraud team that can review a
  fixed number of alerts per day.
- **Confusion matrix by type** shows where errors land.
- **The overfitting check uses ROC-AUC, not PR-AUC.** The held-out fraud
  rate is 4.4× higher (0.339% vs 0.077%), so raw PR-AUC rises even if
  ranking quality doesn't change. ROC-AUC doesn't depend on the base rate.
  This is a good, subtle point to bring up yourself.

### Results and how to frame them honestly
- ROC-AUC 0.854 held-out vs 0.851 train: **no overfitting**, as expected
  for Isolation Forest, which never sees labels and uses 256-row trees.
- PR-AUC 0.028 sounds low, but it's about 8× random. Precision@0.5% is
  about 11.6× the base rate.
- **Main weakness: `CASH_OUT`.** All 246 caught frauds are `TRANSFER`s.
  **0 of 2,112 `CASH_OUT` frauds** were flagged. Fraudulent cash-outs look
  like normal ones in the current features. This is missing signal
  (underfitting), not overfitting.
- Fix: add features that link a `TRANSFER` to the `CASH_OUT` that follows it
  (PaySim fraud is usually transfer, then cash-out), destination-account
  history, and amount relative to the account's usual amount.

### Questions to expect
- *Why is contamination set to the fraud rate?* It sets the decision
  threshold so the flagged fraction matches the expected base rate. Ranking
  metrics (ROC-AUC, PR-AUC) don't depend on it at all.
- *How would you tune an unsupervised model without labels?* Use a small
  labeled validation set, or business constraints such as alert volume. Here
  the labels were used only to evaluate, never to tune on the test window.
- *Why scale features for a tree model?* Isolation Forest doesn't strictly
  need it, because splits are per feature. The scaler keeps the
  scoring-time pipeline consistent and is harmless.
- *Why not Local Outlier Factor, an autoencoder, or One-Class SVM?* They
  scale badly to 6M rows (LOF, OCSVM) or need much more tuning
  (autoencoder). Isolation Forest is linear-time and a strong baseline.

---

## 6. Joint work: the refactor

- Turned the scripts into packages (`preprocessing`, `database`, `sql`,
  `models`, `drift`, `dashboard`), each runnable on its own.
- One feature definition, `MODEL_FEATURE_COLUMNS` in
  `preprocessing/feature_engineering.py`, used for training, scoring **and**
  drift. This prevents training/serving skew.
- SQLite schema with indexes on `step`, `nameOrig`, `type` and `isFraud`;
  velocity computed once with window functions.
- Each run recorded in `model_runs` (hyperparameters, split cutoff, metrics),
  so models are compared directly in SQL.
- 35 pytest tests covering cleaning rules, feature math, PSI/KS behaviour
  and split correctness (no step appears on both sides).

**If asked "who did what?"**, answer plainly and consistently with each
other: Poushik built drift monitoring and the retraining loop; the teammate
did EDA, features, training and evaluation; both did the refactor. Each of
you should still be able to explain the other's half at the level of
section 3.

---

## 7. Curveball questions

- **"Your precision is 4%. Is this model useful?"** As a standalone alerting
  system, not yet. It catches about 12× more fraud per reviewed alert than
  random review, at no labeling cost. It's a baseline and a feature for a
  supervised model, and the project's focus is the monitoring loop around it.
- **"Drift monitoring found nothing. Did it work?"** Yes. The out-of-time
  evaluation showed no performance drop, so "no drift" was the correct
  result, and the tests verify PSI and KS detect an injected shift in a
  continuous feature. Be upfront about limitation #2: PSI can't see drift
  in the binary features at all.
- **"What happens if retraining makes the model worse?"** Right now the new
  run is evaluated and logged, but it's not compared before replacing the
  old one. A production version would only promote a model that beats the
  current one on a held-out window (champion/challenger).
- **"How would this run in production?"** A streaming or micro-batch scorer
  using the same feature code, a scheduled drift job over a sliding window,
  alerts on moderate PSI, gated retraining, a model registry, and CI running
  the test suite.
