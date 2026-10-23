# Interview Prep — Poushik (Drift Monitoring)

Your prep for the Payment Anomaly Detection project. You built the drift
monitoring pipeline and co-refactored the codebase. Your teammate
implemented the EDA, model training and evaluation. Go deep on section 3,
and know section 5 well enough to answer at a high level.

All numbers come from the measured runs in [06-modeling.md](06-modeling.md),
[07-evaluation-metrics.md](07-evaluation-metrics.md) and
[08-drift-monitoring.md](08-drift-monitoring.md).

---

## 1. Your resume bullets

**UPI Transaction Anomaly Detection with Drift Monitoring (Team Project)**
- Built the drift monitoring pipeline for an unsupervised Isolation Forest
  fraud-detection system on the PaySim dataset, tracking feature drift with
  Population Stability Index (PSI) and Kolmogorov-Smirnov tests.
- Co-refactored the codebase into a modular MLOps pipeline with a teammate,
  who implemented the EDA, model training and evaluation.

---

## 2. Your pitch

### 30 seconds

> We built an end-to-end fraud-detection pipeline on PaySim, a
> 6.3-million-row synthetic payments dataset where only 0.13% of
> transactions are fraud. The model is an unsupervised Isolation Forest.
> My part was what happens after training: I built the drift monitoring that
> compares incoming data with the training data using PSI and the KS test,
> logs the results, and retrains the model automatically when the drift is
> significant. I also co-refactored the project into a modular pipeline with
> my teammate, who built the model and its evaluation.

### 2 minutes: walk the pipeline, then zoom in on your part

1. Clean and load 6.3M transactions into SQLite (500k-row chunks).
2. Engineer 12 features: balance residuals, an account-drain flag, log
   amount, a merchant flag, one-hot type, and velocity.
3. Train an Isolation Forest on the earliest 80% of rows by time. Evaluate
   on the later 20%.
4. **Your part:** every drift check compares a 200k-row training reference
   sample with a recent window. It computes PSI and KS for all 12 features,
   stores them in `drift_metrics`, and applies a dual trigger: retrain if 3
   or more features have PSI > 0.25, or PR-AUC drops more than 15%. Every
   decision is logged in `retrain_log`, and the dashboard shows the latest
   drift results.

### Numbers to memorize

| Fact | Value |
|---|---|
| Dataset | PaySim, 6,362,620 rows, 744 hourly steps (30 days) |
| Fraud rate | 0.129% (8,213 rows), only in `TRANSFER` and `CASH_OUT` |
| Features | 12 (7 of them binary) |
| PSI bands | < 0.10 none · 0.10–0.25 moderate · > 0.25 significant |
| KS threshold | p < 0.05 |
| Retrain trigger | 3 or more features with PSI > 0.25, **or** PR-AUC drop > 15% |
| Reference window | 200,000-row sample of the training partition (steps 1–355) |
| Current window | Steps 600–744 (out of time) |
| Drift result | PSI < 0.10 on all 12 features; KS flags 7 of 12; no retrain triggered; about 25 s runtime |
| Model result | Held-out ROC-AUC 0.854 vs train 0.851 (no degradation) |
| Tests | 35 pytest tests, including PSI/KS behaviour |

---

## 3. Your deep dive: drift monitoring

### What you built
- [`drift/psi.py`](../drift/psi.py): PSI per feature.
- [`drift/ks_test.py`](../drift/ks_test.py): two-sample KS test per feature.
- [`drift/retrain.py`](../drift/retrain.py): loads both windows, runs PSI and
  KS on every feature, writes `drift_metrics`, evaluates the trigger, calls
  `train.py` and `evaluate.py` when triggered, and logs every decision to
  `retrain_log`. `--no-auto-retrain` runs the check without retraining.

### PSI: be able to write this on a whiteboard

```
PSI = Σ over buckets (actual% − expected%) × ln(actual% / expected%)
```

- Bucket edges are **deciles of the reference** distribution. The outer
  edges are set to ±∞, so new values outside the training range still land
  in a bucket.
- Duplicate edges are collapsed (features with many zeros).
- Empty buckets get ε = 1e-6 to avoid log(0) and divide-by-zero.
- Bands: **< 0.10** none, **0.10–0.25** moderate, **> 0.25** significant.
  These are the standard credit-risk monitoring thresholds.
- PSI is a symmetric form of KL divergence, so swapping reference and
  current gives the same value.

### KS test
- `scipy.stats.ks_2samp`. The statistic is the largest vertical gap between
  the two empirical CDFs. Drift is flagged when p < 0.05.
- It needs no binning, so it catches shifts within a single PSI bucket.

### Why use both? (most likely question)
They fail in opposite ways. KS gives a p-value, so with millions of rows
even a tiny, harmless difference is "significant". In our run, KS flagged 7
of 12 features with p-values as low as 1e-101, while PSI was under 0.10 on
every feature. If KS alone were the trigger, the model would retrain on
noise every time. So the **trigger uses PSI**, because PSI measures how big
the shift is, and KS is logged as an early warning. (Some of those PSI
zeros come from limitation #2 below, so be precise about this if pushed.)

### The dual retraining trigger
Retrain if **either**:
- **(a) 3 or more features exceed PSI 0.25 at the same time.** Requiring
  several features avoids retraining because of one noisy feature.
- **(b) PR-AUC drops more than 15%** relative to the deployed model. This
  catches **concept drift**.

**Data drift vs concept drift:** PSI and KS only see the input distribution
P(X). Concept drift is a change in P(y | X): the inputs look the same, but
what counts as fraud has changed. Only a performance metric can catch it,
which is why there are two triggers.

### Reference vs current window
The reference is a 200,000-row random sample of the **training partition
only**, saved when the model is trained. The current window defaults to
steps 600–744, entirely in the held-out period. So the check compares what
the model learned from with data it has never seen, which is the comparison
that matters in production.

### Result and how to explain it
No feature reached PSI 0.10, so no retrain was triggered. That's the
correct outcome: your teammate's out-of-time evaluation showed ROC-AUC
didn't drop (0.851 train vs 0.854 held-out). Drift monitoring and the
evaluation agree.

### Known limitations: raise these yourself before an interviewer finds them

1. **The PR-AUC trigger never fires.** `check_and_retrain()` passes the
   stored `pr_auc` as both the current and the baseline value, so the drop
   is always 0. Fix: score the current window with the deployed model and
   compare with the deployed run's held-out PR-AUC. PR-AUC rises when the
   fraud rate rises, so compare lift (PR-AUC ÷ base rate) or ROC-AUC
   instead.
2. **PSI can't see shifts in binary features.** For a 0/1 feature, the
   decile edges collapse to {0, 1}. Setting the outer edges to ±∞ then
   leaves a single bucket, so PSI is always 0, even for a shift from 5% to
   80% ones. 7 of the 12 features are binary (`origZeroedOut`,
   `isMerchantDest` and the five `type_*` columns), so only 5 can ever count
   toward the "3+ features" trigger. KS still catches these shifts, which is
   why `origZeroedOut` and `type_CASH_OUT` show PSI 0 but KS flags them. Fix:
   compute PSI for binary and categorical features on value frequencies, not
   quantile buckets.
3. **The velocity features are nearly constant.** PaySim originator IDs
   almost never repeat, so `txnVelocity1h/24h` are almost always the same
   value. That's why their KS statistic is about 0.
4. **No promotion gate.** A retrained model is evaluated and logged, but
   it's never compared with the current model before it takes over. A
   production version would use champion/challenger.
5. **Batch and manual.** In production the check would run on a schedule
   over a sliding window and send alerts.

Framing: *"While writing up the project, I found two bugs in my own trigger
logic. Here's what they are and how I'd fix them."* This lands much better
than an interviewer finding them.

### Questions to expect
- **Why deciles, not equal-width bins?** Equal-frequency buckets give each
  bucket about 10% of the reference, so the expected percentages are stable
  and no bucket is empty. Equal-width bins break on skewed features like
  `amount`.
- **Why the ε?** If a bucket is empty in either window, ln(0) or a division
  by zero breaks the sum. ε keeps it finite.
- **Why 0.10 and 0.25?** They're the industry-standard bands from credit-risk
  scorecard monitoring. They're conventions, not laws; you'd tune them per
  feature in production.
- **Why not Wasserstein or Jensen-Shannon?** Both are valid. PSI was chosen
  for its well-known bands. Jensen-Shannon is bounded and handles
  categorical features naturally, so it's a good upgrade, especially given
  limitation #2.
- **How do you choose window sizes?** Large enough for stable bucket
  proportions, small enough to react quickly. In production, a daily or
  weekly sliding window.
- **What if drift fires but performance is fine?** That's why the trigger
  needs 3 or more features. Moderate drift should go to a human to review,
  not trigger an automatic retrain.
- **What about label delay?** Fraud labels arrive weeks later (chargebacks).
  That's the reason for having label-free drift metrics at all: PSI and KS
  give an early signal before PR-AUC can be measured.

---

## 4. Shared project questions

**Why unsupervised?** With 0.13% positives, a supervised model mostly
memorizes the specific fraud the simulator injected. Real fraud changes
and labels arrive late. An unsupervised model flags anything unusual and
can be retrained without labels, which suits a drift-triggered retraining
loop. Trade-off: a supervised model would probably score higher on PaySim.
The usual production answer is a hybrid.

**How does Isolation Forest work?** Random feature, random split value,
repeated until each point is isolated. Anomalies are isolated in fewer
splits, so a shorter average path means a more anomalous point.

**Is this really UPI data?** No, say so plainly. PaySim is a synthetic
mobile-money dataset calibrated on real logs. It's a public stand-in for
UPI-style transfers and cash-outs, because real payment logs aren't public.

**What makes it "MLOps"?** Independently runnable stages; one shared feature
definition for training, scoring and drift; versioned runs in `model_runs`;
drift and retrain decisions logged for audit; tests; a dashboard. What's
missing: CI, containers, a model registry, a serving API, scheduling.

---

## 5. Your teammate's half (know this at summary level)

- **EDA:** 0 missing values or duplicates. Fraud only in `TRANSFER` and
  `CASH_OUT`. **97.7%** of frauds drain the sender's account, vs 23.8% of
  legitimate transactions. Balances that don't reconcile are a fraud signal.
  Those findings became the features.
- **Model:** `IsolationForest(n_estimators=200, contamination=0.0013)`, with
  contamination set to the real fraud rate. Trained on steps 1–355 (80% of
  rows); the label is never used as a feature.
- **Time-based split:** a deployed model only scores future transactions,
  and a random split would leak account history across the split.
- **Metrics:** PR-AUC (primary), ROC-AUC, and precision/recall at the top
  0.5% (review capacity). Results: ROC-AUC **0.854** held-out, PR-AUC 0.028
  (about 8× random), and precision@0.5% 0.039.
- **Main weakness:** it catches no `CASH_OUT` fraud (0 of 2,112). The
  features don't separate fraudulent cash-outs from normal ones.

**How it connects to your work:** the training partition is your reference
window, and the evaluation metrics in `model_runs` are what the PR-AUC
trigger compares against.

---

## 6. The refactor (joint work)

- Packages per stage (`preprocessing`, `database`, `sql`, `models`, `drift`,
  `dashboard`), each runnable on its own.
- **One feature list** (`MODEL_FEATURE_COLUMNS`) used for training, scoring
  and drift, so they can never disagree (no training/serving skew).
- SQLite tables `transactions`, `model_runs`, `drift_metrics` and
  `retrain_log`; velocity computed with SQL window functions.
- 35 pytest tests: cleaning rules, feature math, PSI and KS behaviour, and
  the train/held-out split.

**"Who did what?"** Answer the same way your teammate will: *"I built drift
monitoring and the retraining loop. My teammate did the EDA, features,
training and evaluation. We refactored the pipeline together."*

---

## 7. Tough questions

- **"Your drift monitoring found nothing. Did it work?"** Yes. The
  out-of-time evaluation showed no drop, so "no drift" was the right answer,
  and the tests confirm PSI and KS detect an injected shift. Then raise
  limitation #2 yourself.
- **"The model's precision is 4%. Why monitor a weak model?"** Monitoring is
  model-agnostic: the same code would watch a production XGBoost model. And
  4% is about 12× the base rate, at zero labeling cost.
- **"How would this run in production?"** A scheduled drift job over a
  sliding window, alerts on moderate PSI, automatic retraining only on
  significant drift, a champion/challenger gate before promotion, a model
  registry, and CI running the tests.
- **"What would you change first?"** Fix limitations #1 and #2. Without them,
  the trigger can't fire on performance and can't see 7 of the 12 features.
