# Data Mining and Machine Learning — Lecture Track

A narrative outline of each lecture, ordered for slide preparation. Each lecture is broken into a sequence of beats. Beats are concept-first, with concrete examples, pedagogical notes (what to highlight, common pitfalls), and explicit transitions to maintain a coherent story arc across the semester.

**Reading conventions:**
- *Concept* — the idea being introduced.
- *Example* — concrete dataset, plot, or code snippet to anchor it.
- *Note* — pedagogical highlight, pitfall to flag, or stylistic choice.
- *Transition* — how to bridge to the next beat or lecture.

---

## Course-wide narrative arc

The course progresses along three axes simultaneously:

1. **Data complexity**: tabular → time series → high-dimensional → images/text.
2. **Model complexity**: linear → kernel → tree-based → ensemble → neural → attention-based.
3. **Methodological maturity**: descriptive statistics → predictive modeling → validation discipline → interpretability → end-to-end pipelines.

Recurring threads to weave through every lecture:
- The bias-variance tradeoff (introduced W3, revisited every time a new model family appears).
- Generalization and the train/validation/test split discipline (introduced W3, formalized W4, revisited).
- "What does this model assume about the data?" — asked explicitly each time a new model is introduced.
- The toolbox metaphor — every method gets situated in a decision flowchart that culminates in W12.

---

# Week 1 — Setup and Data Preparation

## Lecture 1 — Environment Setup & Course Structure

**Narrative goal:** Get every student to a working environment, set expectations, and motivate why data preparation deserves its own week before any modeling.

1. **Welcome and course philosophy.** Frame the course as a *toolbox*, not a sequence of disconnected algorithms. Each week adds a tool; the final exam is about knowing which to reach for. *Note:* set the tone — this is applied, not theoretical, but rigor matters.
2. **Logistics.** Grading breakdown (20/20/20/40), midterm policy ("best of 2"), exercise rolling deadlines, project groups (≤4, ~25 reports). *Note:* spell out the "rolling hand-over" rule explicitly — students always misunderstand it the first time.
3. **What "data mining" and "machine learning" mean here.** Brief disambiguation: data mining = finding patterns, ML = learning predictive functions. Heavy overlap; we'll use both vocabularies.
4. **Course map.** Show the 12-week skeleton as a single slide. Highlight the three-axis progression (data complexity, model complexity, methodological maturity). *Transition:* "before any of this, we need a working environment."
5. **Why Python.** Ecosystem, scientific stack, dominance in ML. Brief mention of alternatives (R for stats, Julia for numerics) to acknowledge they exist.
6. **Installing Python.** Recommended distribution choice: discuss `uv`, `conda`, or system Python + `venv`. Pick one and stick with it for the course. *Note:* recommend `uv` for speed and reproducibility — students will thank you later.
7. **Virtual environments.** Why they matter. Live demo: create one, activate, deactivate. *Pitfall:* installing into the system Python; never do this.
8. **`requirements.txt`** Provide the course-wide one (numpy, pandas, scikit-learn, matplotlib, seaborn, statsmodels, jupyter to start). Show `pip install -r`.
9. **VS Code setup.** Python extension, Jupyter extension, interpreter selection. Live demo: open a notebook, run a cell.
10. **Jupyter vs scripts.** When to use which. *Note:* notebooks for exploration, scripts for reproducible pipelines — they'll need both.
11. **First-day check.** A 3-line "hello world": import numpy, print version, plot something. Anyone who can't get this working must flag it before the exercise.
12. **A word on Git.** Strongly encouraged, not required for grading. Point to resources.
13. **Introducing the course cluster.** This course provides access to a SLURM-managed compute cluster. *Note:* state the rationale explicitly — (a) no hardware discrimination, everyone has the same compute available; (b) SLURM and remote-machine workflows are an industry-standard skill the university doesn't otherwise teach; (c) the project in W11–W12 will require non-trivial compute that local laptops can't provide.
14. **Today's plumbing.** Three steps, all part of the W1 hand-in: generate an SSH key, log in to the cluster, run `hostname` and paste the output. *Note:* this is pure plumbing — no SLURM, no batch jobs yet. The goal is to surface account/network problems eleven weeks before they would otherwise bite.
15. **An escape hatch from day one.** Demo an interactive session: `srun --pty bash` → activate the shared env → run a simple script. *Note:* this is deliberately the *only* way to run code on the cluster we teach today — it lets a student with a weak laptop do every subsequent exercise on a compute node, while the full batch/`sbatch` workflow waits until it's motivated in W4. Keep it to one simple "it ran on a remote machine" moment.
16. **Where to get help.** Cheat sheet handout, office hours this week reserved for cluster access, TA contact. *Pitfall:* don't try this at 11pm before the W2 hand-in — start in the first 48 hours while support is available. *Transition:* "with the environment set, let's talk about what we actually do with data."

## Lecture 2 — Data Loading, Cleaning, and EDA

**Narrative goal:** Convince students that 80% of ML work happens before any model is trained, and equip them to do it well.

1. **The pipeline view.** Show the canonical ML pipeline: raw data → cleaning → EDA → preprocessing → modeling → evaluation → deployment. This week covers steps 1–3.
2. **Loading data with pandas.** `read_csv`, `read_excel`, common arguments (`sep`, `na_values`, `parse_dates`, `dtype`). *Example:* load Titanic.
3. **First-look diagnostics.** `.head()`, `.info()`, `.describe()`, `.shape`, `.dtypes`. *Note:* the goal is to develop an instinct for "what does this dataset look like?" in 60 seconds.
4. **Missing values — diagnosis.** `.isna().sum()`, missingness heatmaps. Distinguish MCAR, MAR, MNAR conceptually (without heavy formalism). *Pitfall:* not all missingness is missing data — sometimes it's a category.
5. **Missing values — treatment.** Drop rows, drop columns, mean/median/mode imputation, model-based imputation. Discuss tradeoffs. *Example:* on Titanic, `Age` vs `Cabin` need different strategies.
6. **Categorical encoding.** One-hot, ordinal, target encoding. *Note:* foreshadow that the right choice depends on the model (trees vs linear) — we'll come back to this.
7. **Normalization and scaling.** Standardization (z-score), min-max, robust scaling. *Pitfall:* always fit the scaler on training data only — flag this now, formalize it in W4.
8. **Descriptive statistics done right.** Mean vs median, variance vs IQR, skewness. When summary stats lie (Anscombe's quartet — show it).
9. **Visualization fundamentals.** Histograms, box plots, scatter plots, pair plots. *Example:* Titanic survival rate vs class, age, sex.
10. **Correlation analysis.** Pearson vs Spearman. Correlation matrices, heatmaps. *Pitfall:* correlation is not causation; correlation does not capture non-linear relationships.
11. **Outlier detection.** Visual (boxplots), statistical (z-score, IQR), and "should we even remove them?" *Note:* outliers are sometimes the signal you want.
12. **Class imbalance.** First exposure. *Example:* Titanic is moderately imbalanced. Foreshadow: this matters for classification metrics (W3).
13. **Feature engineering preview.** Deriving `Title` from `Name`, `FamilySize` from `SibSp + Parch`. *Note:* good features often beat fancy models.
14. **A reproducible EDA checklist.** Hand out (or display) a 10-item checklist students should run on every new dataset. *Transition:* "now that we can describe data, we'll spend next week describing data that has a time dimension — where everything gets harder."

---

# Week 2 — Time Series with ARIMA

## Lecture 1 — Time Series Fundamentals

**Narrative goal:** Break the IID assumption. Show that temporal structure changes everything about how we describe, split, and model data.

1. **What makes time series different?** Observations are ordered and dependent. The IID assumption is dead. *Note:* this single fact will reshape preprocessing, validation, and modeling.
2. **Examples of time series.** Stock prices, weather, electricity demand, sales. Show 2–3 plots with strikingly different structures.
3. **The decomposition view.** Trend + seasonality + cycle + residual. Additive vs multiplicative. *Example:* airline passengers — classic multiplicative case.
4. **Trend.** Long-term direction. How to estimate (moving average, regression on time). When trends are misleading.
5. **Seasonality.** Periodic patterns. Daily, weekly, yearly. *Example:* electricity demand has both daily and yearly seasonality stacked.
6. **Cycles vs seasonality.** Cycles are not fixed-period; seasonality is. Business cycles vs winter peaks.
7. **Stationarity — intuition.** A stationary series looks the same statistically regardless of when you look at it. Show stationary vs non-stationary side-by-side.
8. **Stationarity — definition.** Mean, variance, autocovariance constant over time. *Note:* most real series are not stationary; we transform them to be.
9. **Why stationarity matters.** Most classical models (including ARIMA) assume it. Without it, forecasts are unreliable.
10. **Testing for stationarity.** ADF test, KPSS test. How to read the output. *Pitfall:* these tests have low power; visual inspection still matters.
11. **Differencing.** First difference, seasonal difference. When and how. *Example:* show airline passengers before and after `log + diff`.
12. **Autocorrelation function (ACF).** What it measures, how to read it. *Example:* ACF of a random walk vs white noise.
13. **Partial autocorrelation (PACF).** What it adds to ACF. The "direct effect" intuition.
14. **Using ACF/PACF to diagnose structure.** Classic patterns: AR(p) → PACF cuts off, ACF decays; MA(q) → ACF cuts off, PACF decays. *Note:* foreshadow the link to model order selection next lecture.
15. **Train/test splitting for time series.** *Critical pitfall:* never shuffle. Always split temporally. Walk-forward validation as a preview.
16. **A toolkit summary.** What we can now do: load, decompose, test for stationarity, transform to stationarity, characterize via ACF/PACF. *Transition:* "we can describe structure — next lecture, we'll model and forecast it."

## Lecture 2 — ARIMA Modeling

**Narrative goal:** Build the ARIMA model piece by piece, fit it, forecast, and critically evaluate the result.

1. **Recap.** We have stationary (or stationarized) series, and we can read ACF/PACF. Now we model.
2. **AR(p) — autoregressive component.** Model: `y_t = c + φ₁y_{t-1} + … + φ_p y_{t-p} + ε_t`. Intuition: this period depends on the last p periods.
3. **MA(q) — moving average component.** Model: `y_t = c + ε_t + θ₁ε_{t-1} + … + θ_q ε_{t-q}`. Intuition: this period depends on the last q shocks. *Note:* often confusing — it's "moving average of errors," not of the series.
4. **ARMA(p,q).** Combine the two. Useful only for stationary series.
5. **The "I" in ARIMA.** Integration order d — number of differences needed to reach stationarity. ARIMA(p,d,q) generalizes ARMA to non-stationary series.
6. **Choosing (p, d, q).** ACF/PACF inspection, AIC/BIC, auto-ARIMA. *Note:* present the manual method first to build intuition, then mention `pmdarima.auto_arima` as a tool.
7. **Fitting in Python.** `statsmodels.tsa.arima.model.ARIMA`. Show the fit summary and how to read coefficient significance.
8. **Residual diagnostics.** A good ARIMA has white-noise residuals. Check via residual plot, ACF of residuals, Ljung-Box test. *Pitfall:* a low training error with structured residuals = model is missing something.
9. **Forecasting.** Point forecasts vs interval forecasts. *Note:* prediction intervals widen with horizon — show this on a plot.
10. **Evaluating forecasts.** MAE, RMSE, MAPE. When MAPE breaks (near-zero values). sMAPE as alternative.
11. **Walk-forward validation, properly.** Expanding window vs sliding window. Why a single train/test split underestimates uncertainty.
12. **Seasonal ARIMA (SARIMA).** Brief: ARIMA(p,d,q)(P,D,Q)_s for seasonal series. *Example:* airline passengers is the textbook SARIMA case.
13. **Exogenous variables (SARIMAX).** When you have additional predictors. One sentence — they can read up.
14. **What ARIMA can't do well.** Long-horizon forecasts, multiple seasonalities, regime changes, non-linear dynamics. *Note:* this honest accounting matters — sets up why neural approaches exist (W10).
15. **A workflow checklist.** Decompose → test stationarity → difference → identify (p,q) → fit → diagnose residuals → forecast → evaluate. *Transition:* "ARIMA is one tool. From next week we leave time series and enter the much larger family of supervised learning — regression and classification on tabular data."

---

# Week 3 — Regression and Classification Basics

## Lecture 1 — Regression Models

**Narrative goal:** Introduce the supervised learning frame, build linear regression carefully, and introduce the bias-variance tradeoff as a permanent feature of the course.

1. **The supervised learning frame.** We have features X and a target y. Goal: learn f(X) ≈ y. Regression (continuous y) vs classification (discrete y).
2. **Why start with linear regression?** It's the simplest, best-understood, and the foundation on which much else builds.
3. **The model.** `y = β₀ + β₁x₁ + … + β_p x_p + ε`. Intuition: a hyperplane through feature space.
4. **Loss function.** Ordinary least squares — minimize sum of squared residuals. *Note:* why squared? Differentiable, convex, has a closed form. Mention alternatives (MAE) briefly.
5. **Closed-form solution.** `β = (XᵀX)⁻¹Xᵀy`. *Note:* don't dwell on the derivation; emphasize that linear regression has a unique, exact solution — most ML doesn't.
6. **Assumptions of linear regression.** Linearity, independence, homoscedasticity, normality of residuals. *Pitfall:* violating these doesn't break the fit but breaks the inference.
7. **Interpreting coefficients.** "Holding all else equal, a unit change in x_i changes y by β_i." *Note:* this clean interpretability is rare in ML and worth flagging.
8. **Evaluation metrics for regression.** R², adjusted R², MSE, RMSE, MAE. When to prefer which. *Pitfall:* R² always increases with more features — that's what adjusted R² fixes.
9. **Polynomial regression.** Add polynomial features, keep the linear-in-parameters formulation. *Example:* fit degree 1, 3, 10 to noisy data — show the progression.
10. **Underfitting and overfitting — visually.** This is the central plot of the course. Three panels: too simple, just right, too complex.
11. **Bias-variance tradeoff — definition.** Expected error = bias² + variance + irreducible noise. Bias = systematic error from oversimplification; variance = sensitivity to training data.
12. **Bias-variance tradeoff — intuition.** Use the "darts on a board" diagram. *Note:* this concept returns *every single week* — flag that explicitly.
13. **How model complexity drives the tradeoff.** Training error always decreases with complexity; test error is U-shaped.
14. **Train/test split — formal introduction.** Why we need it. *Pitfall:* evaluating on training data is the original sin of ML.
15. **Regularization preview.** Mention Ridge and Lasso — full treatment can wait, but state the idea: penalize complexity. *Transition:* "now to classification, where the target is discrete and we need new tools."

## Lecture 2 — Classification Models

**Narrative goal:** Pivot to classification, introduce two simple but instructive algorithms, and build a deep familiarity with classification metrics.

1. **From regression to classification.** Discrete target. Why we can't just use linear regression directly (unbounded outputs, no probabilistic interpretation).
2. **Logistic regression — motivation.** We want outputs in [0, 1]. Squash a linear combination through a sigmoid: `σ(z) = 1/(1+e⁻ᶻ)`.
3. **The model.** `P(y=1|x) = σ(βᵀx)`. Decision boundary is linear.
4. **Loss function.** Cross-entropy / log loss. *Note:* derive briefly from maximum likelihood. No closed form — uses iterative optimization.
5. **Interpreting logistic regression.** Coefficients as log-odds ratios. Less intuitive than linear regression but still interpretable.
6. **Multi-class logistic regression.** One-vs-rest, softmax. One slide, no derivation.
7. **k-Nearest Neighbors — motivation.** Sometimes the "model" is just "look at neighbors." Non-parametric, lazy learning.
8. **The k-NN algorithm.** For each test point, find k nearest training points, vote (classification) or average (regression).
9. **Choosing k.** Small k → high variance; large k → high bias. *Note:* the bias-variance tradeoff appears again — it always does.
10. **Distance metrics.** Euclidean default; mention Manhattan, cosine. *Pitfall:* k-NN is extremely sensitive to feature scaling — must standardize.
11. **Curse of dimensionality.** In high dimensions, "nearest" loses meaning. *Note:* foreshadow PCA (W7).
12. **Confusion matrix.** TP, FP, TN, FN. Build the full table on the board.
13. **Accuracy and when it lies.** *Example:* 99% accuracy on a 99/1 imbalanced dataset is trivial. Class imbalance returns from W1.
14. **Precision, recall, F1.** Definitions, when each matters. *Example:* medical diagnosis (recall matters), spam filtering (precision matters).
15. **ROC and AUC.** Threshold-independent evaluation. How to read an ROC curve. AUC interpretation.
16. **Precision-recall curves.** When PR is more informative than ROC (heavy imbalance).
17. **Per-class metrics in multi-class.** Macro vs micro vs weighted averaging. *Pitfall:* default "accuracy" hides per-class failures.
18. **A classifier evaluation checklist.** Confusion matrix → per-class metrics → ROC/PR if probabilistic → calibration check (mention only). *Transition:* "logistic regression and k-NN are simple baselines. Next week we look at a much more powerful classifier, the SVM — and we learn how to tune any model properly."

---

# Week 4 — SVMs and Model Tuning

## Lecture 1 — Support Vector Machines

**Narrative goal:** Build SVMs from the max-margin intuition, introduce the kernel trick as a conceptual breakthrough, and connect to bias-variance.

1. **Recap and motivation.** Logistic regression draws *a* boundary. SVMs ask: of all separating boundaries, which is *best*?
2. **The max-margin idea.** Choose the hyperplane that maximizes the distance to the nearest training points. *Note:* this is a different philosophy — focus on the boundary, not the probability.
3. **Support vectors.** The few training points that touch the margin are all that matter. Everything else is irrelevant. *Note:* this sparsity is a feature, not a bug.
4. **The hard-margin SVM.** Geometric formulation. Show the optimization visually before any math.
5. **Hard-margin math (light).** Maximize `2/||w||` subject to `y_i(wᵀx_i + b) ≥ 1`. Convex QP. *Note:* don't derive the dual on the slides; flag it as something curious students can look up.
6. **The soft-margin SVM.** Real data isn't linearly separable. Introduce slack variables and the C parameter.
7. **The C parameter.** Large C → fewer margin violations but smaller margin (high variance). Small C → more violations, wider margin (high bias). *Note:* bias-variance again, controlled by a hyperparameter.
8. **Non-linear classification — the problem.** Show a dataset that's clearly not linearly separable (XOR, concentric circles).
9. **The feature map idea.** What if we project to a higher-dimensional space where the data *is* linearly separable? Show the 2D-to-3D classic.
10. **The kernel trick.** We never need to compute the feature map explicitly — only inner products. `K(x, x') = φ(x)·φ(x')`. *Note:* this is the conceptual centerpiece — spend time on it.
11. **Common kernels.** Linear, polynomial, RBF (Gaussian), sigmoid. Show the decision boundary each produces on the same dataset.
12. **RBF kernel deep-dive.** `K(x, x') = exp(-γ||x - x'||²)`. The γ parameter: small γ → smooth boundary (high bias); large γ → wiggly boundary (high variance). *Note:* same bias-variance story, two controlling knobs now (C and γ).
13. **SVMs in practice.** Strengths: high-dim data, clear margins, kernel flexibility. Weaknesses: don't scale to huge n, no native probabilities (Platt scaling exists), sensitive to scaling.
14. **Multi-class SVMs.** One-vs-rest, one-vs-one. Brief.
15. **When to reach for SVM.** Medium-sized datasets, high-dim features (text, images before deep learning). *Transition:* "we now have multiple hyperparameters (C, γ, k, polynomial degree…). How do we choose them honestly? That's the next lecture."

## Lecture 2 — Cross-Validation and Hyperparameter Tuning

**Narrative goal:** Instill validation discipline. By the end of this lecture, students should feel physical discomfort at the thought of test-set contamination.

1. **The problem.** We have hyperparameters. We want to choose them. We cannot use the test set, which must remain pristine.
2. **The naïve split.** Train/test isn't enough — if we tune on test, we've leaked. Train/validation/test.
3. **The problem with a single validation set.** Small datasets → high variance in validation estimates → we choose hyperparameters that happened to work on one split.
4. **k-fold cross-validation.** Split the *training* data into k folds, train on k-1, validate on 1, rotate. Average. *Note:* typical k = 5 or 10.
5. **Stratified k-fold.** For classification with imbalanced classes — preserve class proportions in each fold.
6. **Leave-one-out CV.** k = n. Low bias, high variance, expensive. When it's worth it.
7. **CV for time series.** *Pitfall:* never use standard k-fold on time series — you'd train on the future. Use expanding-window or sliding-window CV. Back to W2's warning.
8. **The nested CV problem.** If you tune *and* evaluate via CV on the same folds, you're optimistic. Nested CV is the rigorous answer; in practice we accept the optimism for large datasets.
9. **Grid search.** Exhaustive over a grid of hyperparameters. Cost = #combinations × k folds × training time. *Example:* SVM with C ∈ {0.1, 1, 10} and γ ∈ {0.01, 0.1, 1} = 9 × 5 = 45 fits.
10. **Random search.** Often more efficient than grid for high-dim hyperparameter spaces (Bergstra & Bengio 2012). One slide of intuition.
11. **Bayesian optimization.** One slide — mention `Optuna`, `scikit-optimize`. They can explore later.
12. **The data leakage problem — the big one.** Any preprocessing that uses information from outside the training fold leaks. *Examples:* scaler fit on full data, target encoding without CV, feature selection on full data, oversampling before split.
13. **Pipelines as a defense.** `sklearn.pipeline.Pipeline` chains preprocessing and modeling. When wrapped in CV, each fold gets its own pipeline fit. *Note:* if you take one habit from this course, take this one.
14. **Pipeline example.** Build one: `StandardScaler → SVC`. Wrap in `GridSearchCV`. Show the code.
15. **`ColumnTransformer`.** For datasets where different columns need different preprocessing (numeric vs categorical). Brief.
16. **Reporting honestly.** Always report test-set performance separately from CV results. Confidence intervals from CV folds.
17. **A model-selection checklist.** Hold out test, define pipeline, choose CV scheme, choose search strategy, fit, report CV + test. *Transition:* "we now have a powerful tunable model (SVM) and a rigorous tuning procedure. Next week we leave the geometric/linear world and enter the world of trees — a completely different model family."

---

# Week 5 — Trees and Ensembles

## Lecture 1 — Decision Trees

**Narrative goal:** Introduce a fundamentally different model family — non-parametric, hierarchical, interpretable — and use it to motivate ensembling.

1. **A different philosophy.** SVMs and linear models slice feature space with hyperplanes. Trees slice it with axis-aligned cuts and ask questions sequentially.
2. **A tree as a series of questions.** *Example:* a small Titanic tree — "is sex female? if yes…; if no…". Show it visually.
3. **What's in a node.** A feature, a threshold, and two children. Leaves contain predictions (class or value).
4. **How to grow a tree.** Greedy, top-down: at each node, find the split that best separates the data.
5. **Splitting criteria — classification.** Gini impurity, entropy. Walk through computing Gini on a small example.
6. **Splitting criteria — regression.** Variance reduction / MSE. The leaf prediction is the mean of the training points there.
7. **Information gain.** Why we compare parent impurity to weighted child impurity. *Note:* this is the engine of the entire algorithm.
8. **Handling different feature types.** Numeric (threshold splits), categorical (subset splits or one-hot first). *Pitfall:* one-hot encoding *hurts* tree performance — they handle categoricals natively if the library does.
9. **Stopping criteria.** Max depth, min samples per leaf, min impurity decrease. *Note:* without limits, trees memorize training data — depth = capacity.
10. **Overfitting in trees.** A fully grown tree achieves 0 training error and terrible test error. Show the curve.
11. **Pre-pruning vs post-pruning.** Pre: stop early via hyperparameters. Post: grow fully then prune (cost-complexity pruning).
12. **Interpretability.** A small tree is the most interpretable model in this course. *Note:* you can literally explain a prediction to a non-technical stakeholder.
13. **Limits of interpretability.** A tree with depth 20 is no longer interpretable. Honest accounting.
14. **Strengths of trees.** No scaling needed, handle mixed types, capture non-linearities and interactions automatically, interpretable when shallow.
15. **Weaknesses.** High variance (small data change → very different tree), axis-aligned splits, no smooth boundaries.
16. **The high-variance problem.** Visualize: train two trees on bootstrap samples — they look very different. *Transition:* "if a single tree is high-variance, what if we average many of them?"

## Lecture 2 — Random Forests

**Narrative goal:** Introduce ensembling via bagging, build random forests, and show how aggregation tames variance.

1. **The wisdom of crowds.** Independent estimators averaging is a classical variance-reduction technique. Quick intuition: variance of mean = variance / n (if independent).
2. **Bootstrap aggregation (bagging).** Resample the training data with replacement, train a model on each sample, aggregate (vote or average) for predictions.
3. **Why bagging works on trees specifically.** Trees are high-variance, low-bias. Averaging reduces variance without inflating bias. *Note:* don't bag low-variance models — gains are small.
4. **Out-of-bag (OOB) error.** Each bootstrap sample leaves ~37% of data unused. Those become a free validation set. *Note:* one of the most elegant ideas in ML — flag it.
5. **From bagging to Random Forests.** Bagging trees still produces correlated trees (they all see the dominant features). Decorrelate by also randomly subsampling features at each split.
6. **The two sources of randomness in RF.** Row sampling (bootstrap) + column sampling (per split). Both reduce correlation between trees.
7. **Key hyperparameters.** `n_estimators` (more is better, plateau), `max_features` (sqrt(p) default for classification, p/3 for regression), `max_depth` (often unrestricted in RF), `min_samples_leaf`.
8. **RF doesn't really overfit with more trees.** More trees → more averaging → lower variance. Plateau, not U-curve. *Note:* this is unusual; flag it.
9. **Prediction.** Classification: majority vote (or average probabilities). Regression: mean of predictions.
10. **Feature importance.** Mean decrease in impurity (Gini importance) and permutation importance. *Pitfall:* MDI is biased toward high-cardinality features — prefer permutation importance.
11. **Permutation importance intuition.** "If I randomly shuffle this feature, how much does performance drop?" Model-agnostic, more reliable.
12. **Strengths of RF.** Strong out-of-the-box performance, low tuning burden, OOB validation, feature importance, robust to outliers and feature scaling.
13. **Weaknesses.** Large memory footprint, slower prediction than single tree or linear model, less interpretable than a single tree, can't extrapolate beyond training range.
14. **When to reach for RF.** Tabular data, mixed feature types, when you want a strong baseline with minimal tuning. *Note:* often the right first thing to try on tabular problems.
15. **Connecting back.** Bias-variance: RF reduces variance, retains low bias. The pattern: ensembling is a powerful general tool.
16. **Foreshadowing.** "RF reduces variance by averaging. What if instead we reduced *bias* by sequentially correcting errors? That's boosting — next week." *Transition:* set up the W6 narrative.

---

# Week 6 — Boosting and Advanced Tree Models

## Lecture 1 — Boosting Concepts

**Narrative goal:** Frame boosting as the bias-reducing counterpart to bagging, build AdaBoost intuition, then generalize to gradient boosting.

1. **Recap.** Bagging reduces variance via parallel independent models. Boosting reduces bias via sequential dependent models.
2. **The boosting idea.** Train a weak learner, look at its errors, train another that focuses on those errors, repeat. The ensemble is a weighted sum.
3. **What's a "weak learner"?** A model slightly better than random. Typically shallow trees ("stumps" or depth-3 to depth-6).
4. **AdaBoost — the original.** Reweight misclassified examples upward at each iteration so the next learner focuses on them.
5. **AdaBoost mechanics.** Walk through 2–3 iterations on a toy 2D dataset visually. *Note:* avoid the formula-heavy derivation on slides; the intuition is what matters.
6. **AdaBoost's final prediction.** Weighted vote, where weights depend on each learner's accuracy.
7. **Gradient Boosting — the generalization.** Reframe: at each step, fit a new model to the *residuals* (or gradient of the loss) of the current ensemble. *Note:* AdaBoost is a special case of gradient boosting with exponential loss.
8. **The functional gradient descent view.** We're doing gradient descent in function space. One sentence; don't dwell.
9. **Key hyperparameters.** Number of trees, learning rate (shrinkage), tree depth, subsampling. *Note:* learning rate × n_trees tradeoff — small lr + many trees is the standard recipe.
10. **Early stopping in boosting.** Crucial. Use a validation set to stop when validation loss stops decreasing.
11. **Bias-variance in boosting.** Boosting can overfit (unlike RF). Each iteration reduces bias but eventually starts fitting noise. *Note:* monitor with a validation curve.
12. **AdaBoost vs Gradient Boosting in practice.** GB is more flexible (any differentiable loss), more popular today. AdaBoost is mostly historical/pedagogical.
13. **Loss functions.** Squared error for regression, log loss for classification, quantile loss for quantile regression. *Note:* this flexibility is a key advantage.
14. **Why boosted trees win on tabular data.** Captures interactions, handles mixed types, robust, well-tuned implementations exist. *Foreshadow:* next lecture covers those implementations.
15. **Boosting's failure modes.** Noisy labels (overfits to them), small datasets (variance dominates), poor extrapolation. *Transition:* "the modern boosting libraries — XGBoost, LightGBM, CatBoost — bring engineering to the algorithm. That's next."

## Lecture 2 — LightGBM, XGBoost, CatBoost, and SHAP

**Narrative goal:** Survey the three dominant modern GBT libraries, compare them honestly, and introduce SHAP for principled interpretability.

1. **The landscape.** Three libraries dominate tabular ML competitions and production: XGBoost (2014), LightGBM (2017), CatBoost (2017). All are gradient-boosted trees with engineering refinements.
2. **XGBoost.** Regularized objective (L1, L2 on leaf weights), efficient histogram-based splits, parallelization. *Note:* the library that started the boosting renaissance.
3. **LightGBM.** Leaf-wise tree growth (vs XGBoost's level-wise), GOSS (gradient-based one-side sampling), EFB (exclusive feature bundling). Fastest of the three on large data.
4. **CatBoost.** Symmetric trees, native handling of categorical features via target encoding with ordering tricks, less prone to target leakage from encoding.
5. **Honest comparison.** No universal winner. Rough guide: LightGBM for speed on large data, CatBoost for heavy categorical features, XGBoost when you want the most battle-tested option. Performance often within noise on well-tuned models.
6. **Key shared hyperparameters.** `n_estimators`, `learning_rate`, `max_depth` / `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`, regularization terms.
7. **Library-specific quirks.** LightGBM's `num_leaves` instead of `max_depth`. CatBoost's `cat_features` parameter. XGBoost's `tree_method='hist'`.
8. **A tuning playbook.** Start with library defaults. Increase `n_estimators` with low `learning_rate` and early stopping. Then tune tree complexity. Then regularization. *Note:* don't tune everything at once.
9. **Categorical handling — done right.** LightGBM and CatBoost handle categoricals natively (better than one-hot). XGBoost needs encoding. *Pitfall:* one-hot blows up cardinality and hurts trees.
10. **Handling missing values.** All three handle NaN natively by learning a default direction at each split. *Note:* one fewer preprocessing step.
11. **Speed and scale.** Histogram-based splitting, GPU support, distributed training. Rough numbers: millions of rows in minutes.
12. **The interpretability problem.** A boosted forest of 1000 trees is not interpretable by inspection. We need post-hoc methods.
13. **Feature importance — limits.** Gain, split count, permutation. *Pitfall:* these give a global view only. They don't explain individual predictions.
14. **SHAP — the idea.** Shapley values from cooperative game theory. For each prediction, attribute the deviation from the baseline to each feature, fairly.
15. **SHAP — properties.** Local accuracy (sums to prediction), missingness, consistency. *Note:* the only feature attribution method with these axiomatic guarantees.
16. **SHAP for trees specifically.** TreeSHAP — exact, polynomial-time algorithm for tree ensembles. Fast enough for production.
17. **SHAP visualizations.** Force plots (single prediction), summary plots (global), dependence plots (feature interactions). Show one of each.
18. **When to use what.** Linear model coefficients > tree feature importance > SHAP global > SHAP local. Pick the simplest sufficient explanation.
19. **A modern tabular ML recipe.** EDA → pipeline with `ColumnTransformer` → LightGBM or XGBoost with early stopping → CV-tuned → SHAP for interpretation.

**Project kickoff (≈10 min at the end of lecture).** Now that the classical toolbox is substantial, launch the project as a *running thread*. *Beats:* (a) form groups (≤4); (b) choose a dataset and frame a problem now — no toy datasets, >1000 rows, a clear target and metric; (c) the key idea — *your dataset travels with you*: apply each new method to it as we cover it (clustering/PCA next week, neural nets from W8). *Note:* sell this explicitly — "the most interesting tools are still ahead, and your project is where you'll try them on your own data." (d) the rubric rewards breadth (including an advanced method) plus interpretation, comparison, error analysis, and limitations — so a lone decision tree won't pass. One-page proposal due W9; report due end of W12. *Transition:* "we've spent six weeks on supervised learning. Next week we look at unsupervised learning — what to do when there's no target — and it's the first new tool to fold into your project."

---

# Week 7 — Clustering and Dimensionality Reduction

## Lecture 1 — Clustering

**Narrative goal:** Introduce unsupervised learning, build K-means carefully, and survey alternatives that fix K-means's failure modes.

1. **The unsupervised setting.** No labels. We want to find structure in X alone. Two big families: clustering (group similar points) and dimensionality reduction (compress features).
2. **What clustering is for.** Customer segmentation, document grouping, anomaly detection, exploratory analysis. *Note:* often a stepping stone, not the final goal.
3. **Defining "similar".** Distance metrics. Euclidean dominates but isn't always right (cosine for text, Manhattan for high-dim sparse).
4. **K-means — the algorithm.** Pick K, initialize K centroids, assign points to nearest centroid, recompute centroids, repeat until stable.
5. **K-means as optimization.** Minimize within-cluster sum of squares (inertia). *Note:* converges to a local minimum, not global.
6. **Initialization matters.** Random init can give bad clusters. K-means++ chooses initial centroids smartly. *Note:* always use K-means++ (sklearn default).
7. **Choosing K — the elbow method.** Plot inertia vs K, look for an elbow. *Pitfall:* often there's no clean elbow.
8. **Choosing K — silhouette score.** Per-point measure of how well it fits its cluster vs neighbors. Average silhouette as a K-selection criterion. *Note:* more reliable than elbow.
9. **Choosing K — domain knowledge.** Often the most honest answer is "we want 5 customer segments because the business team has 5 strategies."
10. **K-means assumptions.** Spherical clusters of similar size and density. Violations break it.
11. **K-means failure modes.** Non-spherical clusters (concentric rings), unequal density, very different sizes. Show pathological examples.
12. **DBSCAN — motivation.** Density-based: clusters are dense regions separated by sparse ones. No need to specify K.
13. **DBSCAN — algorithm.** ε (neighborhood radius), minPts (min points for a dense region). Core points, border points, noise.
14. **DBSCAN strengths.** Handles arbitrary shapes, identifies noise, no K needed. Weaknesses: sensitive to ε, struggles with varying density.
15. **Hierarchical clustering — overview.** Agglomerative (bottom-up) and divisive (top-down). Produces a dendrogram.
16. **Linkage criteria.** Single, complete, average, Ward. Different linkages produce different clusters.
17. **Dendrograms.** Reading them, cutting at a height to choose K. *Note:* gives a multi-scale view K-means doesn't.
18. **Evaluating clusters without labels.** Silhouette, Davies-Bouldin, Calinski-Harabasz. *Pitfall:* all are heuristics — there's no ground truth.
19. **When labels exist (semi-supervised).** Adjusted Rand index, normalized mutual information.
20. **A clustering playbook.** Scale features → try K-means and DBSCAN → evaluate by silhouette → inspect clusters by feature → interpret. *Transition:* "to even compute meaningful distances, we often need fewer dimensions — enter PCA."

## Lecture 2 — PCA and Feature Engineering

**Narrative goal:** Build PCA from the variance-maximization view, contrast feature selection vs extraction, and equip students with practical feature engineering instincts.

1. **The curse of dimensionality, formalized.** In high dim: distances concentrate, data becomes sparse, models overfit. *Note:* this is why dim reduction matters.
2. **Two paths to fewer features.** Feature selection (keep a subset of original features) vs feature extraction (build new features from combinations).
3. **PCA motivation.** Find directions in feature space along which the data varies most. Project onto them.
4. **PCA geometrically.** Center the data, find the direction of maximum variance, then the next orthogonal direction of max variance, etc.
5. **PCA mathematically.** Eigenvectors of the covariance matrix (or SVD of the centered data matrix). The principal components.
6. **Variance explained.** Each PC explains some fraction of total variance. Scree plot. "Keep enough PCs to explain 95%" — common heuristic.
7. **Standardization before PCA.** *Critical pitfall:* PCA is scale-sensitive. Always standardize unless features share the same units.
8. **Interpreting PCs.** Each is a linear combination of original features. Loadings tell you which features matter. *Note:* often hard to interpret semantically — that's the price.
9. **PCA for visualization.** 2D or 3D projections of high-dim data. *Example:* Iris in 2 PCs — clusters separate visibly.
10. **PCA for preprocessing.** Dim reduction before clustering, classification, or regression. Can improve performance, especially on small data.
11. **PCA limitations.** Linear, sensitive to outliers, hard to interpret, supervised information ignored.
12. **Beyond PCA — brief mentions.** Kernel PCA (non-linear), t-SNE and UMAP (visualization only, not for preprocessing). *Note:* show one striking t-SNE plot.
13. **Feature selection — wrapper methods.** Forward selection, backward elimination, recursive feature elimination. Expensive.
14. **Feature selection — filter methods.** Univariate statistics (correlation, mutual information, chi²). Cheap, no model needed.
15. **Feature selection — embedded methods.** L1 regularization (Lasso) zeros out coefficients. Tree feature importance.
16. **Feature engineering — the craft.** Often more impactful than model choice. Examples: ratios, differences, log transforms, interactions, time-based features (day of week, hour).
17. **Domain-driven features.** *Example:* on Titanic, `Title` (Mr/Mrs/Miss) extracted from `Name` is more predictive than raw name. Domain knowledge wins.
18. **Automated feature engineering.** Mention `featuretools` and similar. Useful but no substitute for thinking.
19. **A feature engineering checklist.** Understand domain → derive ratios/aggregates → encode categoricals smartly → handle dates → log-transform skewed numerics → interact features when domain suggests it.
20. **The wrap-up.** We can now find structure (clusters) and reduce dimensions (PCA). With these, EDA and supervised modeling become much more powerful. *Transition:* "this concludes the 'classical' half of the course. Next week, midterm 1, then we enter neural networks."

---

# Week 8 — Midterm 1 and Neural Network Fundamentals

## Lecture 1 — Midterm 1

Pen-and-paper, 2h, one recto-verso sheet allowed. Scope: W1–W7 (EDA, ARIMA, regression, classification, SVM, CV, trees, ensembles, clustering, PCA). No neural networks.

## Lecture 2 — Neural Network Fundamentals

**Narrative goal:** Pivot from classical ML to neural networks. Build the multi-layer perceptron from the perceptron up, make backpropagation feel inevitable, and address overfitting concretely.

1. **Why neural networks now?** Classical methods (especially boosted trees) dominate on tabular data. Neural networks dominate on perceptual data (images, audio, text) and at scale. *Note:* honest framing — they're not always better.
2. **The biological inspiration — and how to drop it.** Neurons, dendrites. *Note:* the analogy is mostly historical. Treat NNs as a class of parametric functions, not as brain simulators.
3. **The perceptron.** A single artificial neuron: weighted sum + bias + step activation. *Example:* draw it. Show it can learn AND, OR.
4. **The XOR problem.** A single perceptron can't learn XOR. This nearly killed neural nets (Minsky & Papert 1969). *Note:* a great cautionary tale about capacity.
5. **Multi-layer perceptron.** Stack neurons in layers. Hidden layers add capacity. MLPs are universal approximators (Cybenko 1989) — *but* this guarantees existence, not learnability.
6. **Activation functions.** Why we need non-linearity (linear stacks of linears are linear). Sigmoid, tanh, ReLU. *Note:* ReLU dominates today — explain why (no saturation on positive side, fast).
7. **Architecture choices.** Input layer (= feature count), output layer (= task: 1 for regression, K for K-class classification with softmax), hidden layers and widths (design choice).
8. **The forward pass.** Walk through one example. Matrix multiplications + activations. *Note:* this is just function evaluation — nothing mysterious yet.
9. **Loss functions, revisited.** MSE for regression, cross-entropy for classification (same as logistic regression — flag this continuity).
10. **The learning problem.** Find weights that minimize the loss. No closed form. Need iterative optimization.
11. **Gradient descent.** Compute the gradient of loss w.r.t. weights, step in the negative direction. *Example:* show a 2D loss landscape.
12. **Learning rate.** Too small → slow; too large → diverge or oscillate. The single most important hyperparameter. *Note:* foreshadow LR schedules in W9.
13. **Stochastic and mini-batch gradient descent.** Compute gradients on a batch, not the full dataset. Trade-offs: noisy gradients (regularizing effect), efficient on GPUs.
14. **Backpropagation — intuition.** Apply the chain rule layer by layer, backward from the loss. *Note:* don't derive on slides; explain visually with a computation graph.
15. **Backpropagation — why it's a big deal.** Same complexity as the forward pass. This is what made deep networks practical.
16. **Autograd.** Modern frameworks (PyTorch, TensorFlow) compute gradients automatically. We write the forward pass; backward comes for free.
17. **The overfitting problem in NNs.** NNs have many parameters — they easily memorize training data. Training loss → 0, validation loss → up.
18. **Mitigations preview.** Regularization (L2 weight decay), dropout, early stopping, data augmentation. Full treatment in W9.
19. **A first MLP recipe.** Define architecture → choose loss → forward → backward (auto) → SGD update → monitor train/val curves. *Transition:* "this is the skeleton. Next lecture, we add the techniques that make NNs actually work — and then we apply them to images."

---

# Week 9 — Training Techniques and CNNs

## Lecture 1 — Training Techniques, Transfer Learning, and Regularization

**Narrative goal:** Equip students with the practical bag of tricks that make NN training work, then introduce transfer learning as the most important practical pattern.

1. **Recap.** We can build and train an MLP. It overfits, trains unstably, and barely beats logistic regression on tabular data. Let's fix that.
2. **The optimization landscape problem.** Non-convex, ill-conditioned, with saddle points everywhere. Vanilla SGD struggles.
3. **Momentum.** Accumulate a velocity vector. Smooths SGD, escapes shallow minima faster. *Note:* simplest meaningful upgrade.
4. **Adam.** Adaptive learning rates per parameter, combines momentum and RMSprop. *Note:* sensible default; you can almost always start with Adam.
5. **AdamW vs Adam.** Decoupled weight decay. Subtle but matters — flag for advanced students.
6. **Learning rate schedules.** Constant, step decay, cosine annealing, warmup. *Example:* show a typical training curve with and without LR decay.
7. **Batch normalization.** Normalize activations within each mini-batch. Speeds up training, allows higher learning rates, mild regularization effect. *Note:* explain at a high level — the *why* is still debated.
8. **Layer normalization.** Mentioned for completeness — used in transformers (W10).
9. **Regularization — L2 weight decay.** Penalize large weights. Reduces variance.
10. **Regularization — Dropout.** Randomly zero activations during training. Prevents co-adaptation; acts like training an ensemble of sub-networks. *Note:* turn off at test time.
11. **Early stopping.** Monitor validation loss; stop when it stops improving. *Note:* the simplest, most reliable regularizer.
12. **Data augmentation.** For images: rotations, crops, flips, color jitter. *Note:* effectively expands the training set; standard for vision.
13. **Initialization matters.** Xavier/Glorot, He initialization. *Note:* random init from N(0, 0.01) doesn't scale to deep networks.
14. **Gradient pathologies.** Vanishing gradients (sigmoids deep in a network) and exploding gradients (RNNs). *Note:* foreshadow why ReLU and skip connections matter.
15. **The transfer learning idea.** Pre-train on a large dataset, fine-tune on yours. *Why it works:* lower layers learn general features that transfer.
16. **Pre-trained models.** ImageNet-pretrained CNNs (ResNet, EfficientNet), BERT for text. Available in `torchvision`, `transformers`.
17. **Fine-tuning strategies.** Feature extraction (freeze the backbone, train only the head) vs full fine-tuning (unfreeze everything with a small LR). When to use which.
18. **Why this is huge in practice.** On small datasets, transfer learning beats training from scratch by huge margins. *Note:* this is the standard, not the exception, in modern computer vision and NLP.
19. **A training recipe.** Adam(W) → cosine schedule with warmup → batch norm → dropout → data augmentation → early stopping → pretrained backbone when possible. *Transition:* "now we apply all this to a specific architecture designed for images — CNNs."

## Lecture 2 — CNN Fundamentals

**Narrative goal:** Build the convolutional neural network from the limitations of MLPs on images, introduce the key operations, and end at modern architectures.

1. **Why MLPs are bad at images.** A 224×224 RGB image → 150,528 inputs. Fully connected first layer has 150k × 1000 ≈ 150M parameters. Massive overfitting, ignores spatial structure.
2. **Two key priors for images.** Locality (nearby pixels relate) and translation invariance (a cat in the corner is still a cat). MLPs encode neither.
3. **The convolution operation.** Slide a small kernel over the image, compute dot products. *Example:* show a 3×3 edge detector applied to an image.
4. **Convolutions in detail.** Kernel size, stride, padding. *Note:* padding preserves spatial dims; stride downsamples.
5. **Parameter sharing.** The same kernel is applied everywhere. *Note:* this is the source of CNN's efficiency — millions fewer parameters than an MLP.
6. **Learnable kernels.** Don't hand-design them — learn them via backprop. *Note:* this is the conceptual breakthrough.
7. **Feature maps.** A convolution layer with K kernels produces K feature maps. Each detects something different.
8. **Multiple input channels.** RGB → kernel has depth 3. Generalize to deeper layers: input has C channels, kernel is k×k×C.
9. **Pooling.** Max pooling, average pooling. Downsample spatially, add small-translation invariance.
10. **The CNN block.** Conv → activation (ReLU) → (optional pool). Stack these.
11. **Hierarchical features.** Early layers learn edges, middle layers learn parts (eyes, wheels), late layers learn objects. *Example:* show Zeiler & Fergus visualizations.
12. **A complete CNN.** Conv blocks (extract features) → flatten → fully connected (classify). Show LeNet-5 as the canonical first example.
13. **AlexNet (2012).** The ImageNet moment. ReLU, dropout, GPU training, data augmentation. *Note:* the model that triggered the deep learning revolution.
14. **VGG.** Stack 3×3 convs. Simple, deep, lots of parameters. Mostly historical now.
15. **ResNet — skip connections.** "Identity shortcuts" let very deep networks train. *Note:* the second key architectural breakthrough after convolution itself.
16. **Modern architectures, briefly.** EfficientNet (compound scaling), Vision Transformers (next week). *Note:* CNNs are no longer the only choice, but still dominant for many tasks.
17. **Practical CNN training.** Use a pretrained backbone, fine-tune. Custom architectures rarely beat tuned ResNets/EfficientNets out of the box.
18. **Beyond classification.** Object detection (YOLO, Faster R-CNN), segmentation (U-Net). One sentence — the CNN core is the same.
19. **A CNN recipe for the exercise.** Small CNN → BN → dropout → augmentation → train → compare to pretrained ResNet18 fine-tuned. *Transition:* "CNNs exploit spatial structure. But sequences — text, time series — need a different architecture. That's next week."

---

# Week 10 — Sequences and Transformers

## Lecture 1 — RNNs and LSTMs

**Narrative goal:** Introduce sequence modeling, build the RNN, expose its limitations, and motivate LSTMs.

1. **The sequence modeling problem.** Inputs and/or outputs are sequences of variable length. Text, audio, time series, video. *Note:* this is fundamentally different from fixed-size inputs.
2. **Why MLPs and CNNs don't fit.** No notion of order, no memory across time, fixed input size.
3. **The RNN idea.** Maintain a hidden state. At each time step, update the hidden state from the input and previous hidden state. Emit an output.
4. **RNN math.** `h_t = tanh(W_h h_{t-1} + W_x x_t + b)`, `y_t = W_y h_t + b_y`. *Note:* same weights applied at every step — parameter sharing across time.
5. **Unrolling an RNN.** Visualize the computation graph unrolled across T time steps. *Note:* this makes backprop visible — "backpropagation through time" (BPTT).
6. **RNN use cases.** One-to-many (caption generation), many-to-one (sentiment classification), many-to-many (translation, time series forecasting).
7. **Training RNNs.** BPTT — apply backprop on the unrolled graph. Truncated BPTT for long sequences.
8. **The vanishing/exploding gradient problem.** Gradients chain across T steps. Multiplied → vanish (if <1) or explode (if >1). *Note:* this is the central technical problem of RNNs.
9. **Why RNNs forget.** Vanishing gradients → can't learn long-range dependencies. In practice, vanilla RNNs handle ~10–20 steps, not 100s.
10. **LSTM motivation.** Add a separate memory cell that information can flow through unchanged.
11. **LSTM architecture.** Forget gate, input gate, output gate, cell state. *Note:* don't derive on slides — show the diagram, explain each gate's job.
12. **Why LSTMs work.** Gates learn to preserve or update memory. The cell state's additive updates avoid the vanishing gradient.
13. **GRU.** Simpler than LSTM (two gates instead of three), similar performance. *Note:* often a default choice today over LSTM.
14. **Bidirectional RNNs.** Run one RNN forward, one backward, concatenate. *Note:* requires the full sequence — not for streaming/forecasting.
15. **Stacked RNNs.** Multiple RNN layers. Deeper hierarchy. Common in practice.
16. **Sequence-to-sequence (encoder-decoder).** Encoder RNN reads input, decoder RNN generates output. Translation, summarization. *Note:* this sets up the attention story next.
17. **RNN limitations even with LSTMs.** Sequential — can't parallelize over time. Still struggle with very long contexts. *Transition:* "the bottleneck of seq2seq — compressing everything into a single hidden state — led to attention, and then to transformers."

## Lecture 2 — Transformers and Attention

**Narrative goal:** Introduce attention from the seq2seq bottleneck, build self-attention, and assemble the transformer.

1. **The seq2seq bottleneck.** The decoder gets only the encoder's final hidden state. For long inputs, this loses information.
2. **The attention idea (Bahdanau 2014).** At each decoding step, look back at *all* encoder hidden states, weighted by relevance.
3. **Attention as weighted retrieval.** Query (current decoder state), Keys (encoder states), Values (encoder states). Compute similarities (query · keys), softmax to get weights, weighted sum of values.
4. **The QKV abstraction.** *Note:* this triple is the conceptual core of everything that follows. Spend time here.
5. **Attention solves the bottleneck.** No information squeezed through a single vector. Also: attention weights are interpretable — show example.
6. **From attention to self-attention.** What if queries, keys, and values all come from the *same* sequence? Each token attends to every other token.
7. **Self-attention math.** Q = XW_Q, K = XW_K, V = XW_V. Output = softmax(QKᵀ / √d_k) V. *Note:* the √d_k scaling is a stability trick, not a deep idea.
8. **Why self-attention is powerful.** Direct connections between any two positions (no vanishing gradient over distance). Parallelizable. Captures long-range dependencies natively.
9. **Multi-head attention.** Run self-attention multiple times in parallel with different projections. Each head can specialize.
10. **Positional encoding.** Self-attention is permutation-invariant — we need to inject position info. Sinusoidal encodings or learned. *Note:* small detail, large consequence.
11. **The transformer block.** Multi-head self-attention → residual + LayerNorm → feed-forward → residual + LayerNorm.
12. **The original transformer (Vaswani 2017).** Encoder-decoder for machine translation. Show the architecture diagram.
13. **Encoder-only models — BERT.** Pretrain with masked language modeling. Fine-tune for classification, NER, QA. *Note:* this is the standard for understanding tasks.
14. **Decoder-only models — GPT.** Pretrain with next-token prediction. Generative. *Note:* the family behind modern LLMs.
15. **Why transformers won.** Parallelism (vs RNNs), long-range modeling, scale with data and parameters. The scaling laws era.
16. **Transformers beyond text.** Vision Transformers (ViT), audio, protein folding, time series. *Note:* tie this back — foundation models cross domains.
17. **Pretrained transformers in practice.** HuggingFace ecosystem. Two lines of code to fine-tune BERT for classification.
18. **What transformers don't fix.** Quadratic complexity in sequence length, hunger for data, opacity. *Note:* honest accounting, as always.
19. **A modern sequence recipe.** For text: pretrained transformer + fine-tune. For time series: classical first (ARIMA), then ML (XGBoost on lag features), then NN if data is large. *Transition:* "we've covered the major model families. Next week: midterm 2, and you launch into projects."

---

# Week 11 — Midterm 2 and Project Kickoff

## Lecture 1 — Midterm 2

Pen-and-paper, 2h, one recto-verso sheet allowed. Scope: W8–W10 (neural network fundamentals, training techniques, CNNs, RNNs/LSTMs, transformers/attention).

## Lecture 2 — Project Clinic & Methodology

**Narrative goal:** Projects have been running since W6 — this is the mid-flight methodology deep-dive that turns work-in-progress into rigorous final reports. Reinforce discipline and de-risk common pitfalls right before the final push.

1. **Where you should be.** Proposal is in (due W9); you have a dataset, a baseline, and a few models tried. Today is about making the rest rigorous. *Note:* this is a clinic, not a kickoff — the toolbox already met your problem weeks ago.
2. **Logistics recap.** Groups of ≤4. ~25 reports total. Report worth 20% of the grade. Due end of W12 — before the exam period.
3. **What a good project looks like.** Clear problem statement, appropriate dataset, defensible methodology, honest evaluation, interpretable conclusion.
4. **What a bad project looks like.** Throwing 10 models at a dataset, picking the one with the best test score, calling it done. *Note:* warn explicitly.
5. **Choosing a problem.** Three sources: (a) provided datasets/scenarios, (b) student-chosen with approval, (c) extensions of case studies (W12). Encourage option (a) for groups that want certainty.
6. **Dataset criteria.** Sufficient size (>1000 rows typically), reasonably clean, license-clear, problem statement matches the data.
7. **Avoiding common dataset pitfalls.** Toy datasets (MNIST, Iris) — don't use them for the project. Heavily Kaggle-overfit datasets — risky. Tiny datasets — high-variance results.
8. **Problem framing — the most important step.** Define inputs, target, evaluation metric *before* modeling. *Pitfall:* changing the metric mid-project to look better.
9. **The methodology backbone.** EDA → preprocessing pipeline → baseline (always start with one) → progressively more complex models → CV-evaluated → final test set.
10. **The baseline rule.** Always include a stupid baseline (predict majority class, predict mean). *Note:* if your fancy model barely beats it, that's important to report.
11. **The progression rule.** Don't start with the most complex model. Start with logistic regression / linear regression / ARIMA. Add complexity only when justified.
12. **Validation discipline reminder.** Test set untouched until the final evaluation. CV for hyperparameter selection. Pipelines for leak-proof preprocessing. *Note:* W4's lecture is in scope for projects.
13. **Reproducibility.** Set random seeds. Pin package versions. Use a Git repo. *Note:* graders will run your code. *Cluster reminder (brief):* you've used it since W1 (access + interactive sessions), W4 (CPU batch jobs), and W9 (GPU) — the project runs there too. No new infrastructure to learn; just scale up the workflow you already know.
14. **Report structure suggestion.** Problem → Data → Methods → Results → Interpretation → Limitations → Conclusion. 8–15 pages typically.
15. **Interpretation matters.** A report that says "XGBoost got 0.87 AUC" is incomplete. Why did it work? Which features mattered? Where does it fail?
16. **Limitations matter more.** Stating what doesn't work and why is a sign of rigor, not weakness.
17. **Milestones (recap).** W6: groups + dataset + problem locked. W9: one-page proposal (problem, dataset, metric, baseline plan). W11 (now): baseline + at least one non-baseline result, advanced method underway. End of W12: report due — before the exam period.
18. **Common failure modes I'll be looking for.** Data leakage, test set contamination, missing baselines, no error analysis, p-hacking on metrics.
19. **Office hours and help.** When and where. *Note:* encourage early questions — last-week panic is the most common failure mode.
20. **The next two project sessions.** Hands-on time during W12 case-study lectures — use it. *Transition:* "to make the methodology concrete, the final week walks through two complete case studies — structured data and image data."

---

# Week 12 — Case Studies and Wrap-Up

## Lecture 1 — Case Study 1: Structured Data

**Narrative goal:** Walk through an end-to-end project on a real tabular dataset, modeling the methodology you want students to follow.

1. **The setup.** Real dataset — Icelandic Meteorological Office weather data (or alternative if access falls through). State the prediction task explicitly (e.g., next-day precipitation, temperature regression).
2. **Why this dataset.** Real, messy, mixed types, temporal structure. Not a Kaggle-cleaned toy.
3. **Step 1: Problem framing.** Inputs, target, evaluation metric. Justify the metric choice. *Note:* model the discipline.
4. **Step 2: EDA.** Show the actual code and plots. Missing values, distributions, correlations, target distribution. *Note:* narrate what you're looking for at each step.
5. **Findings from EDA.** Specific observations: this feature has 30% missing, this one is highly skewed, target is imbalanced if classification. Surface them explicitly.
6. **Step 3: Preprocessing pipeline.** Build a `ColumnTransformer` for numeric + categorical + temporal features. Show the code.
7. **Step 4: Baseline.** A constant predictor or linear model. Quote the metric. *Note:* "this is what we have to beat."
8. **Step 5: First real model.** Linear regression / logistic regression. Cross-validated. Compare to baseline.
9. **Step 6: Tree-based model.** Random Forest with default hyperparameters. Cross-validated. Compare.
10. **Step 7: Boosted model.** LightGBM with sensible defaults + early stopping. Cross-validated. Compare.
11. **Step 8: Tuning.** Grid or random search on the winning model. Show the search space and the result.
12. **Step 9: Test-set evaluation.** Single number. *Note:* "this is the honest answer."
13. **Step 10: Interpretation.** SHAP plots. Which features drive predictions? Does it match domain expectations?
14. **Step 11: Error analysis.** Where does the model fail? Conditional metrics by season, by region, by feature value range.
15. **Step 12: Limitations.** Data quality issues, distribution shift risk, missing features, generalization concerns.
16. **What the project should look like.** Map each step to what the project report should contain. *Note:* make the parallel explicit.
17. **Hands-on time.** Use the remaining lecture time for students to start on their own projects. Walk around.

## Lecture 2 — Case Study 2: Images, Wrap-Up, and the Toolbox

**Narrative goal:** Apply the same methodology to a perceptual problem, then zoom out to consolidate the entire course.

1. **The setup.** Small image dataset — pieces on a mechanical-arm assembly line (or alternative). Classification task: identify piece type.
2. **Why this dataset.** Small (forcing transfer learning), real-world (noisy lighting, varied angles), industrially relevant.
3. **Step 1: Problem framing.** Classes, metric (per-class accuracy or macro-F1 for class balance).
4. **Step 2: EDA for images.** Show samples per class, class distribution, image size distribution, lighting/quality variation. *Note:* image EDA is visual, not statistical.
5. **Step 3: Preprocessing and augmentation.** Resize, normalize. Augmentation strategy: which augmentations make domain sense?
6. **Step 4: Baseline.** Logistic regression on raw pixels or HOG features. *Note:* even on images, start dumb.
7. **Step 5: A small CNN trained from scratch.** Probably overfits — show that explicitly. Validation loss diverging from training loss.
8. **Step 6: Pretrained backbone, feature extraction.** Freeze a ResNet18, train only the classifier head. Show the improvement.
9. **Step 7: Pretrained backbone, fine-tuning.** Unfreeze, small LR, more augmentation. Show the further improvement.
10. **Step 8: Tuning.** Learning rate, augmentation strength, head architecture. Limited search.
11. **Step 9: Test-set evaluation.** Confusion matrix per class. Where do errors cluster?
12. **Step 10: Interpretation.** Grad-CAM or similar — what is the model looking at? *Note:* visual interpretability is a luxury — use it.
13. **Step 11: Error analysis.** Which classes confuse? Are errors due to ambiguity, label noise, or model limitation?
14. **Step 12: Trade-offs.** Speed vs accuracy. Model size vs deployability. *Note:* in industrial settings, these trade-offs often dominate.
15. **From case studies to projects — explicit advice.** What to copy, what to adapt, what pitfalls you saw.
16. **Wrap-up Part 1 — the toolbox.** Present the toolbox summary slide: every method seen, organized by data type and task type.
17. **Wrap-up Part 2 — the decision flowchart.** "If your data is tabular and labeled → start with linear/logistic, then GBT. If unlabeled → clustering + PCA. If sequence → ARIMA then transformer. If image → pretrained CNN." Walk through it.
18. **Wrap-up Part 3 — the one-pager.** Hand out (or display) a single-page summary mapping every use case seen in the class to the methods that solved it.
19. **The methodological core.** Distill: EDA → baseline → simple model → progressively more complex → CV-tuned → interpret → state limitations. The methodology outlasts any specific algorithm.
20. **What we didn't cover.** Reinforcement learning, generative models (GANs, diffusion), graph neural networks, MLOps. Point to resources.
21. **Further learning.** Books (ESL, ISLR, Deep Learning by Goodfellow et al.), courses (fast.ai, CS231n, the HuggingFace course), competitions (Kaggle).
22. **The closing remark.** "ML is 70% data work, 20% methodology, 10% choice of algorithm. The methods change; the discipline doesn't." *Transition:* "questions, and then project work time."

---

# Final Exam Prep Notes (not a lecture — appendix for your reference)

The final is 3h, pen-and-paper, two recto-verso sheets, two sections:
- **Section A — Classical ML:** scope of midterm 1 (W1–W7). Likely emphasis: bias-variance, CV discipline, when to use which model, interpreting metrics, ARIMA/ACF reading, ensemble methods.
- **Section B — Neural networks and variants:** scope of midterm 2 (W8–W10). Likely emphasis: backprop intuition, CNN architecture choices, RNN limitations, self-attention mechanics, transfer learning.

Suggested exam question types: short-answer conceptual ("explain why we standardize before PCA"), interpretation ("read this ACF plot — what ARIMA order?"), pseudo-code ("write a training loop with early stopping"), and one or two longer applied scenarios ("you're given this dataset and constraint — design a pipeline").
