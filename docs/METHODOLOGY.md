# Version 1 archive: architecture and evaluation contract

This file documents the original general analytics dashboard. The current product and evaluation contract is `V2-METHODOLOGY.md`; use that document for launch planning, product/country workflows, uploads and the current forecast models. Version 1 reports remain only as separately labeled academic references.

## Product and scope

Users are retail analysts, managers and project reviewers. Their questions are: how did sales perform; which customers differ; who has not returned; what products co-occur; can completed sessions be classified; how do wholesale accounts spend; what simple forecast is defensible?

Ten navigation views expose these questions and their evidence. Downloads contain the generated report. The session explorer is a real local model calculation, not a fabricated response or external AI call. No action automatically sends a campaign, changes inventory or promises improved revenue.

The prior Streamlit baseline is replaced with a static browser application because the requested target is Vercel. Python remains the analysis/training layer. This preserves the modular-monolith principle without an always-on Python service or training during serverless requests. Vanilla JavaScript and SVG avoid a dependency-heavy front end and ship no third-party scripts or remote fonts. The limits are snapshot data, no authenticated live uploads, and no persistent live inference API.

## Data flow and contracts

Original files (immutable Bronze inputs) → per-source cleaning and feature frames (in-memory Silver) → aggregates, evaluations and browser coefficients (`public/data/report.json`, Gold) → browser views. Silver data is recomputed, not committed. DuckDB and Parquet are deferred because a single offline batch already fits this workload; adding unused storage would create no user benefit.

Each source has a path, byte size, SHA-256, source row count, field list and missing-cell count. Retail missing-cell count is measured after exact deduplication. Source dates are historical. The report records build time separately from the source period. JSON serialization rejects NaN and converts unavailable diagnostics to null.

Customer data from unrelated sources is never linked. The extra datasets enrich separate views rather than creating an invented unified customer population. No customer-level records are published.

## Retail sales and quality

Read both workbook sheets and concatenate. Drop exact duplicate rows across all original columns. Parse dates. Identify cancellation invoices by uppercase C prefix and returns by negative quantity. Keep sales only when not cancelled, quantity >0, price >0, and date is valid. Revenue = quantity × price. Sales without customer IDs contribute to sales and basket analysis but not RFM/cohorts.

Gross sales do not subtract returns. Signed return value includes negative quantities/cancellation rows, measured separately after deduplication, and can overlap other exclusion reasons. It is not a reconciled net-accounting measure. Distinct valid invoices are orders; AOV = gross sales / orders. Exact duplicate removal is an explicit research assumption: indistinguishable repeated legitimate lines cannot be disambiguated without a line ID. Missing descriptions are excluded from product-name grouping by pandas but can still contribute to gross totals.

## RFM, segmentation and wholesale

Snapshot = midnight after the last valid retail date. Recency is days since last purchase, frequency is distinct valid invoices, monetary is positive-sales GBP. Apply log1p then StandardScaler. Fit KMeans for k=2…6, seed 42, n_init 10. Highest silhouette selects k (deterministic sample up to 2,000 rows). Davies–Bouldin measures compactness/separation. Cross-seed adjusted Rand index compares seed 42 and seed 7 fits. This checks initialization sensitivity, not bootstrap stability or future drift.

Wholesale uses the same procedure on six spending features, excluding Region/Channel. Original-unit means describe each cluster. Labels are arbitrary. PCA projects the standardized features into two dimensions for display; up to 700 anonymous points are published. PCA does not train the cluster model. No classification accuracy or causal effectiveness is claimed for unsupervised groups.

## Retention

Each customer belongs to the first observed purchase month. A cohort cell is unique purchasing customers at month age / cohort's month-zero customers. Inactivity >90 days is a descriptive rule at snapshot time. No supervised churn labels exist. Cohorts are left-censored; unobserved future months are blank, observed months with no purchases are zero. Partial acquisition/final months can bias rates.

## Basket analysis

One invoice is one binary product set. Count singles and unordered pairs exactly; prune products below the minimum joint-support threshold, which cannot remove a qualifying pair. Minimum count = max(20, ceil(0.005 × baskets)); confidence ≥0.2; lift >1. Emit both directional rules, order by lift, publish at most 80. This is exact pair counting rather than FP-Growth: it solves the bounded two-product use case without unneeded long itemsets. Support = joint count / baskets; confidence = joint / antecedent count; lift = confidence / consequent prevalence. These are in-sample descriptive association measures, not held-out recommendation performance.

## Purchase classifier

Target is Revenue. Exact full-row duplicates are removed before a stratified 60%/20%/20% train/validation/test split with seed 42. PageValues is excluded because its availability at an actionable pre-purchase cutoff is unverified. Revenue is never a feature. Other fields are session-complete measurements; the classifier is a benchmark, not an early-session predictor. No timestamps are provided for forward temporal validation.

Numeric columns use train-fitted median imputation and standardization. Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend use train-fitted categorical mode imputation and one-hot encoding with unknown categories ignored. Columns have no missing values in the current source, but transformations are explicit. Models: prior-probability DummyClassifier, LogisticRegression(max_iter=1500), RandomForestClassifier(160 trees, max_depth=12, min_samples_leaf=8). Hyperparameters are fixed, not searched on test.

Select between logistic regression and random forest by validation average precision. Select one decision threshold on validation F1 over 0.05…0.95 in 0.01 steps. Apply that threshold to all test rows (including baseline). Do not refit after selection. Report test ROC AUC, AP, precision, recall, F1, accuracy, Brier loss and a 2×2 confusion matrix (actual rows, predicted columns; negative then positive). Baseline AP is class prevalence. Accuracy alone is misleading with class imbalance. No cross-validation confidence intervals are provided.

Permutation importance uses four held-out test permutations scored by average precision as a post-selection diagnostic; it is not used to revise the model. Reliability bins summarize test probabilities; they do not fit or establish calibration. Business costs are unknown, so validation F1 is a declared project choice rather than an optimal commercial threshold.

The browser exports **the evaluated logistic regression**, regardless of the validation winner. Its standardized numeric and categorical coefficients produce sigmoid(intercept + dot(encoded inputs, weights)). Four test examples verify parity with sklearn to 1e-10. The selected classifier's threshold is also used in this explorer and labeled accordingly; the score remains a logistic probability, not the random forest score. Forest results are an evaluated comparison, not a browser forest inference endpoint.

## Forecasting

Aggregate positive-sales daily GBP, fill missing dates with zero. Four expanding-window origins near the end each predict 28 days. Candidates are repeating the final seven days and eight-week weekday means. First three windows select the rule by mean MAE; fourth is an untouched final test. Report per-window MAE and aggregate MAE, RMSE, WAPE across all four, clearly labeled aggregate backtest rather than pure test metrics. The final test MAE is separately visible. Final 28-day outlook uses the chosen rule fitted to all observed history. Seasonal weekly baselines are defensible starting points; no untested sophisticated model is implied. No prediction intervals, promotion features, holiday adjustment or future drift guarantee. Dates follow 2011 source history, not the current date; the last recorded day may be partial.

## Supplementary datasets

Marketing: count, mean spend per Mnt field, Response prevalence, missing Income. No response model is trained without a defensible campaign cutoff. Store: aggregate regional Sales/Profit and count Year versus Sales Date discrepancies. It has incompatible temporal fields, unverified currency/provenance, and is never joined with transaction customers. Geographic data provides coverage, not demand or site-success evidence. No ranking of new store locations is defensible from this alone.

## Source attribution and external definitions

- Chen, D. (2012). **Online Retail II**, UCI Machine Learning Repository, DOI https://doi.org/10.24432/C5CG6D. Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii. UCI identifies unit prices as sterling and C-prefixed invoices as cancellations. Its CC BY 4.0 attribution is retained here for the derived retail summaries.
- **Online Shoppers Purchasing Intention Dataset**, UCI: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset. UCI defines Page Value with reference to transactions. Excluding it is a conservative availability decision for this implementation, not proof that every use of this field constitutes leakage.
- **Wholesale customers**, UCI: https://archive.ics.uci.edu/dataset/292/wholesale+customers. The supplied file is analyzed independently; its annual spending units are not converted to GBP.
- The marketing, store and geography files are user-supplied; an authoritative original URL/license was not supplied or verified. Their provenance remains an explicit limitation. Raw records are not redistributed.

## Operations, security and failure behavior

No user secrets, remote scripts, tracking, model pickle downloads or raw customer records are required. Browser input stays local. Escape source strings before HTML rendering. Numeric inputs validate before prediction. Content Security Policy restricts scripts and network fetches to self; inline styles are allowed for quantitative graphics, never inline scripts. Serve only `public`/`dist`, never the dataset parent directory. Server path validation rejects traversal.

Training fails if required files/columns are missing rather than substituting fake data. UI shows a report-loading failure if JSON cannot load. Build fails without the expected report. Source data changes require explicit retraining and a new report commit. Reproducibility is source fingerprints + code commit + pinned library versions + seed; fingerprint alone is not a code version.

## Acceptance and known limits

All navigation routes render, filters are scoped explicitly, export downloads the real report, model explorer matches Python, retail totals reconcile, forecast partitions are time ordered, source boundaries remain visible, and the static build succeeds. Tests verify these numerical contracts; browser QA checks rendering and interaction separately. Public deployment should be treated as a historical research product. No live retail integration, upload/retrain flow, login, business uplift claim, learned location model, or proven prospective churn model is included.
