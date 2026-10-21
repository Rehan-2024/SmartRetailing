> Historical V2 specification. The current source exclusions, nine-model comparison, reused-test disclosure and deployment regression guard supersede the corresponding sections below. See [the current audit](DATA-AUDIT-AND-ROBUSTNESS.md).

# Version 2: decision and evaluation contract

## Purpose and scope

The core product is a product–country decision tool and a new-product analogue scenario planner. The user receives three prominent results: candidate countries, 28-day units, and revenue arithmetic at the entered unit price. Detailed diagnostics are isolated in Model Lab. No generic 0–100 success score is invented.

Raw data → merchandise cleaning → canonical product/country series → chronological model comparison → generated public market artifact → deterministic browser decision logic. Static hosting on Vercel serves `public/data/market.json`; Python training is offline. The browser runs no remote AI service. Excel reading is loaded only when needed. Original data and private uploads are not sent to the hosting service.

## Data correctness

Concatenate both original workbook sheets and deduplicate on the original fields, before adding provenance. Strip stock-code strings, parse dates, separate cancellation invoices/negative quantities and nonpositive prices. Positive merchandise sales require quantity and price >0, valid dates and non-cancellation invoices. Explicit service codes and descriptions remove postage, fees and adjustments from merchandise workflows. Overall Version 1 accounting totals differ from merchandise-only Version 2 totals by design.

StockCode is the product identity. The latest description labels it without splitting the same code across names. Country is the recorded country/territory label, not a claim of regulatory eligibility. Customer IDs are not published; anonymous valid sales remain. Return unit share = absolute return quantity / (positive sold units + absolute return quantity), not matched-refund probability. Recorded revenue is quantity × unit price in GBP.

## Existing-product 28-day forecasts

Build daily unit matrices for each canonical product–country pair and zero-fill unrecorded days. Zero-fill assumes zero recorded sales, not zero underlying demand. Eligibility at each origin: at least 84 days since first observed sale and at least 10 distinct invoice orders in the preceding 84 days.

Five fixed candidates:

1. Last 28 recorded days repeated as a 28-day total.
2. Last 84-day daily average scaled to 28 days.
3. Damped last/previous-28-day trend, ratio clipped to 0.5…1.5.
4. HistGradientBoostingRegressor on log1p 28-day units, 160 iterations, max 15 leaves, L2=5.
5. HistGradientBoostingRegressor with Poisson loss on units, 160 iterations, max 15 leaves, L2=10.

The feature matrix uses trailing 7-, 28-, previous-28- and 84-day units; active sales days; age of the series; initial observed pair price; and origin position within an annual cycle. The price feature is initial observed price, not a future full-history median. Calendar-origin offsets are relative to dataset start. Log quantities/prices are used where specified by `feature_matrix`.

Training examples occur every 28 days; every training label horizon ends before the first validation origin. The models are fitted once on training data. Two forward 28-day validation origins select the candidate by total absolute unit error. A third window estimates pooled 10th/90th quantiles of (actual−prediction)/(prediction+1). A fourth, later window is the final chronological test. The selected method uses the most recent features for the final forecast without refitting the estimator after final testing. Validation-selected baselines and trained models compete under identical eligibility rules.

Display MAE (units per eligible pair), WAPE (sum absolute error / sum actual units) and RMSE in Model Lab. WAPE is **not accuracy** and 1−WAPE is not called accuracy. Bands use the calibration quantiles scaled by prediction+1, clipped at zero. Report empirical final-test coverage against the nominal 80%; coverage is not guaranteed for future periods or individual products. Actual/predicted plots show the first 200 final-test pairs, not a selected set of successful forecasts.

## New-product analogue matching

Inputs: product description, planned price in the active currency, and a feasible observed market filter. A fixed list of product-type families (for example mug/cup, bag/tote) filters candidates when the query contains a recognized type. This prevents colour/pattern words from matching a different recognized product type. It is an explicit vocabulary guard, not a semantic language model or inferred taxonomy. Unrecognized types fall back to lexical similarity, visibly named as such.

TF–IDF uses lowercase ASCII word tokens of length ≥2 and the declared stopword list. Fit vocabulary and IDF on existing product descriptions only. Cosine similarity uses L2-normalized sparse vectors. Price closeness = exp(−abs(log(reference price / entered price))). Reference catalogue price is median positive recorded unit price for the benchmark and average realized unit price for a private upload. Combined similarity = cosine × price closeness. Retain cosine ≥0.15, combined >0.05. Neither number is confidence or probability.

The number of top neighbours (5, 8 or 12) is selected on historical validation cohorts. Exact existing-description queries exclude the identical existing product. Candidate products require at least 84 days of recorded history and ten orders across their recorded markets in the previous 84 days. At least three matches are required. Each candidate country additionally requires three active matched products and ten total orders in the trailing 84 days. Only active contributing analogues enter the country estimate. No absent country or missing analogue outcome is converted into fictional evidence.

Before a scenario appears, the user must inspect the proposed products and explicitly confirm the comparison set. They can remove accessories and products that are not commercial substitutes. The retrospective benchmark measures the automatic proposal algorithm; it does not validate a human-modified scenario. This distinction is visible beside the confirmation controls and in the exported brief. Lexical matching and a type vocabulary are not a semantic-equivalence guarantee.

Country unit scenario = similarity-weighted mean of each contributing analogue’s preceding 84-day units /3. Revenue scenario = units × entered unit price. This assumes comparable distribution, availability and exposure. No elasticity, campaign lift, profit, competition, market size, logistics cost or causal launch impact is estimated. The analogue spread is min/max contributing rates, not a statistical prediction band.

## New-product historical benchmark

Use cutoffs 2010-06-01, 2010-10-01, 2011-03-01 as validation and 2011-07-01 as final test. Targets are product codes first observed within 28 days after cutoff. Analogue catalogue contains established products with ≥84 days of prior history and ≥10 orders in the previous 84 days. No target sales outcomes enter its matching or rate estimation. Query metadata uses its first recorded description and price, treating these as attributes supplied before launch.

Outcomes cover the first 28 recorded days per target product. Select neighbour count by validation country-hit count divided by all validation targets, counting abstentions as zero. On supported final-test cases report top-country hit rate, support coverage and unit MAE for the sum of supported-country scenario totals versus actual total units. Country baseline is the globally leading country by prior-84-day units, evaluated on exactly the same supported cases. Unit baseline is the median established-product 28-day-equivalent total. A reliable first-country winner is allowed only with ≥20 supported test cases and a strict improvement over the country baseline. Otherwise the UI discloses no demonstrated improvement and labels rankings as unvalidated pilot comparisons.

First observation is not a verified product launch; product codes may represent relistings, and never-sold products are absent. First recorded descriptions/prices may differ from actual prelaunch plans. The UK-heavy retailer can make an always-UK baseline difficult to beat; that is dataset exposure bias, not evidence that every company should launch in the UK. Retrospective evaluation is not prospective commercial validation. The test cohort is small, and no significance or universal launch-success claim is made.

## Private uploads

CSV parsing handles commas/tabs, quoted delimiters and embedded newlines. XLSX reads the first worksheet with a self-hosted, read-only reader. Require OrderID, ProductCode, ProductName, ISO date, nonzero quantity, positive unit price, Country. Keep returns separately and deduplicate business fields. Reject malformed dates, missing columns and mixed declared currencies. Limits are 10 MB/50,000 rows. Users explicitly activate the validated workspace. Product/country exploration and comparison then use that file, never the public benchmark.

Private catalogues fit TF–IDF locally. Scenario neighbour count is fixed at 8, not claimed to have been tuned on the uploaded population. Trailing recorded rates and evidence checks match the shared browser algorithm; at least 84 days of company history are required. Uploaded scenario rankings remain unvalidated. Existing-product forecast fields stay unavailable because no private retraining/backtest occurs. Model Lab continues to display clearly labeled public-benchmark results. Refresh clears the private state.

## Boundaries and acceptance

The auxiliary session, marketing, wholesale and geography populations remain separate. The old conversion simulator is removed from the business interface; its historical results remain an explicitly separate academic reference. Classifier accuracy does not validate a product-market tool.

Acceptance: new-product scenarios show contributing records or abstain; price/revenue arithmetic reconciles; no unsupported country appears; exact browser/Python matching parity passes; product IDs/country pairs are unique; chronological labels do not overlap selection/test; uploaded records actually replace explorer data; workspace switches clear stale results; build and dependency audit pass; browser checks cover navigation, launch forms, filters and CSV/XLSX switching.
