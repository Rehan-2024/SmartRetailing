# SmartRetail Project Progress Report and Viva Guide

Prepared for academic review on 26 October 2026. This report describes the implementation and saved experiments at project revision 634dcc7. It separates completed software from research claims that still need validation.

## 1 Problem statement

Retailers need to decide which products to stock, how much they may sell in the next few weeks, and which observed markets deserve a small new-product pilot. A sales dashboard can describe the past, but a decision also needs a forecast, relevant comparable products, sufficient supporting records, and an explanation of uncertainty.

Our project investigates how real retail transactions can support two tasks: forecasting the next 28 days of recorded unit sales for established product-country pairs, and constructing explainable historical country scenarios for a product with no sales history. The second task is a cold-start problem because the proposed product has no target history of its own.

The practical objective is to help a retailer compare evidence and plan a pilot. It is not to guarantee launch success. Country demand depends on inventory, advertising, distribution, competition and other variables that the supplied data does not contain.

## 2 What we have built

SmartRetail is a working product and launch decision studio with an offline Python ML pipeline and an interactive web application. The current implementation contains seven workflows:

- Plan a launch: enter a description and planned price, inspect and explicitly confirm comparable products, then inspect supported country scenarios for units and revenue.
- Product Explorer: search a stable product code and compare recorded sales, returns and supported historical unit forecasts by country.
- Market Explorer: inspect products in an observed country with a minimum-order evidence filter.
- Compare and Export: shortlist product-country pairs and export the evidence.
- Your Data: read and validate a private CSV or XLSX in the browser and switch to an isolated workspace.
- Model Lab: inspect candidate errors, calibration coverage, case breakdowns and the new-product country benchmark.
- Data and Methods: inspect cleaning rules, source fingerprints, formulas and unsupported claims.

The project originally contained a broader collection of analytics. We narrowed the business workflow around product and launch decisions. Separate session-conversion and customer-clustering experiments remain available in the full notebook; their results are not used as evidence that the launch planner is accurate.

Two Colab notebooks are complete. The presentation notebook has 8 sections, 12 executed code cells and 3 charts. It trains the existing model and a challenger alongside a simple baseline. The full notebook has 21 executed code cells and 10 charts, covering the wider experiments. Both contain genuine saved outputs.

## 3 Dataset choice and authenticity

### Main dataset

We use Online Retail II from the UCI Machine Learning Repository. UCI describes it as real transactions from a UK-based online retailer selling giftware, with many wholesale customers. It covers 1 December 2009 through 9 December 2011 and contains 1,067,371 line-item records. A record is an invoice item, not a separate customer or complete order. The dataset is distributed under CC BY 4.0. [UCI Online Retail II](https://doi.org/10.24432/C5CG6D)

The eight original fields in our workbook are Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID and Country. Price is in GBP. Invoice identifies an order; StockCode identifies a product; Quantity and InvoiceDate provide the observations from which historical features and future targets are constructed.

This dataset fits our prototype because it combines dated repeat transactions, product identity, product text, price and recorded customer country. We can build sales histories, compare products and construct retrospective evaluations using one coherent source. Its weaknesses are equally important: it is old, comes from one retailer, is UK-heavy and has no verified product-launch dates or records of products that never sold.

### Other verified datasets

Online Shoppers Purchasing Intention contains 12,330 sessions and 17 explanatory attributes plus the Revenue target. We remove 125 exact duplicates, leaving 12,205 sessions for a separate classification benchmark. Wholesale Customers contains 440 clients with six spending categories and Channel and Region fields; we use the spending variables for descriptive clustering. Both are independent UCI datasets under CC BY 4.0. [UCI Online Shoppers](https://doi.org/10.24432/C5F88Q) and [UCI Wholesale Customers](https://doi.org/10.24432/C5030X)

These sources represent unrelated populations. We do not join their customer identifiers or use a session-conversion score as a product-launch success probability.

### Authenticity checks and exclusions

We downloaded the official UCI archives and compared SHA-256 hashes of their members with the supplied local files. All three accepted files matched exactly. Training rejects different file bytes. The audit records original URLs, attribution, licenses and hashes. A matching hash proves identity with the published file, not that every transaction is error-free or representative of current demand.

Campaign, store-sales, city and pincode files have unverified exact-file provenance and licenses. At the user's request they are archived in Drive and the CSVs are also in the repository, but they are excluded from all model calculations. Unverified does not mean proven synthetic. No synthetic, oversampled or augmented training rows are used. Tiny artificial software-test fixtures and the example upload template do not enter model training.

## 4 Data preparation and prediction target

We read both workbook sheets, remove exact duplicate original rows, parse timestamps, standardize product codes, and separate cancellations and returns from positive sales. Fees, postage, adjustments and other non-merchandise codes are excluded from product workflows. We retain usable anonymous sales for product analysis, but customer-level RFM analysis requires a customer identifier.

| Data check | Recorded result |
| --- | --- |
| Original invoice-item rows | 1,067,371 |
| Exact duplicates removed | 34,335 |
| Usable positive merchandise-sales rows | 1,003,117 |
| Merchandise product codes | 4,886 |
| Recorded country labels | 43 |
| Product-country pairs observed at least once | 28,275 |
| Usable merchandise rows without customer ID | 226,715 |

We aggregate to daily product-country unit sales and fill unrecorded days with zero recorded sales. That is a modeling assumption: zero recorded sales could reflect no demand, no inventory or no distribution. We cannot identify the cause with this dataset.

The regression target is the sum of positive merchandise units in the following 28 days for one product-country pair. It is recorded gross unit sales, not unconstrained market demand, profit or net revenue. A 28-day horizon gives a four-week planning period and makes successive evaluation windows comparable. It is a design choice, not a proven optimal horizon.

Eligibility requires at least 84 days of observed history and ten order occurrences in the preceding 84 days. Order occurrences are invoice-unique daily counts summed across the window; they are not guaranteed to be distinct customers or distinct invoices across all days. The final historical test contains 1,975 eligible pairs, not every pair in the catalogue.

## 5 Features and model training

The existing model has eight numerical features: log-transformed sales totals over the preceding 28 days, previous 28 days, preceding 84 days and preceding 7 days; active sales days in the preceding 84 days; observed pair age; log-transformed initial price; and annual seasonal position.

The richer models use 20 features. Additional inputs describe earlier sales, maximum and variable daily volume, days since the latest sale, recent trend, preceding order counts, units per order occurrence, annual-lag sales and seasonal sine/cosine terms. All history ends before the forecast origin. Annual history is used only when it exists. Initial observed price replaces a future full-history median to avoid price leakage.

We form 30,205 training examples from eligible pairs at historical origins. Each example has its own past-only features and following-28-day target. Raw transaction rows are aggregated into supervised examples; therefore one million line items does not mean one million independent training examples.

The existing forecast method is scikit-learn HistGradientBoostingRegressor fitted to log1p(unit sales), with max_iter 160, max_leaf_nodes 15, l2_regularization 5 and random_state 42. Predictions use expm1 and are clipped at zero. These settings cap complexity and regularize the fit; they are not claimed to be globally optimal. The estimator also uses its library early-stopping behavior where applicable.

The model pools examples across product-country histories. Product IDs and country names are not categorical model inputs in this implementation. Each pair is distinguished by its own history, age and initial price. Consequently, we do not claim to have learned a causal effect of a country's identity.

### Why histogram gradient boosting

We chose a tree-based tabular method because the inputs are engineered numerical history features and interactions can be nonlinear. Histogram binning makes split search practical for many training examples; boosting adds trees sequentially to reduce remaining loss. The log target reduces the influence of very large sales totals, although it can understate large orders and is not an unbiased estimate of the arithmetic mean after inverse transformation. [scikit-learn histogram gradient boosting](https://scikit-learn.org/1.8/modules/ensemble.html#histogram-based-gradient-boosting)

This is a reasonable modeling choice, not proof that it is universally best. Our final comparison shows only a modest MAE advantage over the baseline and worse RMSE. Domain-based feature descriptions and visible baselines explain the workflow; we have not implemented forecast SHAP explanations or established causal feature effects.

### Alternatives and what was actually tested

We tested nine methods in total: last-28-days sales, an 84-day daily mean, a damped trend, base-feature log-unit boosting, base-feature Poisson boosting, history-feature absolute-error boosting, history-feature log-unit boosting, history-feature Poisson boosting and history-feature extra trees.

ARIMA or SARIMA could provide explicit time-series benchmarks. Croston-type methods could be useful for intermittent demand. Linear or Poisson regression could provide simpler parametric baselines. LSTM or Transformer models could be considered with a careful temporal design. XGBoost, LightGBM or CatBoost are other tree libraries. These are future candidates, not methods whose superiority we have demonstrated. Model choice must be decided by a fair chronological comparison, not by the popularity or complexity of the name.

## 6 Evaluation design

Training targets end before the first validation window. The first two following 28-day windows compare candidate absolute errors; the third calibrates empirical bands; the fourth measures errors on the previously inspected final period.

| Role | Forecast period | Eligible pairs |
| --- | --- | --- |
| Validation 1 | 20 Aug to 16 Sep 2011 | 1,660 |
| Validation 2 | 17 Sep to 14 Oct 2011 | 1,817 |
| Calibration | 15 Oct to 11 Nov 2011 | 1,861 |
| Reused historical test | 12 Nov to 9 Dec 2011 | 1,975 |

The final period has been inspected during earlier project development. We therefore label it a reused historical benchmark, not an untouched independent test. A separate release guard retains the existing model if the validation winner worsens either known-benchmark WAPE or RMSE. This deployment decision uses inspected benchmark results; it is not pure validation-only selection.

Three earlier seasonal runs separately train and choose candidates from their own preceding periods, then report a following period. They provide historical stress evidence, not a new external test. A feature-boundary test changes all future quantities and orders and checks that earlier features stay unchanged. This checks a specific leakage boundary, not every possible source of leakage or bias.

## 7 Actual model results

MAE = mean absolute unit error. RMSE = square root of mean squared unit error. WAPE = sum of absolute unit errors divided by sum of actual units, multiplied by 100. Lower values are better. WAPE is undefined for a slice with zero total actual sales; we use MAE there. We never report 100 minus WAPE as accuracy.

<!--MODEL_TABLE_START-->
| Method | Validation MAE | Test MAE | Test WAPE percent | Test RMSE |
| --- | --- | --- | --- | --- |
| Last 28 days | 108.46 | 120.23 | 45.54 | 286.98 |
| 84-day mean | 104.87 | 131.33 | 49.74 | 351.28 |
| Damped trend | 132.53 | 142.20 | 53.86 | 348.25 |
| Existing log boosting | 100.24 | 118.33 | 44.81 | 357.34 |
| Base Poisson boosting | 105.59 | 132.76 | 50.28 | 394.66 |
| History absolute error | 94.37 | 129.47 | 49.04 | 386.27 |
| History log boosting | 94.82 | 123.11 | 46.63 | 372.81 |
| History Poisson boosting | 110.34 | 131.69 | 49.88 | 368.38 |
| History extra trees | 106.15 | 117.04 | 44.33 | 342.51 |
<!--MODEL_TABLE_END-->

All test errors above refer to the same reused 28-day period and eligible pairs. The existing model's MAE is 118.33 units versus 120.23 for repeating the previous 28 days, about a 1.6% relative reduction. Its WAPE is 44.81% versus 45.54%. Its RMSE is worse, 357.34 versus 286.98, indicating a weakness on large mistakes. These results do not justify a high-accuracy claim.

The history absolute-error challenger lowers validation MAE from 100.24 to 94.37, about 5.9%. In the later known benchmark it worsens MAE to 129.47, WAPE to 49.04% and RMSE to 386.27. We did not promote it. Extra trees achieves a lower error in that final period but loses the validation comparison; selecting it just because that period looks better would cherry-pick the known test.

The challenger's three earlier seasonal runs reduce MAE versus last-28-days by about 24.2%, 10.0% and 6.3%. Those improvements do not erase the later failure. The existing model helps intermittent-sales MAE, 37.02 versus baseline 50.02, but large recent-volume pairs worsen to 295.61 versus baseline 290.97. Non-UK MAE improves to 45.42 versus 58.77, but only 88 non-UK pairs are eligible in the final test; generalization remains limited.

Calibration uses scaled residuals from the separate calibration period and their 10th and 90th percentiles. The pooled band targets 80% and contains 80.20% of final-period outcomes. That is empirical coverage, not 80% model accuracy or a guarantee for every product. Coverage for zero-subsequent-sales cases is only 22.22%, which shows why pooled coverage must not hide case failures.

## 8 New products and country scenarios

A genuinely new product has no lagged sales history, so the established-product regression model cannot directly solve that case. The launch workflow instead fits TF-IDF on historical product descriptions and compares a query with known products. TF-IDF weights informative words; cosine similarity measures their textual overlap. It is not deep semantic understanding. Recognized product-type families prevent some cross-type matches, and an analyst must explicitly confirm the comparables.

The matching score is cosine similarity multiplied by exp of minus the absolute log price ratio. Similar words and closer prices increase the score. The price term is a proximity heuristic, not a learned price-elasticity model. The full benchmark compares 5, 8 and 12 neighbours; earlier cohorts choose 8 using a country-hit objective that counts abstentions.

Each supported country needs at least three active confirmed analogues and ten preceding-84-day order occurrences. Unit scenario is a similarity-weighted mean of their active 84-day unit rates scaled to 28 days. Revenue scenario is units multiplied by the entered price. The displayed analogue range is the minimum and maximum comparable-product rates, not a confidence interval. Unseen countries and unsupported queries receive no estimate.

The retrospective country experiment uses products first observed after a cutoff, with analogue history taken only from before the cutoff. It compares the first 28 recorded days. First appearance is not a verified commercial launch; never-sold products are absent. The first observed description and price are assumed to be metadata available before a real decision.

In the final catalogue-entry cohort, 26 of 28 products are supported, about 92.86% coverage. Top-country hit rate is 84.62%, equal to the recent-sales-leading-country baseline on exactly those 26 cases. Thus country ordering has not demonstrated added predictive value. Unit MAE is 434.12 versus baseline 610.26 on supported cases, but the selected neighbour count was chosen for country hits, not unit errors. No causal launch-demand or launch-success claim follows.

## 9 Supporting academic experiments

The separate session classifier compares a prevalence baseline, logistic regression and random forest. After deduplication, its stratified split contains 7,323 training, 2,441 validation and 2,441 test sessions. Preprocessing fits training data only; PageValues is excluded because its availability at an actionable pre-purchase cutoff is unverified. Validation average precision selects the forest, and validation F1 selects an approximately 0.20 threshold.

The forest reports ROC AUC 0.784, average precision 0.377, F1 0.423 and accuracy 72.59%. The majority baseline achieves 84.35% accuracy but zero positive recall at the evaluated threshold. This illustrates class imbalance: accuracy alone can reward never predicting a purchase. The random split has no reliable session timestamps and does not establish future performance. These results do not validate country recommendations.

RFM and wholesale KMeans use log1p, standardization, k values 2 through 6 and silhouette selection; both choose two groups. Davies-Bouldin and cross-seed adjusted Rand scores are internal checks. PCA is a display projection. Cluster IDs are not ground-truth customer types, and silhouette is not classification accuracy. Ninety-day inactivity is descriptive, not a supervised churn prediction. Basket rules measure co-occurrence using support, confidence and lift, not proven cross-sell uplift.

## 10 System architecture and progress

The architecture is: verified files -> Python cleaning and aggregation -> feature creation and model training -> evaluated JSON artefacts -> browser application. Python uses pandas, NumPy, scikit-learn and openpyxl; plots use matplotlib. The web interface uses HTML, CSS and JavaScript. The build bundles the Excel reader and prepares static files for Vercel.

The web app serves precomputed historical ML forecasts. It does not train a model on every page request, and no live HGB prediction API is implemented. New query text is transformed in the browser against the exported text representation. Browser uploads stay in the active tab; they are not mixed with the public benchmark and disappear on refresh. Uploads support explorers and analogue scenarios, but do not automatically retrain or apply the HGB forecast to a new company's records.

Completed work includes the source audit, real-data pipelines, nine-method comparison, case and seasonal checks, seven business workflows, exports, browser upload validation, two executed notebooks, GitHub versioning, Drive organisation and Vercel build configuration. The previous checks passed 16 JavaScript tests, three Python boundary tests and the static build. The presentation notebook's 12 cells also executed and reproduced the three displayed model errors.

Remaining work includes a new independent evaluation period, recent company records, verified launch and failed-launch outcomes, broader market coverage, inventory and promotion variables, improved large-order forecasts, reliable country ranking and business pilot validation. Authentication, persistent storage, automatic retraining, monitoring and a live forecast-serving backend are not implemented. Vercel deployment configuration is ready; this report does not claim that a live deployment URL has been verified.

Our accurate completion statement is: the scoped research prototype and demonstration workflow are implemented. Commercial forecast accuracy, reliable country selection and production operation remain unvalidated. We do not assign a completion percentage without a defined rubric.

## 11 Suggested demonstration for sir

Start with the problem statement and the distinction between established-product forecasting and new-product cold start. Open the presentation notebook and show source verification, cleaning counts, chronological partitions and the actual fit calls. Explain the baseline before presenting the model errors. Show the worse RMSE and the failed challenger as findings, not items to hide.

Then open the app: choose a recorded product in Product Explorer, inspect country evidence, show the launch form and explicitly review analogues, and export a comparison. Finish with Model Lab and Data and Methods. If presenting uploads, explain that the workspace changes but the public ML benchmark is not retrained. If there is no live site, use the local preview or saved screenshots and describe deployment as configured rather than completed.

For the bag example in the presentation notebook, the selected recorded shopping-bag codes produce historical country scenarios. They demonstrate the calculation and assume comparable exposure. They are not a forecast for an actual 2026 launch. The simple notebook uses three explicitly chosen bag analogues; the full app uses its neighbour-count rule followed by the user's confirmation. Do not claim that this small example reruns the full catalogue-entry evaluation.

## 12 Opening presentation script

Sir, our project is SmartRetail, an explainable product and launch decision tool. The problem is that past-sales charts alone do not tell a retailer how many units an established product may sell next month or how to investigate a new product with no sales history.

We used verified real UCI transactions from one UK retailer. We cleaned duplicate and invalid records, built product-country histories and trained gradient-boosting regressors using only past information. We compared nine methods and kept a simple baseline so that any improvement is measurable.

The existing model achieves a 28-day historical-test MAE of 118.33 units and WAPE of 44.81%. Its improvement over baseline is modest, and its RMSE is worse. A richer model improved validation but failed the later benchmark, so we retained the existing model. We explicitly disclose that this test period has already been inspected.

For new products we use visible text-and-price analogues, analyst confirmation and country evidence thresholds. The country hit rate ties the baseline, so we present pilot scenarios rather than a reliable launch-first verdict. The application, source audit and executed notebooks are complete as a prototype; independent recent data and real pilot outcomes are the next research steps.

## 13 Viva questions and model answers

### 1 What is your project in one sentence

SmartRetail is an explainable prototype that combines historical product-country sales forecasts with analogue-based new-product pilot comparisons, while exposing the data, baseline errors and limits of its evidence.

### 2 How is it more than a dashboard

It includes trained forecast models, future-target evaluation, comparable-product selection, evidence thresholds, unsupported-query abstention, private workspace switching and decision exports. The charts support those workflows rather than being the whole deliverable.

### 3 What exactly do you predict

For established products, positive recorded units over the following 28 days for one eligible product-country pair. We do not predict profit, total market demand or launch success. A new product receives an analogue scenario because it has no lagged history.

### 4 Why did you choose this dataset

It provides repeat dated transactions, stable product codes, descriptions, prices and recorded countries in one real, open-licensed source. Those fields fit our methods. Its age and single-retailer exposure limit current-market conclusions.

### 5 Is your dataset synthetic

The three accepted files exactly match official UCI originals, and the main source is documented as real transactions. No generated rows enter training. Four other supplied files remain unverified archives and are excluded from results.

### 6 Why not merge all seven datasets

They have different units, populations and purposes, with no validated join keys. Joining them would invent relationships. Four files also lack verified exact-file provenance or licenses. More files would not create a defensible demand label.

### 7 Why gradient boosting

It models nonlinear relationships among tabular lagged features by adding trees to reduce remaining loss. Histogram binning makes training practical. We still require baseline comparisons; the architecture's suitability does not prove superior accuracy.

### 8 How is boosting different from a random forest

A random forest averages randomized trees trained largely in parallel. Boosting adds trees sequentially to improve the current prediction under a loss function. Our main forecast uses histogram boosting; the separate session-classification experiment selected a random forest.

### 9 Why predict log units

Sales are skewed and occasional bulk orders can dominate squared-error fitting. log1p compresses large values and handles zero. We use expm1 to return to units. This can understate large orders and does not automatically yield an unbiased mean forecast.

### 10 Why not use LSTM or a Transformer

Those are possible future experiments, not guaranteed upgrades. We first established an auditable tabular baseline with past-only features. A sequence model would need a fair temporal protocol and enough useful histories; we have not shown that it outperforms our current approach.

### 11 Have you tested ARIMA or XGBoost

No. The current nine methods are the three sales baselines, five histogram-boosting configurations and one extra-trees configuration. ARIMA, other tree libraries and deep models should be described as future alternatives, not completed experiments.

### 12 What is a feature versus a target

A feature is information available before the forecast origin, such as preceding-28-day units. The target is what we want to estimate, such as units in the following 28 days. Future target values must not enter the features.

### 13 Why does training have fewer rows than the original dataset

The original rows are invoice items. We aggregate them into daily product-country histories and then create eligible 28-day forecasting examples. The current training set contains 30,205 examples, not one independent example per raw line item.

### 14 How do you prevent leakage

Features stop before each forecast origin, training targets finish before validation, and initial rather than future median prices are used. We also test that editing future quantities and orders does not alter past features. These checks address important boundaries, not every possible bias.

### 15 Why not randomly split the retail data

A future-sales problem must be evaluated in time order. A random split could let later conditions influence a model assessed on earlier conditions or overlap related windows. The separate session classifier uses a random split only because reliable timestamps are unavailable, and we disclose that weakness.

### 16 Why use four different evaluation windows

Two windows compare methods, one separately calibrates empirical ranges, and one later period reports historical-test errors. The last has already been inspected, so it is reused evidence. A newly collected independent period is still needed.

### 17 What is your accuracy

This is regression, so our principal quantities are errors, not classification accuracy. The retained model has MAE 118.33 units, WAPE 44.81% and RMSE 357.34 on the reused 28-day benchmark. WAPE must not be turned into 55.19% accuracy.

### 18 Why keep a baseline

Repeating recent sales is a real alternative available to a retailer. A complex model must demonstrate value beyond that alternative. Here the retained model's absolute-error improvement is small and its squared-error result is worse, so we do not oversell the benefit.

### 19 Why does your best validation model fail later

The relationship between past patterns and subsequent sales varies across periods. Better fit to earlier windows need not transfer to a later seasonal period. That is why we kept the failure visible and used a disclosed release regression guard rather than claiming validation performance guarantees generalization.

### 20 Why not deploy extra trees since its final error is lower

It lost the validation comparison. Picking it only because a previously examined test period looks better would use that test for model selection. It can remain a candidate for a future predeclared independent comparison, but we cannot call that an untouched-test win now.

### 21 What is overfitting and how did you address it

Overfitting means fitting patterns that do not transfer to new observations. We limit tree size, use regularization, compare chronological windows and report later errors. These measures reduce risk; the challenger failure shows that transfer problems still remain.

### 22 Why is RMSE worse when MAE improves

RMSE squares each error before averaging, giving rare large misses more influence. The model can improve many ordinary cases while making a few bulk-order errors worse. Both metrics must stay visible when choosing a planning tool.

### 23 What does 80 point 20 percent coverage mean

It means 80.20% of the measured final-period outcomes fell inside the pooled empirical bands, whose nominal target was 80%. It is not prediction accuracy, does not describe an individual-product confidence guarantee, and can hide poor coverage in particular cases.

### 24 How do you handle zero sales and missing customer IDs

Unrecorded dates are treated as zero recorded sales, with the stockout limitation stated. Missing customer IDs do not remove otherwise usable sales from product forecasting. They are excluded where customer identity is necessary, such as RFM analysis.

### 25 What is the cold-start problem

A product with no sales has no historical lag features. We cannot directly apply a history-based regression as if those observations existed. Our separate analogue workflow uses description, price and comparable products to construct an explicitly unvalidated scenario.

### 26 Explain TF IDF and cosine similarity simply

TF-IDF gives weight to words that help distinguish a description from others in the catalogue. Cosine similarity compares the direction of the weighted text vectors. Matching words can be useful, but synonyms, missing attributes and different product sizes can make lexical matches misleading.

### 27 Is the price effect learned

No. Price proximity adjusts matching similarity, and entered price multiplies scenario units to produce revenue. We have not fitted a causal price-elasticity model, so changing the input price does not establish how actual demand would respond.

### 28 Why must a person confirm analogues

Description overlap cannot guarantee substitutability. A person can reject a small accessory, a different pack size or a product with a misleading name. The confirmation step makes this assumption visible, but it does not solve missing market-exposure data.

### 29 Can your tool tell me which country to launch in first

It can compare supported historical scenarios, but it has not demonstrated a reliable launch-first ranking. Its 84.62% country hit rate equals the simple baseline on the same 26 supported cases. That is an honest limitation, not evidence of 84.62% launch success.

### 30 Why do you report coverage as well as hit rate

A method can look strong by answering only easy cases. Reporting 26 supported cases out of 28 shows the abstentions. The baseline must also be measured on those same supported cases for a fair conditional comparison.

### 31 What is wrong with using first sale as a launch label

The retailer may have stocked or sold the product earlier outside the recorded period or channel. Products that never sold do not appear. The benchmark is therefore catalogue-entry simulation with selection bias, not a complete success-and-failure launch dataset.

### 32 Are the ranges confidence intervals

The launch range is simply the minimum and maximum comparable rates. The established-product forecast band uses calibration residuals. Neither should be described as a guaranteed confidence interval for a future business launch.

### 33 Why does UK appear so often

The source is one UK retailer and reflects its exposure and distribution. In the final forecast test 1,887 of 1,975 eligible pairs are UK pairs. That makes a UK-leading baseline strong and prevents treating the source as balanced global market research.

### 34 Why is classifier accuracy lower than the majority baseline

The session dataset is imbalanced. Always predicting no purchase achieves high accuracy but zero positive recall. Our selected threshold targets validation F1; the forest identifies some buyers at the cost of more false positives. We evaluate precision, recall and average precision, not accuracy alone.

### 35 Why remove PageValues

Its availability at an actionable pre-purchase cutoff is unverified and it is related to transaction-value analytics. We conservatively excluded it. We do not claim to have proved that every possible use of that field is leakage-free or leaked; timing needs domain verification.

### 36 Does your website train the model live

No. Python training runs offline or in Colab. The website reads evaluated historical forecasts and other exported artefacts. Query matching and uploaded-data summaries happen in the browser. A live HGB inference or retraining service is future work.

### 37 What happens when I upload my own company data

The browser validates it and switches explorers and analogue scenarios to an isolated workspace. It does not mix it with the public benchmark or automatically retrain the forecast model. The public Model Lab remains separately labelled; it does not certify accuracy on that upload.

### 38 What are your main technical contributions

An end-to-end prototype connecting verified data, past-only ML features, baseline evaluation, case failures, analyst-reviewed analogues, evidence thresholds, private browser workspaces and reproducible reports. We applied established algorithms; we are not claiming a new boosting algorithm or a new theoretical result.

### 39 What is still incomplete

Independent recent evaluation, robust large-order prediction, reliable country selection, verified launch/failure labels and business pilots. Production features such as authentication, persistent storage, monitored model serving and automatic retraining are also absent. A build-ready site is not the same as a verified live deployment.

### 40 What would you do next

Collect recent real records and reserve an evaluation period before experimentation. Add verified stock availability, promotion and launch outcomes; compare methods across sparse and bulk-order cases; and evaluate country selection against strong same-case baselines. Then validate decisions in a small prospective business pilot rather than promising a fixed accuracy target.

## 14 Final revision checklist

Memorize these facts: 28-day target; 84-day minimum history; 1,067,371 raw line items; 1,003,117 usable merchandise sales; 30,205 training examples; 1,975 final-test pairs; retained MAE 118.33; WAPE 44.81%; worse RMSE 357.34; country hit 84.62% equal to baseline; 26 of 28 supported country cases; no independent fresh test yet.

Avoid saying that the project has achieved high accuracy, predicts launch success, knows current worldwide demand, measures price elasticity, uses all seven datasets for training, retrains automatically on uploads, or has a verified live deployment. The results and workflow are complete as a demonstrable prototype; those broader claims require more evidence.

## 15 References and project evidence

- [Project repository](https://github.com/Rehan-2024/SmartRetailing) and [implementation revision used for this report](https://github.com/Rehan-2024/SmartRetailing/tree/634dcc7).
- [Simple presentation notebook](https://colab.research.google.com/github/Rehan-2024/SmartRetailing/blob/main/notebooks/SmartRetail_Presentation_Colab.ipynb) and [complete model notebook](https://colab.research.google.com/github/Rehan-2024/SmartRetailing/blob/main/notebooks/SmartRetail_All_Models_Colab.ipynb).
- [Dataset source audit and model limitations](https://github.com/Rehan-2024/SmartRetailing/blob/634dcc7/docs/DATA-AUDIT-AND-ROBUSTNESS.md). Reported numerical results are taken from the saved market and academic JSON reports and the executed notebooks, not from a fresh training run performed while writing this document.
- [Online Retail II](https://doi.org/10.24432/C5CG6D), Chen, D. (2012), UCI, CC BY 4.0.
- [Online Shoppers Purchasing Intention](https://doi.org/10.24432/C5F88Q), Sakar, C. and Kastro, Y. (2018), UCI, CC BY 4.0.
- [Wholesale Customers](https://doi.org/10.24432/C5030X), Cardoso, M. (2013), UCI, CC BY 4.0.
- [Histogram gradient boosting](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) and [text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction), official scikit-learn documentation. The project pins scikit-learn 1.8.0; the stable documentation may describe a newer release.
