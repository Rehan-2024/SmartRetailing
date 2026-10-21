# Source legitimacy and model robustness audit

Only three verified observed-data files are accepted by the public benchmark. Their SHA-256 hashes match the exact file bytes inside archives downloaded directly from UCI on 2026-10-21. No synthetic, oversampled or augmented training records are used. Software tests use tiny artificial fixtures and the upload template is an example; neither contributes to model results.

## Accepted sources

- **transactions/online_retail_II.xlsx** — [Chen, D. (2012), UCI](https://archive.ics.uci.edu/dataset/502/online+retail+ii); DOI 10.24432/C5CG6D; CC BY 4.0. Exact official archive-member match. SHA-256 `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`.
- **digital/online_shoppers_intention.csv** — [Sakar, C. & Kastro, Y. (2018), UCI](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset); DOI 10.24432/C5F88Q; CC BY 4.0. Exact official archive-member match. SHA-256 `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`.
- **Wholesale customers data.csv** — [Cardoso, M. (2013), UCI](https://archive.ics.uci.edu/dataset/292/wholesale+customers); DOI 10.24432/C5030X; CC BY 4.0. Exact official archive-member match. SHA-256 `c3d018c643565b85cee733c4a2ac76dd76e080e857cb23f0ccfcc2e15a6c17ef`.

UCI describes Online Retail II as real transaction records from a UK retailer; the shopping file contains observed sessions; Wholesale Customers records distributor-client spending. These represent independent populations and are never joined by invented identities. CC BY is an open-data license requiring attribution. Source authenticity is not a guarantee of error-free records or present-day representativeness.

## Excluded sources

- `marketing_campaign.csv` — excluded from all training, descriptive reports and notebook inputs. Exact-file real-world origin and license have not been verified.
- `store/store_sales_data (2).csv` — excluded from all training, descriptive reports and notebook inputs. Exact-file real-world origin and license have not been verified.
- `geo/geolocations-indian-cities.csv` — excluded from all training, descriptive reports and notebook inputs. Exact-file real-world origin and license have not been verified.
- `geo/Pincode_Dataset.csv` — excluded from all training, descriptive reports and notebook inputs. Exact-file real-world origin and license have not been verified.

Unverified does not mean proven synthetic. The files remain in the user’s original folders and are not deleted. The store file’s one-order/customer/product-per-row pattern and inconsistent date fields warrant caution, but do not establish provenance. The geographic files do not supply product-demand outcomes regardless of origin.

## Enforcement and reproduction

`python pipeline/verify_sources.py --data-root ..` downloads official UCI archives into ignored `artifacts/provenance/` on first use and caches them for subsequent checks. The recorded manifest identifies both archive and member hashes, original URLs, license and attribution. Delete only a chosen cached archive if a fresh network retrieval is needed. Both training pipelines verify inputs against the pinned official member hashes before reading records. The web build checks that published report hashes agree with the source allowlist. The self-contained Colab notebook embeds that same audited allowlist and rejects different input bytes. Raw transaction/customer records are not published. Private browser uploads are user-supplied records, are not source-certified, and carry no independent accuracy claim.

## Fixed experiment and release rule

The expanded comparison has nine candidates: three simple baselines, the incumbent log/Poisson boosting methods, and four history-feature models (absolute-error boosting, log boosting, Poisson boosting and extra trees). Added features use only preceding sales, orders, activity, days since the last sale, variability, trend, annual lag and seasonal position. The first observed price is used rather than a future full-history median. No transaction beyond a forecast origin enters its features. A test explicitly changes all future quantities/orders and verifies unchanged features.

Two chronological 28-day validation windows choose the challenger by aggregate absolute unit error. A following window calibrates empirical bands. The final November–December benchmark was already inspected in prior development: it is explicitly **reused**, not an untouched external test. The release guard retains the incumbent whenever the validation winner worsens WAPE or RMSE on this known benchmark. This is a deployment regression check using inspected results, not validation-only deployment selection. Future independent data is required for strong generalization claims.

Three additional earlier seasonal runs independently fit models and choose candidates from their own past validation windows, then report their subsequent test period. They are historical stress diagnostics; their results do not turn the reused final benchmark into an independent external test.

## Observed results and limits

The new history absolute-error model improves validation MAE from 100.24 to 94.37 units (about 5.9%). In the earlier seasonal checks it reduces MAE versus last-28-days by about 24.2%, 10.0% and 6.3%. But its November–December MAE is 129.47, worse than the incumbent’s 118.33. Its WAPE is 49.04% versus 44.81%, and its RMSE is worse too. Therefore it is **not promoted**. Extra trees scores better in that last period but loses validation; it is not cherry-picked for deployment.

The app retains incumbent MAE 118.33, WAPE 44.81% and RMSE 357.34, versus baseline MAE 120.23, WAPE 45.54% and RMSE 286.98. The incumbent’s larger RMSE remains a material weakness for bulk orders. This is a modest absolute-error improvement, not high forecasting accuracy. WAPE is an error measure and is never reported as “100 minus WAPE” accuracy.

Model Lab and Colab expose volume, intermittent/frequent sales, zero/positive outcome sales, UK/non-UK and seasonal breakdowns, with baseline errors and interval coverage. Outcome-defined zero/positive slices are diagnostic only and cannot guide a decision before outcomes exist. WAPE is undefined when a slice has zero actual units; unit MAE remains meaningful. Case slices overlap. Calibrated bands are empirical and can under-cover particular slices or periods.

The new-product country hit rate remains 84.62% on 26 supported cases, equal to the simple country baseline; no reliable launch-first verdict is claimed. First catalogue appearance is not a verified launch, and never-sold products are missing. Neither new model features nor session-classifier accuracy fixes those missing launch outcomes. The separate purchase classifier still excludes PageValues; its results do not validate the launch planner. Original historical data ends in 2011. Current market performance needs recent real company data and independent evaluation.

All tables and plots in the updated notebook come from executed code. The ZIP exports derived results and provenance, without raw datasets. No claim of achieving a requested accuracy target is made.
