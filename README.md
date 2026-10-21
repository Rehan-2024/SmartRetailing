# SmartRetail — Product & Launch Decision Studio

An explainable product–market tool, ready for Vercel. Seven pages focus on business decisions instead of displaying every available dataset column.

## Complete model notebook

[Open the full model implementation and results in Google Colab](https://colab.research.google.com/github/Rehan-2024/SmartRetailing/blob/main/notebooks/SmartRetail_All_Models_Colab.ipynb).

The self-contained notebook embeds both actual Python pipelines, executed result tables, ROC/precision–recall curves, confusion heatmaps, cluster profiles/PCA, basket rules, cohort retention, forecast comparisons, chronological checks, new-product scenarios and limitations. Saved outputs are from a local full execution; rerun in Colab by uploading the three verified UCI datasets or selecting your Drive folder. The final cell exports derived result artifacts as a ZIP.

Notebook creation/validation tools are optional development utilities: `python -m pip install nbformat nbclient ipykernel matplotlib`; `python scripts/create_colab.py` regenerates an unexecuted notebook and `python scripts/execute_colab.py` fills it with genuine local results. They are not required for the web deployment.

## What you can do

- **Plan a launch:** describe a new product and planned unit price. Review and confirm comparable products before seeing country-level 28-day unit/revenue scenarios. Recognized product types must match. Unrelated or insufficient evidence triggers an explicit refusal to estimate.
- **Product Explorer:** search stable product codes/names, inspect country sales/returns, view supported 28-day unit forecasts and historical bundles.
- **Market Explorer:** compare products in an observed country with minimum-order filters.
- **Compare & Export:** shortlist product–country pairs and download an evidence brief or comparison.
- **Your Data:** read CSV/XLSX privately in the browser, validate records, and explicitly switch to an isolated company-data workspace. Product/market explorers, comparisons and analogue launch scenarios use the uploaded data. Uploaded data is not mixed with the benchmark.
- **Model Lab:** nine forecast candidates, chronological validation/calibration/test, actual-vs-predicted plot, interval coverage, held-out new-product cohorts, and same-case country baselines. Supporting Version 1 conversion results are separately labeled.
- **Data & Methods:** visible cleaning, identity, formulas, fingerprints and limitations.

## Honest launch estimates

New-product scenarios assume comparable historical exposure and distribution. They are **not** promises of launch demand. The analogue country model did not demonstrate an improvement over its simple country baseline, so the app does not claim a reliable “launch first” winner. It presents evidence-based pilot comparisons instead.

The launch unit range is the spread of comparable-product rates, not a confidence interval. Revenue is scenario units × entered price, without elasticity, profit or marketing assumptions. A country absent from the active data cannot be recommended. Historical records are from 2009–2011, not live market intelligence.

The new-product backtest uses newly appearing product codes, which are not verified launches; never-sold products are absent. Model Lab states these limits. The full contract is [docs/V2-METHODOLOGY.md](docs/V2-METHODOLOGY.md).

## Run locally

Requires Node.js 20+. Dependencies are pinned by `package-lock.json`.

```powershell
npm ci
npm test
npm run build
npm run dev
```

Open http://127.0.0.1:3000. Build prepares the self-hosted Excel reader and copies public assets to `dist`; no model training happens during deployment. Run the build before local preview to enable Excel upload.

## Deploy on Vercel

Import this repository. Set Framework Preset **Other**, Build Command `npm run build`, Output Directory `dist`. `vercel.json` already declares these settings and security headers. No API keys or Python runtime are required to serve the site.

Official configuration: https://vercel.com/docs/builds/configure-a-build

## Reproduce the models

Place the supplied datasets outside the repository:

```text
DATA_ROOT/
  transactions/online_retail_II.xlsx
  digital/online_shoppers_intention.csv
  Wholesale customers data.csv
```

```powershell
python -m pip install -r requirements.txt
python pipeline/verify_sources.py --data-root ..
python pipeline/train.py --data-root ..
python pipeline/build_market.py --data-root ..
npm test
npm run build
```

`train.py` reproduces the academic report using only the three verified sources. Both pipelines reject input hashes that differ from the audited official files. `build_market.py` produces the current merchandise catalogue, market series, analogue model and relevant evaluation. Both read raw files without modifying them. The market report records workbook and pipeline SHA-256 fingerprints. Original source records and direct customer identifiers are not served.

## Upload behavior

Required columns: OrderID, ProductCode, ProductName, Date (ISO YYYY-MM-DD), Quantity, UnitPrice, Country. A downloadable template and familiar column aliases are supported. One selected currency per file. Limits: 10 MB and 50,000 rows; first XLSX worksheet only; CSV or XLSX, not legacy XLS. Exact duplicate business rows are removed; invalid rows are counted; returns stay separate. Uploaded catalogue prices use recorded revenue / recorded units. Launch scenarios on uploaded data have **no independent accuracy claim**. Forecast models are not automatically refitted to private files.

Files stay in the active browser tab and disappear on refresh. No server upload, cloud storage, authentication or persistent customer database is implied. Replacing the workspace clears the shortlist and scenario to prevent source mixing.

## Tests

Automated checks cover canonical product identity, financial reconciliation, selection/time boundaries, exact Python/browser analogue parity, unsupported-query abstention, scenario arithmetic, upload parsing/validation and workspace isolation. Legacy numerical checks remain to protect the separately labeled academic report. Browser QA checks actual navigation, form behavior and upload switching.

## Verified data and robustness audit

See [the audit and model results](docs/DATA-AUDIT-AND-ROBUSTNESS.md) and [the machine-readable source manifest](public/data/sources.json). All three accepted local files exactly match official UCI archive members, with CC BY 4.0 attribution. The four campaign/store/geographic files are excluded because real-world origin and license of those exact files are unverified. Originals stay outside the repository. No generated or augmented rows enter training.

Nine forecast candidates include extra trees and boosted models with strictly lagged order, activity, volatility, trend and annual-history features. Two validation windows select a challenger; an explicit release guard retains the existing model if the challenger worsens reused-benchmark WAPE or RMSE; a separate window calibrates bands. The final-period test was inspected in previous versions and is now explicitly labeled a reused historical benchmark. Three additional seasonal runs independently train/select from their preceding periods. Model Lab and Colab show volume, intermittent-sales, zero-sales, UK/non-UK and period breakdowns, including failures. WAPE is undefined when actual units sum to zero; it is never converted to an accuracy score.

Run feature leakage and source rejection checks with `python -m unittest discover -s tests -p test_pipeline.py`. These small artificial fixtures verify software boundaries and never become model-training data.

## Google Drive dataset folder

All seven originals are stored in the connected user's [SmartRetailData](https://drive.google.com/drive/folders/1cp5PzhIxOnkND4waySrPrOR3vw9fC-L7) folder. The Colab notebook defaults to Drive mode: mount that Google account, read `/content/drive/MyDrive/SmartRetailData`, and save outputs to `/content/drive/MyDrive/SmartRetailResults`. Upload mode remains available. Only the three verified UCI inputs are required for training; the other four files are listed as excluded archives.

The six original CSV files are also committed under [`datasets/`](datasets/README.md), with exact byte hashes in `datasets/manifest.json`. The main XLSX is available through Drive and UCI. Raw datasets are outside `public/` and are not served in the Vercel build. Four unverified CSVs are present solely as requested archives; they remain excluded from all model calculations and have no asserted open-data license.
