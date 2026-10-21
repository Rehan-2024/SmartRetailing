# SmartRetail — Product & Launch Decision Studio

An explainable product–market tool, ready for Vercel. Seven pages focus on business decisions instead of displaying every available dataset column.

## What you can do

- **Plan a launch:** describe a new product and planned unit price. Review and confirm comparable products before seeing country-level 28-day unit/revenue scenarios. Recognized product types must match. Unrelated or insufficient evidence triggers an explicit refusal to estimate.
- **Product Explorer:** search stable product codes/names, inspect country sales/returns, view supported 28-day unit forecasts and historical bundles.
- **Market Explorer:** compare products in an observed country with minimum-order filters.
- **Compare & Export:** shortlist product–country pairs and download an evidence brief or comparison.
- **Your Data:** read CSV/XLSX privately in the browser, validate records, and explicitly switch to an isolated company-data workspace. Product/market explorers, comparisons and analogue launch scenarios use the uploaded data. Uploaded data is not mixed with the benchmark.
- **Model Lab:** five forecast candidates, chronological validation/calibration/test, actual-vs-predicted plot, interval coverage, held-out new-product cohorts, and same-case country baselines. Supporting Version 1 conversion results are separately labeled.
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
  marketing_campaign.csv
  store/store_sales_data (2).csv
  geo/geolocations-indian-cities.csv
  geo/Pincode_Dataset.csv
```

```powershell
python -m pip install -r requirements.txt
python pipeline/train.py --data-root ..
python pipeline/build_market.py --data-root ..
npm test
npm run build
```

`train.py` reproduces the preserved Version 1 academic report. `build_market.py` produces the current merchandise catalogue, market series, analogue model and relevant evaluation. Both read raw files without modifying them. The market report records workbook and pipeline SHA-256 fingerprints. Original source records and direct customer identifiers are not served.

## Upload behavior

Required columns: OrderID, ProductCode, ProductName, Date (ISO YYYY-MM-DD), Quantity, UnitPrice, Country. A downloadable template and familiar column aliases are supported. One selected currency per file. Limits: 10 MB and 50,000 rows; first XLSX worksheet only; CSV or XLSX, not legacy XLS. Exact duplicate business rows are removed; invalid rows are counted; returns stay separate. Uploaded catalogue prices use recorded revenue / recorded units. Launch scenarios on uploaded data have **no independent accuracy claim**. Forecast models are not automatically refitted to private files.

Files stay in the active browser tab and disappear on refresh. No server upload, cloud storage, authentication or persistent customer database is implied. Replacing the workspace clears the shortlist and scenario to prevent source mixing.

## Tests

Automated checks cover canonical product identity, financial reconciliation, selection/time boundaries, exact Python/browser analogue parity, unsupported-query abstention, scenario arithmetic, upload parsing/validation and workspace isolation. Legacy numerical checks remain to protect the separately labeled academic report. Browser QA checks actual navigation, form behavior and upload switching.
