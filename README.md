# SmartRetail Intelligence

A Vercel-ready retail decision-support application built from seven supplied datasets. The dashboard contains ten views: overview, sales, RFM segmentation, retention cohorts, market basket pairs, purchase-intention simulator, wholesale spending profiles, sales forecasts, model evaluation, and data/methodology.

## Run the application

Requires Node.js 20 or later. No npm dependencies or API keys are needed.

```powershell
npm test
npm run build
npm run dev
```

Open http://127.0.0.1:3000. `dev` serves `public`; `build` copies validated, already generated assets to `dist`. It does not retrain models. The committed report is calculated from the supplied files, not invented sample values.

## Deploy to Vercel

1. Import `Rehan-2024/SmartRetailing` into your Vercel account.
2. Select the repository root, Framework Preset **Other**, Build Command `npm run build`, Output Directory `dist`.
3. Deploy. `vercel.json` contains these settings and security headers. No environment variables or Python runtime are required for hosting.

Official configuration reference: https://vercel.com/docs/builds/configure-a-build

Raw source datasets are not required in the deployment. The public report is an immutable historical snapshot. To update results, regenerate the report and deploy the new commit. Hosting is not live training, a database, or a connected store feed.

## Reproduce the analysis

The expected layout outside the repository is:

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
npm test
npm run build
```

Training may take several minutes to read the million-row workbook. The pipeline reads **both** sheets, records SHA-256 fingerprints and actual row counts, and writes `public/data/report.json`. CSV encoding for cities and wholesale is Latin-1; marketing is tab-delimited. Original files remain untouched. A different library version/platform can change numerical results; generated reports record the environment and seed.

## What each dataset does

- **Online Retail II:** GBP gross positive sales, RFM KMeans, cohort retention, exact two-product basket associations, chronological forecast baselines.
- **Online Shoppers Intention:** independent session purchase classification. Deduplicated stratified train/validation/test split, prevalence baseline, logistic regression and random forest. Browser inference uses the evaluated logistic regression coefficients.
- **Wholesale Customers:** separate KMeans profiles on six spending columns, with PCA visualization. Channel and Region are not clustering features.
- **Marketing Campaign:** descriptive spending and response summaries; no unsupported prospective response model.
- **Store sales:** separate region sales/profit summary and Year versus Sales Date quality check. Provenance/currency unverified.
- **Cities and pincodes:** coverage counts and provenance. There is no trained store-location recommendation: these files do not supply rent, competition, footfall, or success labels.

There are no shared customer identifiers across these populations. They are never joined. Store figures are never mixed with GBP transaction revenue.

## Transparent methodology

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for preprocessing, evaluation, decisions, limitations, and acceptance criteria. All displayed metrics can be traced to `pipeline/train.py`. The evaluation view distinguishes validation from held-out testing and descriptive metrics from predictive evidence.

## Privacy and publication

Only aggregates, anonymous cluster projection points, and model coefficients are served. Raw workbook/CSV files and customer names, IDs, dates of birth are excluded from Git and deployment. Source schemas may name sensitive fields, but record values are not published. This is a public research dashboard with no login, uploads, or persistent visitor records. Real customer datasets require a separate access-control and data-retention design before use.

## Verification

`npm test` checks financial reconciliation, classification split/confusion totals, exact sklearn-to-JavaScript probability parity, forecast time boundaries/model selection, cohort rates, and safe anonymous cluster output. No automated tests claim business uplift or future accuracy.
