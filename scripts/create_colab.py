"""Create a self-contained Colab notebook from the actual project implementations."""
from pathlib import Path
import hashlib
import json
import subprocess
import textwrap
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
legacy_path=ROOT/'pipeline/train.py'
market_path=ROOT/'pipeline/build_market.py'
legacy=legacy_path.read_text(encoding='utf-8-sig').split("if __name__=='__main__':")[0]
market=market_path.read_text(encoding='utf-8-sig').split("if __name__=='__main__':")[0]
legacy=legacy.replace('def main(root,out):','def build_legacy_report(root,out):')
legacy=legacy.replace("hashlib.sha256(Path(__file__).read_bytes()).hexdigest()",'LEGACY_SCRIPT_SHA256')
market=market.replace('def main(root,out):','def build_market_report(root,out):')
market=market.replace("hashlib.sha256(Path(__file__).read_bytes()).hexdigest()",'MARKET_SCRIPT_SHA256')
# Add diagnostics from the same evaluated probabilities; no estimator/split changes.
expanded=[]
for line in legacy.splitlines():
    expanded.append(line)
    if line.strip().startswith('evaluation.append('):
        expanded.extend([
            '        fpr,tpr,_ = roc_curve(yt,prob)',
            '        precision_curve,recall_curve,_ = precision_recall_curve(yt,prob)',
            '        evaluation[-1]["rocCurve"] = {"fpr":fpr.tolist(),"tpr":tpr.tolist()}',
            '        evaluation[-1]["prCurve"] = {"precision":precision_curve.tolist(),"recall":recall_curve.tolist()}',
        ])
legacy='\n'.join(expanded)
report=json.loads((ROOT/'public/data/report.json').read_text())
market_report=json.loads((ROOT/'public/data/market.json').read_text())
snapshot={
    'generatedAt':market_report['generatedAt'],
    'sourceSHA256':market_report['sourceSHA256'],
    'demand':market_report['demand']['candidates'],
    'selectedDemand':market_report['demand']['selected'],
    'coldStart':market_report['coldStart'],
    'purchase':report['purchase']['test'],
    'selectedPurchase':report['purchase']['selected'],
    'legacyEnvironment':report['environment'],
    'segments':{k:report['segments'][k] for k in ['selectedK','candidates','stabilityARI','profiles']},
    'wholesale':{k:report['wholesale'][k] for k in ['selectedK','candidates','stabilityARI','profiles']},
}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s):cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))

md('''
# SmartRetail: All Model Implementations & Results

**An executable Google Colab laboratory for the actual SmartRetail project.**

The notebook embeds the full Python implementations; it does not hide training behind an API or require cloning the repository. Saved outputs in the delivered file come from a full execution on the supplied datasets. Run it again with your copies to reproduce the analysis.

**Models included:** five merchandise demand forecasters; new-product analogue matching with 5/8/12-neighbour comparison; purchase-intention prevalence/logistic/forest comparison; RFM and wholesale KMeans with PCA; basket rules; and the two Version 1 aggregate forecast baselines. All seven datasets are audited, but unrelated populations are never merged.

**Important:** country ranking is a retrospective catalogue-entry benchmark, not a validated launch-success model. The country baseline was not beaten. Unit/revenue scenarios and similarity scores are not success probabilities. WAPE is an error measure, not “accuracy.” Marketing, store and geography files receive descriptive analysis only; this notebook does not invent campaign, city-demand or location models.

## Run in Colab
1. Select **Runtime → Run all** on a CPU runtime.
2. In the upload cell, select the seven original files together. Alternatively choose `DATA_MODE = "drive"` and configure `DRIVE_DATA_ROOT`.
3. Let both training pipelines finish. The million-row workbook is read twice; memory/runtime requirements depend on the selected Colab runtime.
4. Inspect the Model Lab sections, assertions and limitations. The final cell offers a ZIP download of reports and source-fingerprint metadata.

Only upload files you are authorized to place in your Google Colab session. The notebook has no external prediction API, telemetry, or raw-data publishing step. The exported ZIP contains derived public-style artifacts, not raw files.
''')
md(f'''## 1. Provenance and project reference

Embedded from project commit `{commit}`. The hashes below identify the original project scripts; notebook adaptation removes CLI entry points, renames the two `main` functions, and adds ROC/precision–recall diagnostics without changing training decisions. The notebook itself has different bytes from the scripts.

Reference results are embedded so the file remains reviewable without an internet connection. They are **previous project-run results**, not newly executed output until the training cells finish.
''')
code(f'''PROJECT_COMMIT = {commit!r}
LEGACY_SCRIPT_SHA256 = {sha(legacy_path)!r}
MARKET_SCRIPT_SHA256 = {sha(market_path)!r}
REFERENCE_RESULTS = {snapshot!r}
print("Project commit:", PROJECT_COMMIT)
print("Saved project result timestamp:", REFERENCE_RESULTS["generatedAt"])
print("Source workbook SHA-256:", REFERENCE_RESULTS["sourceSHA256"])
''')
md('''## 2. Environment setup

The pinned numerical packages match the project. On a fresh Colab session this cell installs them. If Colab has already imported incompatible versions, restart the session after installation, then run all cells again. Local execution skips installation and uses the prepared project environment.
''')
code('''
import os, sys, subprocess, importlib.util
from pathlib import Path
IN_COLAB = importlib.util.find_spec("google.colab") is not None if importlib.util.find_spec("google") else False
if IN_COLAB:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "numpy==2.3.5", "pandas==3.0.2", "scikit-learn==1.8.0",
        "openpyxl==3.1.5", "matplotlib==3.10.8"])
import json, hashlib, platform, zipfile
import numpy as np
import pandas as pd
import sklearn
import matplotlib
from IPython import get_ipython
from IPython.display import display, Markdown
if get_ipython() is not None:
    get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
plt.rcParams.update({"figure.figsize": (10, 4), "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110})
print({"python": platform.python_version(), "numpy": np.__version__,
       "pandas": pd.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__})
''')
md('''## 3. Select your data

Files: `online_retail_II.xlsx`, `online_shoppers_intention.csv`, `Wholesale customers data.csv`, `marketing_campaign.csv`, `store_sales_data (2).csv`, `geolocations-indian-cities.csv`, `Pincode_Dataset.csv`.

Uploads are placed in the expected source folders. Drive mode expects those folders already present, as in the project README. No dataset is downloaded from an unverified third-party source.
''')
code('''
DATA_MODE = "upload" if IN_COLAB else "local"  # upload | drive | local
DRIVE_DATA_ROOT = "/content/drive/MyDrive/SmartRetailData"
DATA_ROOT = Path("/content/retail_data") if IN_COLAB else Path(os.environ.get("SMART_RETAIL_DATA_ROOT", str(Path.cwd().parent)))
OUTPUT_ROOT = Path("/content/smartretail_results") if IN_COLAB else Path.cwd() / "artifacts" / "colab-run"
FILE_LAYOUT = {
    "online_retail_II.xlsx": "transactions/online_retail_II.xlsx",
    "online_shoppers_intention.csv": "digital/online_shoppers_intention.csv",
    "Wholesale customers data.csv": "Wholesale customers data.csv",
    "marketing_campaign.csv": "marketing_campaign.csv",
    "store_sales_data (2).csv": "store/store_sales_data (2).csv",
    "geolocations-indian-cities.csv": "geo/geolocations-indian-cities.csv",
    "Pincode_Dataset.csv": "geo/Pincode_Dataset.csv",
}
if IN_COLAB and DATA_MODE == "drive":
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_ROOT = Path(DRIVE_DATA_ROOT)
elif IN_COLAB and DATA_MODE == "upload":
    from google.colab import files
    uploaded = files.upload()
    for name, relative in FILE_LAYOUT.items():
        if name in uploaded:
            destination = DATA_ROOT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(uploaded[name])
    del uploaded
missing = [str(DATA_ROOT / relative) for relative in FILE_LAYOUT.values() if not (DATA_ROOT / relative).is_file()]
if missing:
    raise FileNotFoundError("Missing required source files:\\n" + "\\n".join(missing))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
actual_sha = hashlib.sha256((DATA_ROOT / FILE_LAYOUT["online_retail_II.xlsx"]).read_bytes()).hexdigest()
SAME_WORKBOOK = actual_sha == REFERENCE_RESULTS["sourceSHA256"]
print("Input directory:", DATA_ROOT)
print("Matches original workbook bytes:", SAME_WORKBOOK)
print("Outputs:", OUTPUT_ROOT)
''')
md('''## 4. Full implementation: Version 1 analytical models

The following cell defines the actual preprocessing, clustering, classification, association and aggregate forecasting functions. It does not start training until the next cell.

**Classification contract:** exact deduplication → stratified 60/20/20 split; training-only preprocessing; `PageValues` excluded; model selected on validation average precision; shared threshold chosen on validation F1; final test after selection. The retained logistic coefficients are a separately evaluated artifact, not the random forest. The original business simulator was removed in Version 2.

**Clustering contract:** log1p → standardization → KMeans, k=2…6; silhouette, Davies–Bouldin, seed stability and PCA. These are descriptive internal checks, not supervised accuracy.
''')
cells.append(nbf.v4.new_code_cell(legacy))
code('''
build_legacy_report(DATA_ROOT, OUTPUT_ROOT / "legacy")
legacy_results = json.loads((OUTPUT_ROOT / "legacy" / "report.json").read_text(encoding="utf8"))
print("Completed Version 1 analysis:", legacy_results["runId"])
''')
md('''## 5. Source quality and boundaries

The seven sources are independent. Anonymous customer rows contribute to sales, not RFM/cohorts. Gross positive-sales revenue excludes returns; it is not net accounting revenue. The store file has unverified provenance/currency and inconsistent date fields. Geographic coverage does not supply product-demand or location-success labels.
''')
code('''
display(pd.DataFrame(legacy_results["lineage"])[["source", "file", "rows", "bytes", "missingCells", "sha256"]])
r = legacy_results["retail"]
display(pd.DataFrame({"check": ["Raw retail rows", "Exact duplicates removed", "Excluded after deduplication", "Valid positive-sales rows", "Valid sales missing customer ID"],
                      "rows": [r["rawRows"], r["duplicatesRemoved"], r["excludedAfterDedup"], r["cleanRows"], r["missingCustomerRows"]]}))
assert r["rawRows"] == r["duplicatesRemoved"] + r["excludedAfterDedup"] + r["cleanRows"]
print("Retail row reconciliation passed.")
''')
md('''## 6. Purchase-intention model comparison — all reported metrics

Accuracy can be misleading with an imbalanced target. Inspect average precision, recall and confusion matrices alongside it. Changing the threshold changes accuracy, precision and recall. This session classifier does not validate a new-product country recommendation.
''')
code('''
p = legacy_results["purchase"]
print("Split:", p["split"], "| selected:", p["selected"], "| threshold:", p["threshold"])
display(pd.DataFrame(p["validation"]))
classification_table = pd.DataFrame(p["test"]).drop(columns=["confusionMatrix", "rocCurve", "prCurve"])
display(classification_table)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, model in zip(axes, p["test"]):
    cm = np.array(model["confusionMatrix"])
    ax.imshow(cm, cmap="Greens")
    for (i,j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", color="white" if value>cm.max()/2 else "black")
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["No purchase", "Purchase"],
           yticklabels=["No purchase", "Purchase"], xlabel="Predicted outcome", ylabel="Actual outcome", title=model["model"])
fig.suptitle("Purchase intention: held-out test confusion matrices")
plt.tight_layout(); plt.show()
''')
code('''
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for model in p["test"]:
    axes[0].plot(model["rocCurve"]["fpr"], model["rocCurve"]["tpr"], label=model["model"] + f" (AUC {model['rocAUC']:.3f})")
    axes[1].plot(model["prCurve"]["recall"], model["prCurve"]["precision"], label=model["model"] + f" (AP {model['averagePrecision']:.3f})")
axes[0].plot([0,1],[0,1],"--",color="grey",label="Random ranking")
axes[0].set(title="ROC — held-out session test", xlabel="False positive rate", ylabel="True positive rate")
axes[1].set(title="Precision–recall — held-out session test", xlabel="Recall", ylabel="Precision")
for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.15)
plt.tight_layout(); plt.show()
fig, axes = plt.subplots(1,2,figsize=(13,4))
importance = pd.DataFrame(p["importance"]).head(10).sort_values("averagePrecisionDrop")
axes[0].barh(importance.feature, importance.averagePrecisionDrop, color="#47795b")
axes[0].set(title="Selected model: permutation importance", xlabel="Mean test AP decrease (4 permutations)")
cal = pd.DataFrame(p["calibration"])
axes[1].plot(cal.predicted,cal.observed,"o-",label="Selected model")
axes[1].plot([0,1],[0,1],"--",color="grey",label="Perfect reliability reference")
axes[1].set(title="Selected model: test probability reliability",xlabel="Mean predicted purchase probability",ylabel="Observed purchase share")
axes[1].legend();plt.tight_layout();plt.show()
print(p["limitation"])
''')
md('''## 7. RFM and wholesale clustering — model selection, profiles and PCA

Group numbers are arbitrary identifiers. Silhouette is not percentage accuracy. PCA shows a projection rather than a separate predictive model.
''')
code('''
for name, section in [("Retail RFM", legacy_results["segments"]), ("Wholesale spending", legacy_results["wholesale"])]:
    display(Markdown("### " + name))
    print("Selected k:", section["selectedK"], "| cross-seed ARI:", section["stabilityARI"])
    display(pd.DataFrame(section["candidates"]))
    display(pd.DataFrame(section["profiles"]))
    candidates = pd.DataFrame(section["candidates"])
    points = pd.DataFrame(section["points"])
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(candidates.k,candidates.silhouette,"o-",color="#47795b")
    axes[0].set(title=name+": candidate cluster counts",xlabel="Number of clusters (k)",ylabel="Silhouette score")
    for label, group in points.groupby("cluster"):
        axes[1].scatter(group.x,group.y,s=10,alpha=.55,label=f"Cluster {label+1}")
    axes[1].set(title=name+": PCA projection",xlabel="Principal component 1",ylabel="Principal component 2")
    axes[1].legend(fontsize=8)
    plt.tight_layout();plt.show()
    print("PCA variance fractions:", section["pcaExplainedVariance"])
    print(section["limitation"])
''')
md('''## 8. Basket rules and retention

Exact two-itemset counting produces support, confidence and lift; it is not a causal cross-sell uplift test. Cohorts are left-censored; blank future cells are unobserved periods, not zero retention. The 90-day inactivity flag is a rule, not a supervised churn model.
''')
code('''
display(pd.DataFrame(legacy_results["basket"]["rules"]))
print(legacy_results["basket"]["method"])
print(legacy_results["basket"]["limitation"])
cohorts = legacy_results["cohorts"]
cohort_matrix = pd.DataFrame([c["retention"] for c in cohorts], index=[c["cohort"] for c in cohorts])
fig, ax = plt.subplots(figsize=(12,7))
im=ax.imshow(cohort_matrix.to_numpy(dtype=float),aspect="auto",vmin=0,vmax=1,cmap="Greens")
ax.set(yticks=range(len(cohorts)),yticklabels=cohort_matrix.index,
       xticks=range(cohort_matrix.shape[1]),xticklabels=[f"M{i}" for i in range(cohort_matrix.shape[1])],
       title="Retail cohort retention: purchasing share of acquisition cohort",xlabel="Months since first observed purchase",ylabel="First observed purchase month")
fig.colorbar(im,ax=ax,label="Purchasing share (0–1)")
plt.tight_layout();plt.show()
''')
md('''## 9. Version 1 aggregate revenue forecasting baselines

These are overall GBP sales baselines, not product-country unit forecasts. Validation chooses between them; the final fold is held out. Aggregate backtest figures include validation windows and are labeled accordingly.
''')
code('''
old_forecast = legacy_results["forecast"]
display(pd.DataFrame([{**{k:v for k,v in m.items() if k!="folds"}, "final_test_mae":m["folds"][-1]["mae"]} for m in old_forecast["metrics"]]))
for model in old_forecast["metrics"]:
    print(model["model"]);display(pd.DataFrame(model["folds"]))
future = pd.DataFrame(old_forecast["future"])
plt.plot(pd.to_datetime(future.date),future.revenue,color="#47795b")
plt.title("Version 1: next 28 historical-source days of gross GBP sales")
plt.xlabel("Historical forecast date");plt.ylabel("Gross positive sales (GBP)")
plt.xticks(rotation=30);plt.tight_layout();plt.show()
print(old_forecast["protocol"])
print(old_forecast["limitation"])
''')
md('''## 10. Full implementation: current product–market models

The following cell defines the complete current pipeline: merchandise cleaning, canonical identity, daily product-country matrices, three deterministic forecasting baselines, log-unit/Poisson gradient boosting, calibration, forward-only evaluation, type-guarded TF–IDF/price analogue matching, and withheld catalogue-entry cohorts.

**Demand protocol:** training labels finish before validation; two 28-day validation windows select the model by absolute unit error; one calibration window fits an empirical residual band; one later test window measures generalization. No universal “accuracy” is calculated from WAPE.

**Country protocol:** three historical cohorts select neighbour count; a fourth is the final test. Each target is new to the recorded catalogue, and its sales outcomes are absent from analogue training. Prelaunch availability of the description/price is assumed. A first recorded sale is not proof of a true launch, and never-sold products are absent.
''')
cells.append(nbf.v4.new_code_cell(market))
code('''
build_market_report(DATA_ROOT, OUTPUT_ROOT / "current")
current = json.loads((OUTPUT_ROOT / "current" / "market.json").read_text(encoding="utf8"))
print("Built merchandise catalogue:", len(current["products"]), "products /", len(current["markets"]), "product-country pairs")
display(pd.DataFrame(list(current["audit"].items()), columns=["Audit", "Result"]))
''')
md('''## 11. Current demand forecasting — every candidate and error measure

MAE and RMSE are units per product-country pair per 28 days; WAPE is total absolute error divided by total actual units. A model can improve MAE while worsening RMSE: neither metric should be hidden. Selection is by validation absolute error, not the most flattering final-test score.
''')
code('''
demand = current["demand"]
forecast_rows=[]
for candidate in demand["candidates"]:
    for partition in ["validation","test"]:
        forecast_rows.append({"model":candidate["model"],"partition":partition,**candidate[partition]})
forecast_table=pd.DataFrame(forecast_rows)
print("Selected on validation:",demand["selected"])
display(forecast_table)
display(pd.DataFrame(demand["folds"]))
fig,axes=plt.subplots(1,2,figsize=(14,4))
test_table=forecast_table[forecast_table.partition=="test"]
axes[0].barh(test_table.model,test_table.mae,color="#47795b")
axes[0].set(title="Current demand models: final-test MAE",xlabel="Absolute error (units / pair / 28 days)")
axes[1].barh(test_table.model,test_table.wape*100,color="#819666")
axes[1].set(title="Current demand models: final-test WAPE",xlabel="Weighted absolute percentage error (%)")
plt.tight_layout();plt.show()
print(demand["protocol"])
''')
code('''
points=pd.DataFrame(demand["testPoints"])
fig,axes=plt.subplots(1,2,figsize=(13,4))
maximum=max(points.actual.max(),points.predicted.max())
axes[0].scatter(points.actual,points.predicted,alpha=.55,s=18,color="#47795b")
axes[0].plot([0,maximum],[0,maximum],"--",color="grey")
axes[0].set(title="Demand: first 200 final-test pairs",xlabel="Actual units / 28 days",ylabel="Predicted units / 28 days")
axes[1].bar(["Nominal","Observed test"],[demand["intervalNominal"],demand["coverage"]],color=["#a9b2a4","#47795b"])
axes[1].set(ylim=(0,1),title="Pooled empirical prediction-band coverage",ylabel="Fraction inside band (0–1)")
plt.tight_layout();plt.show()
print("Calibration rows:",demand["calibrationRows"],"| empirical final-test coverage:",demand["coverage"])
print(demand["limitation"])
''')
md('''## 12. New-product country and unit benchmarks — results including failure

All neighbour-count comparisons are shown. The country baseline is evaluated on the **same supported test cases**. Abstention coverage is displayed to prevent high accuracy from hiding unsupported queries. The UI permits only pilot comparisons when the country baseline is not beaten. Manual analogue overrides are scenario choices with no independent accuracy claim.
''')
code('''
cold=current["coldStart"]
cold_table=pd.DataFrame(cold["results"])
display(cold_table)
display(pd.DataFrame(cold["folds"]))
selected_cold=cold_table[(cold_table.role=="test")&(cold_table.neighbors==cold["selectedNeighbors"])].iloc[0]
print("Validation-selected neighbours:",cold["selectedNeighbors"])
print("Country baseline beaten:",cold["beatsCountryBaseline"])
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].bar(["Analogue method","Country baseline"],[selected_cold.topCountryHitRate,selected_cold.baselineHitRate],color=["#47795b","#a9b2a4"])
axes[0].set(ylim=(0,1),title="Same-case final-test top-country hit rate",ylabel="Correct top-country fraction (0–1)")
axes[1].bar(["Analogue scenario","Unit baseline"],[selected_cold.unitMAE,selected_cold.baselineUnitMAE],color=["#47795b","#a9b2a4"])
axes[1].set(title="Final-test total unit scenario error",ylabel="MAE (units / target product)")
plt.tight_layout();plt.show()
print(cold["protocol"])
print(cold["limitation"])
''')
md('''## 13. Inspect a new-product scenario yourself

Edit the product description and planned price. These are **scenario inputs**, not an actual company launch claim. Review the proposed comparables and edit `CHOSEN_PRODUCT_CODES` to remove accessories or products that are not substitutes. A recognized product-type guard reduces obvious word-overlap errors, but does not establish semantic or commercial equivalence.

Country scenarios need ≥3 active analogues and ≥10 orders in the trailing 84 days. No unseen country is manufactured. The range is the min/max analogue rate, not a confidence interval. Revenue is units × entered price, with no price elasticity or profit model.
''')
code('''
NEW_PRODUCT_DESCRIPTION="ceramic coffee mug"
PLANNED_UNIT_PRICE_GBP=2.50
FEASIBLE_COUNTRIES=None  # None = observed markets; or e.g. ["France","Germany"]
CHOSEN_PRODUCT_CODES=None  # replace with a list after reviewing candidate names
products=current["products"]
text_model=current["textModel"]
def notebook_matches(description,price):
    import re
    tokens=re.findall(r"\\b[a-z0-9]{2,}\\b",description.lower())
    tokens=[w for w in tokens if w not in text_model["stopWords"] and w in text_model["vocabulary"]]
    from collections import Counter
    query={text_model["vocabulary"][w]:n*text_model["idf"][text_model["vocabulary"][w]] for w,n in Counter(tokens).items()}
    norm=np.sqrt(sum(v*v for v in query.values())) or 1
    query={j:v/norm for j,v in query.items()}
    families=[set(g) for g in text_model["typeFamilies"] if set(tokens).intersection(g)]
    recent_counts=Counter()
    for row in current["markets"]:recent_counts[row["product"]]+=row["recentOrders"]
    cutoff=pd.Timestamp(current["period"]["end"])-pd.Timedelta(days=84)
    found=[]
    for i,product in enumerate(products):
        if pd.Timestamp(product["first"])>cutoff or recent_counts[i]<10:continue
        if product["description"].lower()==description.lower():continue
        words=set(re.findall(r"\\b[a-z0-9]{2,}\\b",product["description"].lower()))
        if families and not any(words.intersection(g) for g in families):continue
        cosine=sum(query.get(j,0)*v for j,v in product["vector"])
        price_factor=np.exp(-abs(np.log(max(.01,product["price"])/max(.01,price))))
        score=cosine*price_factor
        if cosine>=.15 and score>.05:
            found.append({"index":i,"code":product["code"],"description":product["description"],"referencePrice":product["price"],"textSimilarity":cosine,"priceCloseness":price_factor,"weight":score})
    return sorted(found,key=lambda r:-r["weight"])[:cold["selectedNeighbors"]]
matches=notebook_matches(NEW_PRODUCT_DESCRIPTION,PLANNED_UNIT_PRICE_GBP)
display(pd.DataFrame(matches))
print("Inspect these names. Selection does not establish prospective accuracy.")
''')
code('''
chosen=matches if CHOSEN_PRODUCT_CODES is None else [r for r in matches if r["code"] in CHOSEN_PRODUCT_CODES]
scenario_rows=[]
if len(chosen)<3:
    print("INSUFFICIENT EVIDENCE: fewer than three comparable products.")
else:
    lookup={m["index"]:m for m in chosen}
    for country in current["countries"]:
        if FEASIBLE_COUNTRIES is not None and country not in FEASIBLE_COUNTRIES:continue
        if country=="Unknown":continue
        rows=[r for r in current["markets"] if r["country"]==country and r["product"] in lookup and r["recentOrders"]>=1]
        if len(rows)<3 or sum(r["recentOrders"] for r in rows)<10:continue
        weights=np.array([lookup[r["product"]]["weight"] for r in rows])
        rates=np.array([r["recent28"] for r in rows])
        units=float(np.average(rates,weights=weights))
        scenario_rows.append({"country":country,"scenarioUnits28d":units,"revenueAtEnteredPriceGBP":units*PLANNED_UNIT_PRICE_GBP,
            "analogueSpreadLow":float(rates.min()),"analogueSpreadHigh":float(rates.max()),"activeAnalogues":len(rows),"recentOrders":sum(r["recentOrders"] for r in rows)})
scenario_table=pd.DataFrame(scenario_rows)
if len(scenario_table):
    scenario_table=scenario_table.sort_values("scenarioUnits28d",ascending=False)
    display(scenario_table)
    if not cold["beatsCountryBaseline"]:
        print("NO VALIDATED LAUNCH-FIRST WINNER: the automatic country method did not beat its baseline. These are pilot comparisons.")
else:
    print("No eligible country has sufficient active analogue evidence.")
print("Ranges are analogue spreads, not statistical confidence intervals. Scenarios assume comparable distribution and exposure.")
''')
md('''## 14. Descriptive sources — explicitly not extra trained models

Campaign response prevalence and spending, store consistency checks, and geographic coverage are descriptive outputs only. No joint customer identities, city-demand estimates, market-size labels or supervised churn labels are inferred from these files.
''')
code('''
display(pd.DataFrame({"marketingRows":[legacy_results["marketing"]["rows"]],
                      "responseShare":[legacy_results["marketing"]["responseRate"]],
                      "missingIncome":[legacy_results["marketing"]["missingIncome"]]}))
display(pd.DataFrame(list(legacy_results["marketing"]["spending"].items()),columns=["Category","Mean recorded spend"]))
display(pd.DataFrame(legacy_results["store"]["regions"]))
print("Store Year/Sales Date mismatches:",legacy_results["store"]["dateMismatches"])
print(legacy_results["store"]["note"])
print(legacy_results["geography"])
''')
md('''## 15. Verification and comparison with the saved project run

These checks test arithmetic, data boundaries and model-selection discipline. They do not prove commercial uplift or future accuracy. Numerical values can vary across Python/library/platform versions; differences are reported rather than silently replaced with reference values.
''')
code('''
assert len({p["code"] for p in current["products"]})==len(current["products"])
assert len({(r["product"],r["country"]) for r in current["markets"]})==len(current["markets"])
assert abs(sum(r["revenue"] for r in current["markets"])-current["audit"]["merchandiseRevenue"])<1
assert demand["trainLatestLabelIndex"]<demand["folds"][0]["originIndex"]
assert [r["role"] for r in demand["folds"]]==["validation","validation","calibration","test"]
for a,b in zip(demand["folds"],demand["folds"][1:]):assert a["forecastEnd"]<b["forecastStart"]
expected=min(demand["candidates"],key=lambda m:m["validation"]["mae"])["model"]
assert demand["selected"]==expected
for model in p["test"]:assert sum(map(sum,model["confusionMatrix"]))==p["split"]["test"]
assert sum(p["split"].values())==p["rows"]
assert cold["beatsCountryBaseline"]==bool(selected_cold.supported>=20 and selected_cold.topCountryHitRate>selected_cold.baselineHitRate)
reference=pd.DataFrame([{ "model":r["model"],"referenceTestMAE":r["test"]["mae"],"referenceTestWAPE":r["test"]["wape"]} for r in REFERENCE_RESULTS["demand"]])
observed=pd.DataFrame([{ "model":r["model"],"rerunTestMAE":r["test"]["mae"],"rerunTestWAPE":r["test"]["wape"]} for r in demand["candidates"]])
comparison=reference.merge(observed,on="model")
comparison["maeDifference"]=comparison.rerunTestMAE-comparison.referenceTestMAE
display(comparison)
print("All structural checks passed. Same original workbook:",SAME_WORKBOOK)
print("Reference environment:",REFERENCE_RESULTS["legacyEnvironment"])
''')
md('''## 16. Export results

The ZIP contains regenerated reports and scenario/metric tables, not raw datasets. Save the notebook from Colab separately if you want the new cell outputs preserved. Colab filesystem paths are temporary unless copied to your Drive.

**Credits:** Online Retail II, Daqing Chen, UCI, DOI https://doi.org/10.24432/C5CG6D (CC BY 4.0). Online Shoppers Purchasing Intention and Wholesale Customers are independent UCI datasets. Supplemental file provenance is unverified. Read the project's V2 methodology for the full data and evaluation contract.
''')
code('''
classification_table.to_csv(OUTPUT_ROOT/"classification_metrics.csv",index=False)
forecast_table.to_csv(OUTPUT_ROOT/"demand_forecast_metrics.csv",index=False)
cold_table.to_csv(OUTPUT_ROOT/"new_product_benchmark.csv",index=False)
if len(scenario_table):scenario_table.to_csv(OUTPUT_ROOT/"new_product_scenario.csv",index=False)
(OUTPUT_ROOT/"run_metadata.json").write_text(json.dumps({"projectCommit":PROJECT_COMMIT,"legacyScriptSHA256":LEGACY_SCRIPT_SHA256,
    "marketScriptSHA256":MARKET_SCRIPT_SHA256,"sameWorkbook":SAME_WORKBOOK,"sourceSHA256":actual_sha,
    "environment":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__}},indent=2),encoding="utf8")
archive=OUTPUT_ROOT.parent/"SmartRetail_Model_Results.zip"
with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for file in OUTPUT_ROOT.rglob("*"):
        if file.is_file():z.write(file,file.relative_to(OUTPUT_ROOT))
print("Saved:",archive,"| bytes:",archive.stat().st_size)
if IN_COLAB:
    from google.colab import files
    files.download(str(archive))
''')

notebook=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3'},
    'colab':{'name':'SmartRetail_All_Models_Colab.ipynb','provenance':[]},
    'smartretail':{'project_commit':commit,'scope':'Full embedded implementations, rerun results and limitations'},
})
destination=ROOT/'notebooks/SmartRetail_All_Models_Colab.ipynb'
destination.parent.mkdir(exist_ok=True)
nbf.validate(notebook)
for i,cell in enumerate(notebook.cells):
    if cell.cell_type=='code':compile(cell.source,f'cell_{i}','exec')
nbf.write(notebook,destination)
print(f'Created {destination} ({len(cells)} cells).')
