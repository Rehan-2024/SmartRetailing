"""Rebuild every public insight from independent, locally supplied source files."""
import argparse, hashlib, json, platform, time
from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
from collections import Counter
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (silhouette_score, davies_bouldin_score, adjusted_rand_score,
    roc_auc_score, average_precision_score, precision_score, recall_score, f1_score,
    brier_score_loss, confusion_matrix, accuracy_score)
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

SEED=42
def records(df): return json.loads(df.to_json(orient='records',date_format='iso'))
def clean_json(x):
    if isinstance(x,dict): return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean_json(v) for v in x]
    if isinstance(x,np.generic): x=x.item()
    if isinstance(x,float) and not np.isfinite(x): return None
    return x
def cluster(frame):
    transformed=np.log1p(frame)
    scaler=StandardScaler(); z=scaler.fit_transform(transformed)
    candidates=[]; fitted={}
    for k in range(2,7):
        model=KMeans(n_clusters=k,random_state=SEED,n_init=10).fit(z)
        score=silhouette_score(z,model.labels_,sample_size=min(2000,len(z)),random_state=SEED)
        candidates.append({'k':k,'silhouette':score,'daviesBouldin':davies_bouldin_score(z,model.labels_)})
        fitted[k]=model
    winner=max(candidates,key=lambda a:a['silhouette']); model=fitted[winner['k']]
    other=KMeans(n_clusters=winner['k'],random_state=7,n_init=10).fit(z)
    labeled=frame.assign(cluster=model.labels_)
    summary=labeled.groupby('cluster').mean().round(2)
    summary['customers']=labeled.groupby('cluster').size()
    pca=PCA(n_components=2).fit(z); xy=pca.transform(z)
    rng=np.random.default_rng(SEED); ix=rng.choice(len(z),min(len(z),700),replace=False)
    return {'selectedK':winner['k'],'candidates':candidates,'stabilityARI':adjusted_rand_score(model.labels_,other.labels_),
        'profiles':records(summary.reset_index()),'points':[{'x':float(xy[i,0]),'y':float(xy[i,1]),'cluster':int(model.labels_[i])} for i in ix],
        'pcaExplainedVariance':pca.explained_variance_ratio_.tolist(),
        'method':'log1p → StandardScaler → KMeans; k=2…6 selected by silhouette. PCA is a display projection, not a classifier.',
        'limitation':'Internal descriptive validation on the same population; no ground-truth segments or evidence of campaign uplift.'},model.labels_

def classification(source):
    # Deduplicate before splitting so identical sessions cannot straddle partitions.
    raw=len(source); source=source.drop_duplicates().copy()
    y=source.pop('Revenue').astype(int)
    # PageValues is a value derived from page visits relative to transactions; exclude
    # it because its availability at an actionable pre-purchase cutoff is unverified.
    X=source.drop(columns=['PageValues'])
    cat=['Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend']
    num=[c for c in X.columns if c not in cat]
    X[cat]=X[cat].astype(str)
    a,xt,b,yt=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
    xtr,xv,ytr,yv=train_test_split(a,b,test_size=.25,stratify=b,random_state=SEED)
    def prep(): return ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
    models={'Prevalence baseline':DummyClassifier(strategy='prior'), 'Logistic regression':LogisticRegression(max_iter=1500,random_state=SEED),'Random forest':RandomForestClassifier(n_estimators=160,min_samples_leaf=8,max_depth=12,random_state=SEED,n_jobs=-1)}
    fitted={}; val=[]
    for name,m in models.items():
        p=Pipeline([('prep',prep()),('model',m)]).fit(xtr,ytr); fitted[name]=p
        prob=p.predict_proba(xv)[:,1]
        val.append({'model':name,'validationAveragePrecision':average_precision_score(yv,prob),'validationROC':roc_auc_score(yv,prob)})
    selected=max((v for v in val if v['model']!='Prevalence baseline'),key=lambda v:v['validationAveragePrecision'])['model']
    p=fitted[selected]; vp=p.predict_proba(xv)[:,1]
    thresholds=np.linspace(.05,.95,91); threshold=float(max(thresholds,key=lambda t:f1_score(yv,vp>=t)))
    evaluation=[]
    for name,model in fitted.items():
        prob=model.predict_proba(xt)[:,1]; pred=prob>=threshold
        evaluation.append({'model':name,'threshold':threshold,'rocAUC':roc_auc_score(yt,prob),'averagePrecision':average_precision_score(yt,prob),'precision':precision_score(yt,pred,zero_division=0),'recall':recall_score(yt,pred,zero_division=0),'f1':f1_score(yt,pred,zero_division=0),'accuracy':accuracy_score(yt,pred),'brier':brier_score_loss(yt,prob),'confusionMatrix':confusion_matrix(yt,pred,labels=[0,1]).tolist()})
    importance=permutation_importance(p,xt,yt,n_repeats=4,random_state=SEED,scoring='average_precision',n_jobs=1)
    calibration=[]; prob=p.predict_proba(xt)[:,1]
    for lo in np.arange(0,1,.1):
        mask=(prob>=lo)&(prob<lo+.1)
        if mask.sum(): calibration.append({'predicted':float(prob[mask].mean()),'observed':float(yt.to_numpy()[mask].mean()),'n':int(mask.sum())})
    # Export the evaluated logistic pipeline as coefficients for exact browser inference.
    lp=fitted['Logistic regression']; pre=lp.named_steps['prep']; scale=pre.named_transformers_['num'].named_steps['scale']; enc=pre.named_transformers_['cat'].named_steps['encode']; lr=lp.named_steps['model']
    inference={'numeric':num,'categorical':cat,'mean':scale.mean_.tolist(),'scale':scale.scale_.tolist(),'categories':[c.tolist() for c in enc.categories_],'coefficients':lr.coef_[0].tolist(),'intercept':float(lr.intercept_[0]),'threshold':threshold,
        'defaults':{**{c:float(xtr[c].median()) for c in num},**{c:str(xtr[c].mode()[0]) for c in cat}},
        'parityCases':[{'input':row.to_dict(),'probability':float(lp.predict_proba(pd.DataFrame([row]))[0,1])} for _,row in xt.head(4).iterrows()]}
    return {'rawRows':raw,'rows':len(X),'duplicatesRemoved':raw-len(X),'positiveRate':float(y.mean()),'split':{'train':len(xtr),'validation':len(xv),'test':len(xt)},'seed':SEED,'selected':selected,'threshold':threshold,'validation':val,'test':evaluation,'calibration':calibration,'importance':[{'feature':c,'averagePrecisionDrop':float(v)} for c,v in sorted(zip(X.columns,importance.importances_mean),key=lambda a:-a[1])],
        'excluded':['Revenue (target)','PageValues (unverified pre-purchase availability)'],
        'protocol':'Stratified 60/20/20 split after deduplication. Preprocessing fits training only. Model selected by validation average precision; threshold selected by validation F1. Previously inspected historical test reused for reproducibility; not a fresh external test. No refit after test.',
        'limitation':'Session-complete benchmark, not a real-time or causal conversion promise. No timestamps for temporal validation. Random split cannot measure future-period drift. Browser simulator uses evaluated logistic regression, even if random forest wins.', 'inference':inference}

def forecast(daily):
    # Explicit chronological expanding-window test; no future values enter training.
    series=daily.set_index('date')['revenue'].asfreq('D',fill_value=0)
    horizon=28; origins=list(range(len(series)-112,len(series),28)); results=[]; predictions=[]
    for model in ['Last week','Weekday mean (8 weeks)']:
        actual=[]; predicted=[]; folds=[]
        for origin in origins:
            train=series.iloc[:origin]; test=series.iloc[origin:origin+horizon]
            p=[float(train.iloc[-7:].iloc[i%7]) if model=='Last week' else float(train.iloc[-56:][train.iloc[-56:].index.dayofweek==day.dayofweek].mean()) for i,day in enumerate(test.index)]
            actual.extend(test.tolist()); predicted.extend(p)
            folds.append({'trainEnd':str(train.index[-1].date()),'testStart':str(test.index[0].date()),'testEnd':str(test.index[-1].date()),'mae':float(np.mean(np.abs(test.to_numpy()-p)))})
        y=np.array(actual); p=np.array(predicted)
        results.append({'model':model,'mae':float(abs(y-p).mean()),'rmse':float(np.sqrt(((y-p)**2).mean())),'wape':float(abs(y-p).sum()/abs(y).sum()),'folds':folds})
    # First 3 origins select model; final origin remains held out.
    selected=min(results,key=lambda r:np.mean([f['mae'] for f in r['folds'][:3]]))['model']
    dates=pd.date_range(series.index[-1]+pd.Timedelta(days=1),periods=horizon)
    for i,d in enumerate(dates):
        v=float(series.iloc[-7:].iloc[i%7]) if selected=='Last week' else float(series.iloc[-56:][series.iloc[-56:].index.dayofweek==d.dayofweek].mean())
        predictions.append({'date':str(d.date()),'revenue':v})
    return {'selected':selected,'metrics':results,'future':predictions,'horizon':horizon,'validationOrigins':3,'testOrigins':1,'protocol':'Four expanding-window 28-day origins. First three origins select by MAE; fourth is the reused chronological test benchmark, previously inspected during development. Aggregate metrics include all four and are labeled backtest metrics. Final forecast refits the selected rule on all history.',
        'limitation':'Historical gross-sales planning baseline; the final source day may be partial. Zero-filled dates assume no recorded sales. No causal, promotion, holiday, or prediction-interval model. Forecast follows the historical dataset, not today.'}

def main(root,out):
    from verify_sources import require_verified
    require_verified(root,['retail','purchase','wholesale'])
    started=time.time(); files=[('retail','transactions/online_retail_II.xlsx'),('purchase','digital/online_shoppers_intention.csv'),('wholesale','Wholesale customers data.csv')]
    lineage=[]
    for name,file in files:
        path=root/file
        lineage.append({'source':name,'file':file,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    print('Reading both retail sheets...',flush=True)
    sheets=pd.read_excel(root/files[0][1],sheet_name=None)
    retail=pd.concat(sheets.values(),ignore_index=True)
    raw=len(retail); dup=int(retail.duplicated().sum()); retail=retail.drop_duplicates()
    retail['InvoiceDate']=pd.to_datetime(retail.InvoiceDate,errors='coerce')
    cancel=retail.Invoice.astype(str).str.upper().str.startswith('C')
    returns=retail[cancel|retail.Quantity.lt(0)]
    valid=retail[~cancel & retail.Quantity.gt(0)&retail.Price.gt(0)&retail.InvoiceDate.notna()].copy()
    valid['revenue']=valid.Quantity*valid.Price
    valid['date']=valid.InvoiceDate.dt.floor('D')
    daily=valid.groupby('date').agg(revenue=('revenue','sum'),orders=('Invoice','nunique')).reset_index()
    monthly=valid.groupby(valid.InvoiceDate.dt.strftime('%Y-%m')).agg(revenue=('revenue','sum'),orders=('Invoice','nunique'),units=('Quantity','sum')).reset_index().rename(columns={'InvoiceDate':'month'})
    identified=valid.dropna(subset=['Customer ID']); snapshot=valid.InvoiceDate.max().normalize()+pd.Timedelta(days=1)
    rfm=identified.groupby('Customer ID').agg(last=('InvoiceDate','max'),Frequency=('Invoice','nunique'),Monetary=('revenue','sum'))
    rfm['Recency']=(snapshot-rfm.pop('last')).dt.days
    print('Clustering customers...',flush=True)
    segments,labels=cluster(rfm[['Recency','Frequency','Monetary']])
    segments['snapshotDate']=str(snapshot.date()); segments['customers']=len(rfm)
    segments['inactive90Days']=int((rfm.Recency>90).sum())
    # Cohorts need customer IDs, one activity per customer-month.
    activity=identified[['Customer ID','InvoiceDate']].copy(); activity['period']=activity.InvoiceDate.dt.to_period('M')
    first=activity.groupby('Customer ID').period.min(); activity['cohort']=activity['Customer ID'].map(first)
    activity['age']=activity.period.astype('int64')-activity.cohort.astype('int64')
    cohort=activity.groupby(['cohort','age'])['Customer ID'].nunique().unstack(); sizes=cohort[0]
    last_period=activity.period.max()
    cohorts=[{'cohort':str(c),'customers':int(sizes[c]),'retention':[None if c+i>last_period else (0.0 if pd.isna(v) else float(v/sizes[c])) for i,v in enumerate(cohort.loc[c].iloc[:13])]} for c in cohort.index]
    # Count all single products and pairs; minimum support prunes rare products exactly for pair rules.
    print('Mining basket pairs...',flush=True)
    baskets=valid.groupby('Invoice').StockCode.agg(lambda s:sorted(set(s.astype(str))))
    n=len(baskets); single=Counter(v for basket in baskets for v in basket); min_count=max(20,int(np.ceil(.005*n)))
    keep={v for v,c in single.items() if c>=min_count}; pairs=Counter(pair for basket in baskets for pair in combinations([v for v in basket if v in keep],2))
    descriptions=valid.dropna(subset=['Description']).drop_duplicates('StockCode').set_index('StockCode').Description
    names={str(k):str(v) for k,v in descriptions.items()}; rules=[]
    for (a,b),count in pairs.items():
        if count<min_count: continue
        for left,right in [(a,b),(b,a)]:
            confidence=count/single[left]; lift=confidence/(single[right]/n)
            if confidence>=.2 and lift>1: rules.append({'antecedent':names.get(left,left),'consequent':names.get(right,right),'support':count/n,'confidence':confidence,'lift':lift,'baskets':count})
    rules=sorted(rules,key=lambda r:-r['lift'])[:80]
    print('Training purchase models...',flush=True)
    purchase=classification(pd.read_csv(root/files[1][1]))
    wholesale=pd.read_csv(root/files[2][1],encoding='latin1'); spending=[c for c in wholesale if c not in ['Channel','Region']]
    profiles,_=cluster(wholesale[spending])
    for item,df in zip(lineage,[retail,pd.read_csv(root/files[1][1]),wholesale]):
        item['rows']=raw if item['source']=='retail' else len(df); item['columns']=list(df.columns); item['missingCells']=int(df.isna().sum().sum())
    code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report={'runId':hashlib.sha256((''.join(x['sha256'] for x in lineage)+code_hash).encode()).hexdigest()[:12], 'pipelineSHA256':code_hash, 'generatedAt':datetime.now(timezone.utc).isoformat(),'seed':SEED,
      'environment':{'python':platform.python_version(),'pandas':pd.__version__,'sklearn':sklearn.__version__},'lineage':lineage,
      'retail':{'currency':'GBP','start':str(valid.InvoiceDate.min()),'end':str(valid.InvoiceDate.max()),'rawRows':raw,'duplicatesRemoved':dup,'cleanRows':len(valid),'excludedAfterDedup':len(retail)-len(valid),'missingCustomerRows':int(valid['Customer ID'].isna().sum()),'returnRows':len(returns),'returnValue':float((returns.Quantity*returns.Price).sum()),'revenue':float(valid.revenue.sum()),'orders':int(valid.Invoice.nunique()),'customers':int(identified['Customer ID'].nunique()),'units':int(valid.Quantity.sum()),'aov':float(valid.revenue.sum()/valid.Invoice.nunique()),'daily':records(daily),'monthly':records(monthly),
      'countries':records(valid.groupby('Country').revenue.sum().sort_values(ascending=False).head(15).rename('revenue').reset_index()),'products':records(valid.groupby('Description').agg(revenue=('revenue','sum'),units=('Quantity','sum')).sort_values('revenue',ascending=False).head(20).reset_index()),'definition':'Gross positive sales after exact-row deduplication; excludes cancellations, nonpositive quantity/price and invalid dates. Returns reported separately, not subtracted. Missing customer IDs retained in sales and excluded from customer analysis.'},
      'segments':segments,'cohorts':cohorts,'basket':{'baskets':n,'minSupportCount':min_count,'minConfidence':.2,'rules':rules,'method':'Exact frequent 2-itemset counting across invoice baskets. Binary presence per invoice. Minimum support 0.5% (at least 20 invoices), confidence >=20%, lift >1. Rules ranked by lift; top 80 published.','limitation':'Descriptive co-occurrence on full history. No held-out cross-sell uplift, multi-item antecedents, or causal recommendation claims.'},'purchase':purchase,'wholesale':profiles,
      'forecast':forecast(daily)}
    out.mkdir(parents=True,exist_ok=True); (out/'report.json').write_text(json.dumps(clean_json(report),ensure_ascii=True,allow_nan=False),encoding='utf8')
    print(f'Finished in {time.time()-started:.1f}s. Report: {out / "report.json"}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--data-root',type=Path,default=Path('..')); parser.add_argument('--output',type=Path,default=Path('public/data')); args=parser.parse_args(); main(args.data_root,args.output)

