"""Build merchandise-only decision artifacts and forward-only benchmarks.

No outcomes from an unseen product are used in its analogue prediction.
The analogue benchmark is retrospective, not evidence of launch causality.
"""
import argparse, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor

NON_PRODUCTS={'ADJUST','ADJUST2','AMAZONFEE','B','BANK CHARGES','C2','CRUK','D','DOT','M','PADS','POST','S','TEST001','TEST002'}
STOP='a an the and or of for with in on to from by at is it this that set pack assorted'.split()
TYPES=[['mug','mugs','cup','cups'],['bag','bags','tote','totes'],['plate','plates','platter'],['bowl','bowls'],['bottle','bottles'],['candle','candles','tealight','tealights'],['cushion','cushions','pillow'],['necklace','necklaces'],['clock','clocks'],['box','boxes','tin','tins'],['card','cards'],['notebook','notebooks','journal'],['holder','holders'],['vase','vases'],['basket','baskets'],['lamp','lamps'],['toy','toys'],['ornament','ornaments']]
HORIZON=28

def safe(x):
    if isinstance(x,dict): return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [safe(v) for v in x]
    if isinstance(x,np.ndarray): return safe(x.tolist())
    if isinstance(x,np.generic): return safe(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x

def feature_matrix(units,origin,prices,first):
    last28=units[:,origin-28:origin].sum(1)
    prev28=units[:,origin-56:origin-28].sum(1)
    last84=units[:,origin-84:origin].sum(1)
    recent7=units[:,origin-7:origin].sum(1)
    active=(units[:,origin-84:origin]>0).sum(1)
    return np.column_stack([np.log1p(last28),np.log1p(prev28),np.log1p(last84),np.log1p(recent7),active,np.maximum(0,origin-first),np.log1p(prices),np.full(len(prices),(origin%365)/365)])

def baseline_predictions(u,o):
    a=u[:,o-28:o].sum(1); b=u[:,o-56:o-28].sum(1)
    return {'Last 28 days':a,'84-day daily mean':u[:,o-84:o].sum(1)/3,'Damped trend':a*np.clip(np.sqrt((a+1)/(b+1)),.5,1.5)}

def metrics(actual,pred):
    error=np.abs(actual-pred)
    return {'mae':float(error.mean()),'wape':float(error.sum()/max(actual.sum(),1)),'rmse':float(np.sqrt(((actual-pred)**2).mean())),'rows':len(actual)}

def demand_benchmark(units,orders,prices):
    n,days=units.shape; first=np.argmax(units>0,axis=1); evaluation=list(range(days-112,days,28))
    train_origins=list(range(112,evaluation[0]-27,28)); xx=[]; yy=[]
    def eligible(o): return (first<=o-84)&(orders[:,o-84:o].sum(1)>=10)
    for o in train_origins:
        mask=eligible(o); xx.append(feature_matrix(units,o,prices,first)[mask]); yy.append(units[mask,o:o+28].sum(1))
    x=np.concatenate(xx); y=np.concatenate(yy)
    models={
        'Gradient boosting (log units)':HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=15,l2_regularization=5,random_state=42),
        'Gradient boosting (Poisson)':HistGradientBoostingRegressor(loss='poisson',max_iter=160,max_leaf_nodes=15,l2_regularization=10,random_state=42)}
    for name,model in models.items(): model.fit(x,np.log1p(y) if 'log units' in name else y)
    results={name:[] for name in list(baseline_predictions(units,evaluation[0]))+list(models)}; folds=[]
    for o in evaluation:
        mask=eligible(o); actual=units[mask,o:o+28].sum(1); pred=baseline_predictions(units,o)
        for name,model in models.items():
            values=model.predict(feature_matrix(units,o,prices,first)); pred[name]=np.maximum(0,np.expm1(values) if 'log units' in name else values)
        for name in results: results[name].append((actual,pred[name][mask]))
        folds.append({'originIndex':o,'rows':int(mask.sum()),'role':['validation','validation','calibration','test'][len(folds)]})
    selected=min(results,key=lambda name:sum(np.abs(a-p).sum() for a,p in results[name][:2]))
    cal_a,cal_p=results[selected][2]; residual=(cal_a-cal_p)/(cal_p+1)
    qlo,qhi=np.quantile(residual,[.1,.9]); test_a,test_p=results[selected][3]
    low=np.maximum(0,test_p+qlo*(test_p+1)); high=np.maximum(0,test_p+qhi*(test_p+1))
    final=baseline_predictions(units,days)
    for name,m in models.items():
        v=m.predict(feature_matrix(units,days,prices,first)); final[name]=np.maximum(0,np.expm1(v) if 'log units' in name else v)
    forecast=final[selected]; supported=eligible(days)
    summary=[]
    for name in results:
        va=np.concatenate([r[0] for r in results[name][:2]]); vp=np.concatenate([r[1] for r in results[name][:2]])
        summary.append({'model':name,'validation':metrics(va,vp),'test':metrics(*results[name][3])})
    return {'selected':selected,'candidates':summary,'folds':folds,'trainRows':len(y),'trainLatestLabelIndex':train_origins[-1]+27,'coverage':float(((test_a>=low)&(test_a<=high)).mean()),'intervalNominal':.8,'calibrationRows':len(cal_a),'testPoints':[{'actual':float(a),'predicted':float(p)} for a,p in zip(test_a[:200],test_p[:200])],
        'protocol':'Five candidates. Two chronological 28-day validation windows select by total absolute unit error. Third window calibrates a pooled residual band. Fourth is untouched final test. Training labels finish before validation. Eligibility: ≥84 days of history and ≥10 orders in the preceding 84 days. No hyperparameters changed after seeing the test.',
        'limitation':'One retailer, recorded sales rather than unconstrained demand; stockouts, distribution and promotions are unknown. Residual bands are pooled empirical bands, not guaranteed future coverage.'},forecast,np.maximum(0,forecast+qlo*(forecast+1)),np.maximum(0,forecast+qhi*(forecast+1)),supported

def analogues(vectorizer,matrix,descriptions,prices,target_desc,target_price,k=8,exclude=None):
    query=vectorizer.transform([target_desc]); cosine=(matrix@query.T).toarray().ravel()
    score=cosine*np.exp(-np.abs(np.log(np.maximum(prices,.01)/max(target_price,.01))))
    import re
    tokens=set(re.findall(r'\b[a-z0-9]{2,}\b',target_desc.lower()))
    families=[set(group) for group in TYPES if tokens.intersection(group)]
    if families:
        for i,desc in enumerate(descriptions):
            other=set(re.findall(r'\b[a-z0-9]{2,}\b',str(desc).lower()))
            if not any(other.intersection(group) for group in families): score[i]=0
    if exclude is not None: score[exclude]=0
    ix=np.argsort(-score)[:k]; ix=ix[(cosine[ix]>=.15)&(score[ix]>.05)]
    return ix,score[ix],cosine[ix]

def cold_start_benchmark(sales,dates):
    # First appearance is not proof of a true launch; absent/failed products are missing.
    origins=[pd.Timestamp('2010-06-01'),pd.Timestamp('2010-10-01'),pd.Timestamp('2011-03-01'),pd.Timestamp('2011-07-01')]
    first=sales.groupby('StockCode').InvoiceDate.min(); candidate_results={k:[] for k in [5,8,12]}; baseline_results=[]; folds=[]
    for fold,o in enumerate(origins):
        before=sales[sales.InvoiceDate<o]; recent=before[before.InvoiceDate>=o-pd.Timedelta(days=84)]
        counts=recent.groupby('StockCode').Invoice.nunique(); ids=counts[counts>=10].index
        ids=ids[first.reindex(ids)<=o-pd.Timedelta(days=84)]
        meta=before[before.StockCode.isin(ids)].sort_values('InvoiceDate').groupby('StockCode').agg(Description=('Description','last'),price=('Price','median'))
        ids=meta.index.tolist(); desc=meta.Description.fillna('').astype(str).tolist(); prices=meta.price.to_numpy()
        if len(ids)<20: continue
        vec=TfidfVectorizer(stop_words=STOP,token_pattern=r'(?u)\b[a-z0-9]{2,}\b',lowercase=True); matrix=vec.fit_transform(desc)
        eligible_recent=recent[recent.StockCode.isin(ids)]
        country=eligible_recent.groupby(['StockCode','Country']).Quantity.sum().unstack(fill_value=0).reindex(ids,fill_value=0)/3
        country_orders=eligible_recent.groupby(['StockCode','Country']).Invoice.nunique().unstack(fill_value=0).reindex(index=ids,columns=country.columns,fill_value=0)
        globals_=recent.groupby('Country').Quantity.sum().sort_values(ascending=False); baseline_country=globals_.index[0]
        target_ids=first[(first>=o)&(first<o+pd.Timedelta(days=28))].index
        tests=sales[sales.StockCode.isin(target_ids)&(sales.InvoiceDate<o+pd.Timedelta(days=56))].sort_values('InvoiceDate')
        fold_count=0
        for code,group in tests.groupby('StockCode',sort=True):
            group=group[group.InvoiceDate<group.InvoiceDate.min()+pd.Timedelta(days=28)]
            row=group.iloc[0]; actual=group.groupby('Country').Quantity.sum(); actual_best=actual.idxmax(); target_price=float(row.Price)
            base_units=float(country.sum(axis=1).median()); base_actual=float(actual.sum())
            baseline_results.append({'fold':fold,'hit':int(baseline_country==actual_best),'error':abs(base_actual-base_units),'actual':base_actual})
            for k in candidate_results:
                ix,weights,_=analogues(vec,matrix,desc,prices,str(row.Description),target_price,k)
                if len(ix)<3: candidate_results[k].append({'fold':fold,'supported':False,'actual':base_actual}); continue
                rates=country.iloc[ix].to_numpy(); recent_orders=country_orders.iloc[ix].to_numpy(); active=recent_orders>0
                eligible=(active.sum(axis=0)>=3)&(recent_orders.sum(axis=0)>=10)
                if not eligible.any(): candidate_results[k].append({'fold':fold,'supported':False,'actual':base_actual}); continue
                weighted=(rates*weights[:,None]).sum(axis=0)/np.maximum((active*weights[:,None]).sum(axis=0),1e-12)
                weighted[~eligible]=0
                best=country.columns[int(np.argmax(weighted))]; total=float(weighted.sum())
                candidate_results[k].append({'fold':fold,'supported':True,'hit':int(best==actual_best),'error':abs(base_actual-total),'actual':base_actual,'baselineHit':int(baseline_country==actual_best),'baselineError':abs(base_actual-base_units),'code':str(code)})
            fold_count+=1
        folds.append({'cutoff':str(o.date()),'lastHistoryDate':str(before.InvoiceDate.max().date()),'role':'test' if fold==3 else 'validation','newProducts':fold_count})
    # Select coverage-aware country hit rate on validation, never the last fold.
    def objective(rows):
        rows=[r for r in rows if r['fold']<3]
        return sum(r.get('hit',0) for r in rows)/max(len(rows),1)
    selected=max(candidate_results,key=lambda k:objective(candidate_results[k])); stats=[]
    for k,rows in candidate_results.items():
        for role in ['validation','test']:
            subset=[r for r in rows if (r['fold']==3)==(role=='test')]; supported=[r for r in subset if r['supported']]
            stats.append({'neighbors':k,'role':role,'total':len(subset),'supported':len(supported),'coverage':len(supported)/max(len(subset),1),'topCountryHitRate':sum(r['hit'] for r in supported)/max(len(supported),1),'baselineHitRate':sum(r['baselineHit'] for r in supported)/max(len(supported),1),'unitMAE':sum(r['error'] for r in supported)/max(len(supported),1),'baselineUnitMAE':sum(r['baselineError'] for r in supported)/max(len(supported),1)})
    selected_test=next(r for r in stats if r['neighbors']==selected and r['role']=='test')
    validated=selected_test['supported']>=20 and selected_test['topCountryHitRate']>selected_test['baselineHitRate']
    return {'selectedNeighbors':selected,'results':stats,'folds':folds,'beatsCountryBaseline':validated,
        'protocol':'Historical catalogue-entry simulation. At each cutoff, compare products first observed in the following 28 days with established products known before cutoff. Use only supplied description and first observed price as query metadata. Outcomes cover the first 28 recorded days per target. Three earlier cohorts select neighbor count; July 2011 cohort is final test. Unit estimates use preceding 84-day rates scaled to 28 days. As in the app, each supported country requires three active analogues and ten orders; average only active analogues. Unit MAE compares the sum over supported countries with target total recorded units.',
        'limitation':'First recorded sale is not a verified launch. Products with no sales never enter this dataset. Metadata must represent what was known before launch. No advertising, availability, distribution, competition or market-size features exist. This is not a launch-success probability or a causal expansion forecast.'}

def main(root,out):
    path=root/'transactions/online_retail_II.xlsx'; print('Reading source workbook for product-market build...',flush=True)
    frame=pd.concat(pd.read_excel(path,sheet_name=None).values(),ignore_index=True); raw=len(frame); duplicates=int(frame.duplicated().sum()); frame=frame.drop_duplicates()
    frame.InvoiceDate=pd.to_datetime(frame.InvoiceDate,errors='coerce'); frame.StockCode=frame.StockCode.astype(str).str.strip(); frame.Description=frame.Description.astype('string').str.strip()
    merch=~frame.StockCode.str.upper().isin(NON_PRODUCTS)&~frame.Description.str.upper().fillna('').str.contains('POSTAGE|DOTCOM|BANK CHARGE|AMAZON FEE|MANUAL|CARRIAGE|ADJUSTMENT',regex=True)
    returns=frame[merch & (frame.Quantity.lt(0)|frame.Invoice.astype(str).str.upper().str.startswith('C')) & frame.Price.gt(0)].copy()
    sales=frame[merch & frame.Quantity.gt(0)&frame.Price.gt(0)&frame.InvoiceDate.notna()&~frame.Invoice.astype(str).str.upper().str.startswith('C')].copy()
    sales['Revenue']=sales.Quantity*sales.Price; sales['date']=sales.InvoiceDate.dt.floor('D'); sales.Country=sales.Country.fillna('Unknown')
    meta=sales.sort_values('InvoiceDate').groupby('StockCode').agg(description=('Description','last'),price=('Price','median'),first=('InvoiceDate','min'),last=('InvoiceDate','max'),units=('Quantity','sum'),revenue=('Revenue','sum'),orders=('Invoice','nunique')).reset_index()
    meta.description=meta.description.fillna('Unnamed product'); products=meta.StockCode.tolist(); product_idx={c:i for i,c in enumerate(products)}
    dates=pd.date_range(sales.date.min(),sales.date.max()); day_idx={v:i for i,v in enumerate(dates)}
    # Initial observed price is available at every eligible historical origin.
    # A full-history median here would leak future price information into training.
    pairs=sales.sort_values('InvoiceDate').groupby(['StockCode','Country']).agg(units=('Quantity','sum'),revenue=('Revenue','sum'),orders=('Invoice','nunique'),customers=('Customer ID','nunique'),price=('Price','first'),first=('date','min')).reset_index()
    pair_keys=list(zip(pairs.StockCode,pairs.Country)); pair_idx={key:i for i,key in enumerate(pair_keys)}
    units=np.zeros((len(pairs),len(dates))); orders=np.zeros_like(units)
    grouped=sales.groupby(['StockCode','Country','date']).agg(units=('Quantity','sum'),orders=('Invoice','nunique'))
    for (code,country,day),r in grouped.iterrows():
        i=pair_idx[(code,country)]; j=day_idx[day]; units[i,j]=r['units']; orders[i,j]=r['orders']
    print(f'Evaluating five demand candidates across {len(pairs)} product-country series...',flush=True)
    demand,pred,lo,hi,support=demand_benchmark(units,orders,pairs.price.to_numpy())
    for f in demand['folds']:
        o=f['originIndex']; f['historyEnd']=str(dates[o-1].date()); f['forecastStart']=str(dates[o].date()); f['forecastEnd']=str(dates[o+27].date())
    print('Evaluating genuinely withheld catalogue-entry products...',flush=True)
    cold=cold_start_benchmark(sales,dates)
    vec=TfidfVectorizer(stop_words=STOP,token_pattern=r'(?u)\b[a-z0-9]{2,}\b',lowercase=True); matrix=vec.fit_transform(meta.description.astype(str))
    vocabulary=vec.vocabulary_; sparse=[]
    for i in range(len(products)):
        row=matrix.getrow(i); sparse.append([[int(j),round(float(v),7)] for j,v in zip(row.indices,row.data)])
    ret=returns.assign(return_units=returns.Quantity.abs()).groupby(['StockCode','Country']).return_units.sum().to_dict()
    monthly=sales.groupby(['StockCode','Country',sales.InvoiceDate.dt.strftime('%Y-%m')]).agg(units=('Quantity','sum'),revenue=('Revenue','sum')).reset_index()
    month_map={}
    for _,r in monthly.iterrows(): month_map.setdefault((r.StockCode,r.Country),[]).append([r.InvoiceDate,round(r.units,2),round(r.revenue,2)])
    entries=[]
    for i,r in pairs.iterrows():
        return_units=float(ret.get((r.StockCode,r.Country),0)); recent=float(units[i,-84:].sum()/3)
        entries.append({'product':product_idx[r.StockCode],'country':r.Country,'units':float(r.units),'revenue':round(r.revenue,2),'orders':int(r.orders),'customers':int(r.customers),'returnRate':return_units/(r.units+return_units),'recent28':round(recent,2),'recentOrders':int(orders[i,-84:].sum()),'forecast':round(float(pred[i]),2) if support[i] else None,'low':round(float(lo[i]),2) if support[i] else None,'high':round(float(hi[i]),2) if support[i] else None,'monthly':month_map.get((r.StockCode,r.Country),[])})
    # Frequent pairs restricted to genuine merchandise; top co-occurrences per product.
    from collections import Counter
    from itertools import combinations
    baskets=sales.groupby('Invoice').StockCode.agg(lambda x:sorted(set(x))); singles=Counter(x for b in baskets for x in b); keep={x for x,n in singles.items() if n>=30}; counts=Counter(pair for b in baskets for pair in combinations([x for x in b if x in keep],2)); associations={}
    for (a,b),count in counts.items():
        if count<20: continue
        for left,right in [(a,b),(b,a)]:
            confidence=count/singles[left]; lift=confidence/(singles[right]/len(baskets))
            if confidence>=.1 and lift>1: associations.setdefault(str(product_idx[left]),[]).append({'product':product_idx[right],'orders':count,'confidence':round(confidence,4),'lift':round(lift,3)})
    associations={k:sorted(v,key=lambda r:-r['orders'])[:6] for k,v in associations.items()}
    parity=[]
    recent_product_orders=sales[sales.date>=dates[-1]-pd.Timedelta(days=83)].groupby('StockCode').Invoice.nunique().reindex(meta.StockCode,fill_value=0).to_numpy()
    exclude_final=((dates[-1]-meta['first']).dt.days.to_numpy()<84)|(recent_product_orders<10)
    for text,price in [('ceramic floral mug',4.5),('red heart bag',2.0)]:
        ix,weights,cos=analogues(vec,matrix,meta.description.tolist(),meta.price.to_numpy(),text,price,cold['selectedNeighbors'],exclude=exclude_final)
        parity.append({'description':text,'price':price,'indices':ix.tolist(),'scores':weights.tolist()})
    report={'version':2,'generatedAt':datetime.now(timezone.utc).isoformat(),'sourceSHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'pipelineSHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'parityCases':parity,'period':{'start':str(dates[0].date()),'end':str(dates[-1].date()),'forecastStart':str((dates[-1]+pd.Timedelta(days=1)).date()),'forecastEnd':str((dates[-1]+pd.Timedelta(days=28)).date())},'audit':{'rawRows':raw,'duplicatesRemoved':duplicates,'merchandiseSalesRows':len(sales),'excludedNonMerchandiseRows':int((~merch).sum()),'missingCustomerSalesRows':int(sales['Customer ID'].isna().sum()),'merchandiseRevenue':float(sales.Revenue.sum())},
        'products':[{'code':r.StockCode,'description':r.description,'price':round(r.price,4),'orders':int(r.orders),'units':float(r.units),'revenue':round(r.revenue,2),'first':str(r['first'].date()),'last':str(r['last'].date()),'vector':sparse[i]} for i,r in meta.iterrows()],
        'countries':sorted(pairs.Country.unique().tolist()),'markets':entries,'textModel':{'vocabulary':vocabulary,'idf':vec.idf_.tolist(),'stopWords':STOP,'typeFamilies':TYPES,'minCosine':.15,'minScore':.05},'demand':demand,'coldStart':cold,'associations':associations,
        'boundaries':['Historical UK retailer observations, not current global market research.','Unseen countries are not scored. No city, competitor, promotion, inventory or distribution data.','New-product outputs are analogue scenarios under comparable exposure, not promised launch demand.','Revenue scenarios equal estimated units × entered unit price; no elasticity or profit model.','Evidence counts are not statistical confidence. No arbitrary 100-point opportunity score.']}
    out.mkdir(parents=True,exist_ok=True); (out/'market.json').write_text(json.dumps(safe(report),allow_nan=False,separators=(',',':')),encoding='utf8'); print(f'Wrote {out / "market.json"}. Selected demand method: {demand["selected"]}. Analogue country baseline beaten: {cold["beatsCountryBaseline"]}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=Path('..'));p.add_argument('--output',type=Path,default=Path('public/data'));a=p.parse_args();main(a.data_root,a.output)
