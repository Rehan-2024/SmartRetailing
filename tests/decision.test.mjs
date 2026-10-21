import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {matchProducts,launchPlan,parseCSV,validateTransactions,uploadWorkspace} from '../public/decision.js';
const d=JSON.parse(await readFile(new URL('../public/data/market.json',import.meta.url)));

test('source manifest permits only exact official copies and excludes unknown files',async()=>{
const audit=JSON.parse(await readFile(new URL('../public/data/sources.json',import.meta.url)));
const approved=audit.sources.filter(s=>s.status==='verified-exact');assert.equal(approved.length,3);
for(const r of approved){assert.equal(r.sha256,r.officialMemberSHA256);assert.equal(r.license,'CC BY 4.0');assert.ok(r.archiveURL.startsWith('https://archive.ics.uci.edu/static/public/'));}
assert.equal(approved.find(r=>r.source==='retail').sha256,d.sourceSHA256);
assert.equal(audit.sources.filter(r=>r.status==='excluded-unverified').length,4);
});

test('case slices reconcile and seasonal checks preserve time boundaries',()=>{
const b=d.demand;const rows=b.candidates.find(r=>r.model===b.selected).test.rows;
for(const labels of [['No subsequent sales','Subsequent positive sales'],['United Kingdom','Other recorded countries'],['Intermittent (≤12 active days / 84)','Frequent (>12 active days / 84)']])assert.equal(b.cases.filter(r=>labels.includes(r.case)).reduce((s,r)=>s+r.selected.rows,0),rows);
assert.equal(b.cases.find(r=>r.case==='No subsequent sales').selected.wape,null);
assert.equal(b.temporalStress.length,3);for(const r of b.temporalStress)assert.ok(r.trainLatestLabelIndex<r.testOriginIndex);
assert.match(b.protocol,/REUSED/);
});
test('canonical products, product-country keys and merchandise totals reconcile',()=>{
assert.equal(new Set(d.products.map(p=>p.code)).size,d.products.length);
assert.equal(new Set(d.markets.map(r=>JSON.stringify([r.product,r.country]))).size,d.markets.length);
assert.ok(Math.abs(d.markets.reduce((s,r)=>s+r.revenue,0)-d.audit.merchandiseRevenue)<1);
assert.ok(!d.products.some(p=>['POST','DOT','M','BANK CHARGES'].includes(p.code)));
});
test('challenger selection, release guard and chronological partitions are explicit',()=>{
const b=d.demand;const selected=b.candidates.slice().sort((a,b)=>a.validation.mae-b.validation.mae)[0];assert.equal(b.validationWinner,selected.model);
const incumbent=b.candidates.find(r=>r.model===b.releaseDecision.incumbent);
const passes=selected.test.wape<=incumbent.test.wape&&selected.test.rmse<=incumbent.test.rmse;
assert.equal(b.selected,passes?selected.model:incumbent.model);
assert.ok(b.trainLatestLabelIndex<b.folds[0].originIndex);
assert.deepEqual(b.folds.map(f=>f.role),['validation','validation','calibration','test']);
for(let i=1;i<b.folds.length;i++)assert.ok(b.folds[i-1].forecastEnd<b.folds[i].forecastStart);
for(const m of d.markets)if(m.forecast!==null){assert.ok(m.low>=0);assert.ok(m.high>=m.low);assert.ok(m.recentOrders>=10);}
});
test('browser analogue matches agree with Python TF-IDF and price weighting',()=>{
assert.ok(d.parityCases.length>0);
for(const c of d.parityCases){const matches=matchProducts(d,c.description,c.price);assert.equal(matches.length,c.indices.length);for(let i=0;i<matches.length;i++){assert.equal(matches[i].index,c.indices[i]);assert.ok(Math.abs(matches[i].score-c.scores[i])<1e-4);}}
});
test('new-product scenarios abstain for unsupported matches and never invent countries',()=>{
assert.equal(launchPlan(d,[],5,d.countries).supported,false);
assert.equal(matchProducts(d,'quantum submarine turbine',30000).length,0);
const matches=matchProducts(d,'red heart bag',2);const plan=launchPlan(d,matches,2,d.countries);
for(const c of plan.countries){assert.ok(d.countries.includes(c.country));assert.ok(c.analogues>=3);assert.ok(c.orders>=10);assert.ok(Math.abs(c.revenue-c.units*2)<1e-8);assert.ok(Math.abs(c.contributions.reduce((s,r)=>s+r.weight,0)-1)<1e-8);}
assert.equal(launchPlan(d,matches,2,['A country absent from source']).supported,false);
const mugMatches=matchProducts(d,'ceramic floral mug',4.5);assert.ok(mugMatches.every(m=>/\b(mug|mugs|cup|cups)\b/i.test(m.product.description)));
assert.equal(launchPlan({...d,period:{start:'2026-01-01',end:'2026-01-12'}},matches,2,d.countries).supported,false);
});
test('CSV handles quoted commas, duplicate rows, malformed dates and returns',()=>{
const csv='OrderID,ProductCode,ProductName,Date,Quantity,UnitPrice,Country\nA,P,"Mug, floral",2026-01-01,2,4,India\nA,P,"Mug, floral",2026-01-01,2,4,India\nB,P,Mug,2026-02-30,2,4,India\nC1,P,Mug,2026-01-02,-1,4,India\n';
const checked=validateTransactions(parseCSV(csv));assert.equal(checked.valid.length,1);assert.equal(checked.duplicates,1);assert.equal(checked.invalid,1);assert.equal(checked.returns.length,1);
assert.throws(()=>parseCSV('a,b\n"unclosed,b'));
assert.throws(()=>validateTransactions([{ProductName:'Missing required fields'}]));
});
test('upload workspace is isolated, uses stable IDs and changes product/country totals',()=>{
const rows=[{OrderID:'1',ProductCode:'P',ProductName:'Old name',Date:'2026-01-01',Quantity:2,UnitPrice:4,Country:'India'},{OrderID:'2',ProductCode:'P',ProductName:'New name',Date:'2026-01-02',Quantity:3,UnitPrice:5,Country:'India'}];
const w=uploadWorkspace(validateTransactions(rows),'INR');assert.equal(w.isUpload,true);assert.equal(w.products.length,1);assert.equal(w.products[0].units,5);assert.equal(w.markets[0].revenue,23);assert.equal(w.markets[0].orders,2);assert.equal(w.currency,'INR');assert.equal(w.markets[0].forecast,null);assert.ok(w.textModel.vocabulary.old!==undefined);assert.equal(w.coldStart.beatsCountryBaseline,false);assert.ok(w.markets[0].recent28>0);
});
test('cold-start benchmark reports coverage and retains honest baseline gating',()=>{
const c=d.coldStart;const selected=c.results.find(r=>r.role==='test'&&r.neighbors===c.selectedNeighbors);
assert.equal(c.beatsCountryBaseline,selected.supported>=20&&selected.topCountryHitRate>selected.baselineHitRate);
for(const r of c.results){assert.ok(r.supported<=r.total);assert.ok(r.coverage>=0&&r.coverage<=1);}
assert.equal(c.folds.at(-1).role,'test');
});
