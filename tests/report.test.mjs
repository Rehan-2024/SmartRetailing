import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {predict} from '../public/inference.js';
const d=JSON.parse(await readFile(new URL('../public/data/report.json',import.meta.url)));
test('browser application parses as an async module body',async()=>{
const source=await readFile(new URL('../public/app.js',import.meta.url),'utf8');
const AsyncFunction=Object.getPrototypeOf(async function(){}).constructor;
assert.doesNotThrow(()=>new AsyncFunction(source.replace(/^\uFEFF?import[^;]+;/,'')));
});
test('retail audit reconciles input to valid sales',()=>{
assert.equal(d.retail.rawRows,d.retail.duplicatesRemoved+d.retail.excludedAfterDedup+d.retail.cleanRows);
assert.ok(Math.abs(d.retail.monthly.reduce((s,r)=>s+r.revenue,0)-d.retail.revenue)<.01);
assert.ok(Math.abs(d.retail.aov*d.retail.orders-d.retail.revenue)<.01);
});
test('classification split reconciles and confusion matrices cover test set',()=>{
const p=d.purchase;assert.equal(p.rows,Object.values(p.split).reduce((a,b)=>a+b,0));
for(const r of p.test){assert.equal(r.confusionMatrix.flat().reduce((a,b)=>a+b,0),p.split.test);assert.ok(r.rocAUC>=0&&r.rocAUC<=1);assert.ok(r.averagePrecision>=0&&r.averagePrecision<=1);}
assert.equal(p.selected,p.validation.filter(r=>r.model!=='Prevalence baseline').sort((a,b)=>b.validationAveragePrecision-a.validationAveragePrecision)[0].model);
});
test('browser inference matches evaluated sklearn pipeline to floating-point tolerance',()=>{
for(const c of d.purchase.inference.parityCases){assert.ok(Math.abs(predict(d.purchase.inference,c.input).probability-c.probability)<1e-10);}
assert.throws(()=>predict(d.purchase.inference,{...d.purchase.inference.defaults,Administrative:'invalid'}));
});
test('forecast windows are chronological, disjoint and precede future forecasts',()=>{
for(const m of d.forecast.metrics){for(let i=0;i<m.folds.length;i++){const f=m.folds[i];assert.ok(f.trainEnd<f.testStart);assert.ok(f.testStart<=f.testEnd);if(i)assert.ok(m.folds[i-1].testEnd<f.testStart);}}
assert.ok(d.forecast.future[0].date>d.retail.end.slice(0,10));assert.equal(d.forecast.future.length,28);
const chosen=d.forecast.metrics.slice().sort((a,b)=>a.folds.slice(0,3).reduce((s,r)=>s+r.mae,0)-b.folds.slice(0,3).reduce((s,r)=>s+r.mae,0))[0];assert.equal(d.forecast.selected,chosen.model);
});
test('cohorts respect observation boundaries and rates',()=>{
const last=d.retail.end.slice(0,7).split('-').map(Number);const lastMonth=last[0]*12+last[1];
for(const c of d.cohorts){assert.equal(c.retention[0],1);const [year,month]=c.cohort.split('-').map(Number);for(const [i,v] of c.retention.entries()){assert.ok(v===null||(v>=0&&v<=1));assert.equal(v===null,year*12+month+i>lastMonth);}}
});
test('all source hashes present and exported report omits direct identifiers',()=>{
assert.equal(d.lineage.length,3);for(const s of d.lineage)assert.match(s.sha256,/^[a-f0-9]{64}$/);
for(const group of [d.segments,d.wholesale]){assert.equal(group.profiles.reduce((s,r)=>s+r.customers,0),group===d.segments?d.retail.customers:440);for(const p of group.points)assert.deepEqual(Object.keys(p).sort(),['cluster','x','y']);}
});
