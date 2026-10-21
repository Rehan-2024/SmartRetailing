// Shared, deterministic business logic. No network or hidden scoring service.
export function encode(text,model){
  const counts=new Map();const stop=new Set(model.stopWords);
  for(const token of (text.toLowerCase().match(/\b[a-z0-9]{2,}\b/g)||[])){if(!stop.has(token)&&Object.hasOwn(model.vocabulary,token)){const j=model.vocabulary[token];counts.set(j,(counts.get(j)||0)+1);}}
  const raw=[...counts].map(([j,n])=>[j,n*model.idf[j]]);const norm=Math.sqrt(raw.reduce((s,[,v])=>s+v*v,0));return new Map(raw.map(([j,v])=>[j,v/(norm||1)]));
}
export function matchProducts(data,description,price,excludedCode=null){
  const query=encode(description,data.textModel);
  const tokens=new Set(description.toLowerCase().match(/\b[a-z0-9]{2,}\b/g)||[]);
  const families=(data.textModel.typeFamilies||[]).filter(group=>group.some(word=>tokens.has(word)));
  const recentOrders=new Map();for(const r of data.markets)recentOrders.set(r.product,(recentOrders.get(r.product)||0)+r.recentOrders);
  const cutoff=new Date(data.period.end+'T00:00:00Z').getTime()-84*86400000;
  return data.products.map((p,i)=>{const cosine=p.vector.reduce((s,[j,v])=>s+(query.get(j)||0)*v,0);const priceFactor=Math.exp(-Math.abs(Math.log(Math.max(.01,p.price)/Math.max(.01,price))));return {index:i,product:p,cosine,priceFactor,score:cosine*priceFactor};})
    .filter(m=>m.product.code!==excludedCode&&new Date(m.product.first+'T00:00:00Z').getTime()<=cutoff&&(recentOrders.get(m.index)||0)>=10&&m.cosine>=data.textModel.minCosine&&m.score>data.textModel.minScore&&(!families.length||families.some(group=>group.some(word=>(m.product.description.toLowerCase().match(/\b[a-z0-9]{2,}\b/g)||[]).includes(word)))))
    .sort((a,b)=>b.score-a.score).slice(0,data.coldStart.selectedNeighbors);
}
export function launchPlan(data,matches,price,feasible){
  if((new Date(data.period.end)-new Date(data.period.start))/86400000<84)return {supported:false,reason:'At least 84 days of recorded company history are required before an analogue scenario is calculated.',countries:[]};
  if(matches.length<3)return {supported:false,reason:'Fewer than three comparable products meet the text and price checks. Upload pilot sales or choose a better-supported comparison.',countries:[]};
  const byProduct=new Map(matches.map(m=>[m.index,m]));const groups=new Map();
  for(const row of data.markets){const m=byProduct.get(row.product);if(!m||row.recentOrders<1||!feasible.includes(row.country))continue;if(!groups.has(row.country))groups.set(row.country,[]);groups.get(row.country).push({row,match:m});}
  const countries=[];
  for(const [country,rows] of groups){if(rows.length<3||rows.reduce((s,x)=>s+x.row.recentOrders,0)<10)continue;
    const total=rows.reduce((s,x)=>s+x.match.score,0);
    // Use recent 84-day mean scaled to 28 days, exactly as in cold-start backtest.
    const units=rows.reduce((s,x)=>s+x.row.recent28*x.match.score,0)/total;
    const sample=rows.map(x=>x.row.recent28).sort((a,b)=>a-b);
    const returnRate=rows.reduce((s,x)=>s+x.row.returnRate*x.match.score,0)/total;
    countries.push({country,units,revenue:units*price,low:sample[0],high:sample.at(-1),analogues:rows.length,orders:rows.reduce((s,x)=>s+x.row.recentOrders,0),returnRate,
      contributions:rows.map(x=>({product:x.row.product,weight:x.match.score/total,units:x.row.recent28}))});
  }
  countries.sort((a,b)=>b.units-a.units);
  return {supported:countries.length>0,reason:countries.length?'':'No selected country has three active analogues and ten recorded orders in the last 84 days.',countries};
}
export function parseCSV(text){
  text=text.replace(/^\uFEFF/,'');const delimiter=text.split(/\r?\n/,1)[0].includes('\t')?'\t':',';const rows=[];let row=[],value='',quoted=false;
  for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){value+='"';i++;}else quoted=!quoted;}else if(c===delimiter&&!quoted){row.push(value);value='';}else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(value);if(row.some(v=>v!==''))rows.push(row);row=[];value='';}else value+=c;}
  if(quoted)throw Error('An opening quotation mark has no closing quotation mark.');row.push(value);if(row.some(v=>v!==''))rows.push(row);
  if(rows.length<2)throw Error('Include a header and at least one transaction row.');const header=rows.shift().map(s=>s.trim());if(new Set(header).size!==header.length)throw Error('Column headers must be unique.');
  return rows.map((values,i)=>{if(values.length!==header.length)throw Error('Column count differs on row '+(i+2));return Object.fromEntries(header.map((h,j)=>[h,values[j]]));});
}
export function validateTransactions(raw){
  const aliases={stockcode:'code',productcode:'code',productid:'code',description:'description',productname:'description',product:'description',invoicedate:'date',date:'date',orderdate:'date',quantity:'quantity',units:'quantity',unitssold:'quantity',price:'price',unitprice:'price',country:'country',city:'city',invoice:'order',ordernumber:'order',orderid:'order',currency:'currency'};
  if(!raw.length)throw Error('No transaction rows found.');const headers=Object.keys(raw[0]);const mapped=headers.map(h=>aliases[h.toLowerCase().replace(/[^a-z0-9]/g,'')]);
  const required=['code','description','date','quantity','price','country','order'];const missing=required.filter(c=>!mapped.includes(c));if(missing.length)throw Error('Missing required columns: '+missing.join(', '));
  for(const key of required)if(mapped.filter(k=>k===key).length>1)throw Error('More than one column maps to '+key+'. Keep one authoritative column.');
  const seen=new Set();const valid=[];const returns=[];let invalid=0,duplicates=0;const reasons=[];
  for(const [i,r] of raw.entries()){const row={};headers.forEach((h,j)=>{if(mapped[j])row[mapped[j]]=String(r[h]??'').trim();});
    const stamp=JSON.stringify(required.map(k=>row[k]));if(seen.has(stamp)){duplicates++;continue;}seen.add(stamp);
    const quantity=Number(row.quantity),price=Number(row.price);const iso=row.date.match(/^\d{4}-\d{2}-\d{2}(?:[ T].*)?$/);const date=new Date(row.date);
    const roundtrip=Number.isFinite(date.getTime())?date.toISOString().slice(0,10):'';
    if(required.some(k=>!row[k])||!iso||roundtrip!==row.date.slice(0,10)||!Number.isFinite(quantity)||!Number.isFinite(price)||quantity===0||price<=0){invalid++;if(reasons.length<8)reasons.push('Row '+(i+2)+': require nonempty fields, ISO date, nonzero quantity and positive unit price.');continue;}
    const item={...row,date:roundtrip,quantity,price};if(quantity<0||row.order.toUpperCase().startsWith('C'))returns.push(item);else valid.push(item);
  }
  return {valid,returns,invalid,duplicates,reasons,received:raw.length};
}
export function uploadWorkspace(checked,currency,typeFamilies=[]){
  const productMap=new Map(),markets=new Map();let end='',start='9999-99-99';
  for(const row of checked.valid){if(!productMap.has(row.code))productMap.set(row.code,{code:row.code,description:row.description,price:row.price,units:0,revenue:0,orders:new Set(),first:row.date,last:row.date});const p=productMap.get(row.code);p.units+=row.quantity;p.revenue+=row.quantity*row.price;p.orders.add(row.order);p.first=p.first<row.date?p.first:row.date;p.last=p.last>row.date?p.last:row.date;end=end>row.date?end:row.date;start=start<row.date?start:row.date;
    const key=JSON.stringify([row.code,row.country]);if(!markets.has(key))markets.set(key,{code:row.code,country:row.country,units:0,revenue:0,orders:new Set(),customers:null,returnUnits:0,forecast:null,low:null,high:null,monthly:new Map()});const m=markets.get(key);m.units+=row.quantity;m.revenue+=row.quantity*row.price;m.orders.add(row.order);const month=row.date.slice(0,7);const v=m.monthly.get(month)||[month,0,0];v[1]+=row.quantity;v[2]+=row.quantity*row.price;m.monthly.set(month,v);}
  for(const row of checked.returns){const m=markets.get(JSON.stringify([row.code,row.country]));if(m)m.returnUnits+=Math.abs(row.quantity);}
  const products=[...productMap.values()].map(p=>({...p,price:p.revenue/p.units,orders:p.orders.size}));const idx=new Map(products.map((p,i)=>[p.code,i]));
  const recentStart=new Date(end+'T00:00:00Z');recentStart.setUTCDate(recentStart.getUTCDate()-83);const cutoff=recentStart.toISOString().slice(0,10);const recent=new Map();
  for(const row of checked.valid){if(row.date<cutoff)continue;const key=JSON.stringify([row.code,row.country]);if(!recent.has(key))recent.set(key,{units:0,orders:new Set()});const r=recent.get(key);r.units+=row.quantity;r.orders.add(row.order);}
  const stopWords='a an the and or of for with in on to from by at is it this that set pack assorted'.split(' ');const stops=new Set(stopWords),vocabulary={},counts=[],df=[];
  for(const product of products){const count=new Map();for(const word of product.description.toLowerCase().match(/\b[a-z0-9]{2,}\b/g)||[]){if(stops.has(word))continue;if(!Object.hasOwn(vocabulary,word))vocabulary[word]=Object.keys(vocabulary).length;const j=vocabulary[word];count.set(j,(count.get(j)||0)+1);}for(const j of count.keys())df[j]=(df[j]||0)+1;counts.push(count);}
  const idf=df.map(n=>Math.log((1+products.length)/(1+n))+1);
  products.forEach((p,i)=>{const raw=[...counts[i]].map(([j,n])=>[j,n*idf[j]]);const norm=Math.sqrt(raw.reduce((s,[,v])=>s+v*v,0));p.vector=raw.map(([j,v])=>[j,v/(norm||1)]);});
  return {products,markets:[...markets.values()].map(m=>{const r=recent.get(JSON.stringify([m.code,m.country]));return {...m,product:idx.get(m.code),orders:m.orders.size,returnRate:m.returnUnits/(m.units+m.returnUnits),recent28:(r?.units||0)/3,recentOrders:r?.orders.size||0,monthly:[...m.monthly.values()].sort((a,b)=>a[0].localeCompare(b[0]))};}),countries:[...new Set(checked.valid.map(r=>r.country))].sort(),period:{start,end},currency,isUpload:true,associations:{},textModel:{vocabulary,idf,stopWords,typeFamilies,minCosine:.15,minScore:.05},coldStart:{selectedNeighbors:8,beatsCountryBaseline:false},boundaries:['Private uploaded data; no independent new-product evaluation has been run.','Analogue scenarios assume comparable exposure and distribution, not proven launch demand.','Only countries observed in this file are eligible. No live model training is performed.']};
}
