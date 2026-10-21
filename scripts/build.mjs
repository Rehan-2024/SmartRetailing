import {mkdir,cp,readFile} from 'node:fs/promises';
const data=JSON.parse(await readFile('public/data/report.json','utf8'));
if(!data.retail || !data.purchase || !data.forecast) throw Error('Run python pipeline/train.py --data-root .. before building');
await mkdir('dist',{recursive:true});
await cp('public','dist',{recursive:true});
console.log('Built dashboard with verified report '+data.runId);
