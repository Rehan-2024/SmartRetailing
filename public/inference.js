export function predict(model,input){
  const values=[];
  model.numeric.forEach((c,i)=>{const v=Number(input[c]);if(!Number.isFinite(v))throw Error(`Invalid ${c}`);values.push((v-model.mean[i])/model.scale[i]);});
  model.categorical.forEach((c,i)=>model.categories[i].forEach(v=>values.push(String(input[c])===String(v)?1:0)));
  const contributions=values.map((v,i)=>v*model.coefficients[i]);
  const logit=model.intercept+contributions.reduce((a,b)=>a+b,0);
  return {probability:1/(1+Math.exp(-logit)),contributions};
}
