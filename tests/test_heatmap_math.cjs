const assert = require('node:assert/strict');
const {selectSamples,occupancyGrid}=require('../sn_gamestate/custom_video/heatmap.js');
const sum=grid=>grid.reduce((a,b)=>a+b,0);
const samples=[[0,0,0,0,'1',false],[52.5,34,.04,1,'1',false],[-52.5,-34,.08,2,'2',true]];
for(const sigma of [0,.5,2,5]){
 const grid=occupancyGrid(samples,25,sigma);
 assert.equal(grid.length,105*68);
 assert.ok(Math.abs(sum(grid)-.12)<1e-10,'Gaussian blur must preserve total person-seconds, including borders');
 assert.ok(grid.every(v=>Number.isFinite(v)&&v>=0));
}
assert.equal(occupancyGrid([[52.5,34,0,0,'1',false]],25,0)[105*68-1],.04);
assert.equal(sum(occupancyGrid([[53,0,0,0,'1',false]],25,0)),0);
assert.equal(selectSamples({samples},{start:0,end:.04,track:'all',excludeTruncated:false}).length,1);
assert.equal(selectSamples({samples},{start:0,end:1,track:'2',excludeTruncated:true}).length,0);
assert.equal(selectSamples({samples},{start:0,end:1,track:'1',excludeTruncated:false}).length,2);
console.log('Heatmap math: conservation, boundaries, time range, ID and truncation filters passed');
