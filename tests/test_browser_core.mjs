import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {predict, search, citationCheck} from '../web/core.js';

const meta=JSON.parse(await readFile(new URL('../web/model/model.json',import.meta.url),'utf8'));
const bytes=await readFile(new URL('../web/model/weights.bin',import.meta.url));
const weights=new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);
const cases=JSON.parse(await readFile(new URL('./parity_cases.json',import.meta.url),'utf8'));
const docs=JSON.parse(await readFile(new URL('../data/knowledge.json',import.meta.url),'utf8'));

test('Browser and Python inference agree on labels, rejection and top-three scores',()=>{
  for(const item of cases){const actual=predict(item.text,meta,weights);assert.equal(actual.accepted,item.expected.accepted,item.text);assert.equal(actual.intent,item.expected.intent,item.text);assert.deepEqual(actual.top.map(x=>x.intent),item.expected.top.map(x=>x.intent),item.text);for(let i=0;i<actual.top.length;i++)assert.ok(Math.abs(actual.top[i].score-item.expected.top[i].score)<1e-5,item.text);}
});
test('BM25 retrieves the runbook for duplicate transactions in both languages',()=>{
  assert.equal(search('cobrança duplicada duas entradas iguais',docs)[0].id,'RB-03');
  assert.equal(search('duplicate transaction charged twice',docs)[0].id,'RB-03');
});
test('Unknown citations and absent citations are rejected',()=>{
  assert.ok(citationCheck('Use o canal oficial [RB-01].',[docs[0]]));
  assert.equal(citationCheck('Use o canal oficial [RB-99].',[docs[0]]),false);
  assert.equal(citationCheck('Sem nenhuma fonte.',[docs[0]]),false);
});
