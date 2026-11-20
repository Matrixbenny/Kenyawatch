const test = require('node:test');
const assert = require('node:assert/strict');
const { analyseToken, morphologyExtras, prepareModelText, loadDefaultRules } = require('../utils/shengMorphology');
const { normaliseForMatching } = require('../utils/kenyaLingo');
const { classifyText } = require('../utils/experimentalCategoryModel');

const rules = loadDefaultRules();

test('glossary terms match whole words only, never inside a longer word', () => {
  assert.equal(normaliseForMatching('mkuchoma moto'), 'mkuchoma moto');
  assert.equal(normaliseForMatching('hongoyake'), 'hongoyake');
  assert.match(normaliseForMatching('Walitaka kitu kidogo'), /rushwa bribe/);
  assert.match(normaliseForMatching('hongo'), /rushwa bribe/);
});

test('exposes the stem inside Swahili verb frames, as Beck (2015) describes for Sheng', () => {
  const stems = (word) => analyseToken(word, rules);
  assert.ok(stems('anamwitishanga').includes('itisha'));   // a-na-mw-itish-ang-a
  assert.ok(stems('alinidai').includes('dai'));            // a-li-ni-dai
  assert.ok(stems('asicontract').includes('contract'));    // a-si-contract (English stem)
  assert.ok(stems('walinithreaten').includes('threaten'));
  assert.ok(stems('wanafanyanga').includes('fanya'));      // repetitive -ang-
  assert.ok(stems('hatupendi').includes('penda'));         // negative present -i
  assert.ok(stems('anadaiwa').includes('dai'));            // passive -w-
  assert.ok(stems('maofficer').includes('officer'));       // class 6 plural of a borrowed noun
  assert.ok(stems('pesaaa').includes('pesa'));             // lengthened spelling
});

test('does not invent stems from ordinary words or short accidental pieces', () => {
  assert.deepEqual(analyseToken('mkuchoma', rules), []);
  assert.deepEqual(analyseToken('siasa', rules), []);
  assert.ok(!analyseToken('angetumia', rules).includes('mia'));
  assert.deepEqual(analyseToken('kitu', rules), []);
  assert.deepEqual(analyseToken('5000', rules), []);
});

test('keeps the original words and appends glossary-expanded stems after them', () => {
  const prepared = prepareModelText('Walinidai pesa', rules);
  assert.ok(prepared.startsWith('walinidai pesa '));
  assert.match(prepared, /\bdai\b/);
  assert.match(prepared, /demanded money/); // the stem "nidai" is in the glossary
  assert.equal(prepareModelText('Walinidai pesa', null), normaliseForMatching('Walinidai pesa'));
  assert.deepEqual(morphologyExtras('', rules), []);
});

test('the model scores report text through the preprocessing stored in its artefact', () => {
  const model = {
    schemaVersion: 1,
    model: 'LinearSVC',
    modelVersion: 'test-version',
    trainingDataStatus: 'synthetic_only',
    realWorldPerformanceClaim: false,
    categories: ['Bribery', 'Other'],
    preprocessing: { normaliser: 'kenya-lingo', morphology: rules },
    features: { word: { vocabulary: { itisha: 0 }, idf: [1], ngramRange: [1, 1] } },
    weights: [[3], [0]],
    intercepts: [0, 0.5]
  };
  // "itisha" (demand) only appears once the verb frame is read.
  assert.equal(classifyText({ title: 'Kanjo', description: 'anamwitishanga pesa' }, model).suggestion, 'Bribery');
  const withoutLayer = { ...model, preprocessing: { normaliser: 'kenya-lingo', morphology: null } };
  assert.equal(classifyText({ title: 'Kanjo', description: 'anamwitishanga pesa' }, withoutLayer).suggestion, 'Other');
});
