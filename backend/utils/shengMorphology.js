// Kenyan street-language ("lugha ya mtaa") morphology layer for the category model.
//
// Beck (2015) analysed naturally occurring Sheng from Eastleigh and found that it
// keeps the Swahili verbal frame (subject, tense, relative and object prefixes)
// while the stem varies: English stems inside Swahili verbs (a-si-contract,
// a-nge-tumi-a condom), the repetitive extension -ang- (wa-na-fany-ang-a),
// class 6 "ma-" plurals for borrowed nouns (ma-chali, ma-STIs) and lengthened
// vowels in clipped words (kolee, stoori). A word such as "anamwitishanga" is
// therefore a familiar stem ("itisha", demand) in a familiar frame.
//
// This module exposes those stems as extra tokens. The original words are kept,
// so nothing the reporter wrote is lost. The same rules (ml/lingo/
// sheng_morphology.json) are applied by the Python training code in
// ml/src/text_prep.py, and a parity test checks that both give identical output.
const fs = require('node:fs');
const path = require('node:path');

const DEFAULT_RULES_PATH = path.resolve(__dirname, '../../ml/lingo/sheng_morphology.json');
const VOWEL = /[aeiou]/;
const LETTERS_ONLY = /^[a-z]+$/;

function longestFirst(list) {
  return [...list].sort((a, b) => b.length - a.length || (a < b ? -1 : a > b ? 1 : 0));
}

function compileRules(rules) {
  return {
    ...rules,
    subject: longestFirst(rules.subject),
    tam: longestFirst(rules.tam),
    relative: longestFirst(rules.relative.filter((item) => item.length >= 2)),
    object: longestFirst(rules.object),
    negativeSubject: longestFirst(rules.negativeSubject),
    negativeTam: longestFirst(rules.negativeTam),
    habitual: longestFirst(rules.habitual)
  };
}

function loadDefaultRules(rulesPath = DEFAULT_RULES_PATH) {
  return JSON.parse(fs.readFileSync(rulesPath, 'utf8'));
}

// Stem forms for one verb remainder: the remainder itself, without the habitual
// -ang- (itishanga -> itisha), without the passive -w- (tozwa -> toza, daiwa ->
// dai) and, for negative present forms, with the final -i restored to -a
// (hatupendi -> penda).
function stemVariants(rest, negativePresent) {
  const out = [rest];
  if (/(anga|angi|ange)$/.test(rest) && rest.length - 4 >= 2) out.push(`${rest.slice(0, -4)}a`);
  if (rest.endsWith('wa') && rest.length >= 5) {
    const base = rest.slice(0, -2);
    out.push(VOWEL.test(base.slice(-1)) ? base : `${base}a`);
  }
  if (negativePresent && rest.endsWith('i') && rest.length >= 4 && !VOWEL.test(rest[rest.length - 2])) {
    out.push(`${rest.slice(0, -1)}a`);
  }
  return out;
}

// After the subject and tense prefixes, an optional relative marker and an
// optional object marker may follow. Both readings (with and without the longest
// matching marker) are kept, because the boundary is ambiguous ("alinidai" holds
// "dai", "alitumia" does not hold "mia"). A remainder left after stripping a
// marker must be at least four letters, or three ending in -i (dai), so that
// short accidental pieces are not added.
function strippedOk(rest, rules) {
  return rest.length >= rules.minStem + 1 || (rest.length === rules.minStem && rest.endsWith('i'));
}

function remainders(afterTam, rules) {
  const found = [];
  const rel = rules.relative.find((item) => afterTam.startsWith(item));
  for (const relPart of rel ? ['', rel] : ['']) {
    const afterRel = afterTam.slice(relPart.length);
    const om = rules.object.find((item) => afterRel.startsWith(item));
    for (const omPart of om ? ['', om] : ['']) {
      const rest = afterRel.slice(omPart.length);
      const stripped = relPart !== '' || omPart !== '';
      const longEnough = stripped ? strippedOk(rest, rules) : rest.length >= rules.minStem;
      if (longEnough && VOWEL.test(rest) && !found.includes(rest)) found.push(rest);
    }
  }
  return found;
}

function parseVerb(token, rules) {
  // Affirmative: subject + tense/aspect + (relative) + (object) + stem.
  for (const sc of rules.subject) {
    if (!token.startsWith(sc)) continue;
    const afterSc = token.slice(sc.length);
    for (const tam of rules.tam) {
      if (!afterSc.startsWith(tam)) continue;
      const rests = remainders(afterSc.slice(tam.length), rules);
      if (rests.length) return rests.flatMap((rest) => stemVariants(rest, false));
    }
  }
  // Negative: negative subject + (tense) + (object) + stem; without a tense
  // marker the negative present ends in -i (sitaki, hatupendi).
  for (const neg of rules.negativeSubject) {
    if (!token.startsWith(neg)) continue;
    const afterNeg = token.slice(neg.length);
    for (const tam of rules.negativeTam) {
      if (!afterNeg.startsWith(tam)) continue;
      const rests = remainders(afterNeg.slice(tam.length), rules);
      if (rests.length) return rests.flatMap((rest) => stemVariants(rest, false));
    }
    if (token.endsWith('i')) {
      const rests = remainders(afterNeg, rules);
      if (rests.length) return rests.flatMap((rest) => stemVariants(rest, true));
    }
  }
  // Habitual hu- (huitisha, hudai).
  for (const hab of rules.habitual) {
    if (!token.startsWith(hab)) continue;
    const rests = remainders(token.slice(hab.length), rules);
    if (rests.length) return rests.flatMap((rest) => stemVariants(rest, false));
  }
  return [];
}

function analyseToken(token, rules) {
  if (token.length < rules.minToken || !LETTERS_ONLY.test(token)) return [];
  const extras = [];
  // Lengthened or emphatic spelling: "pesaaa" -> "pesa", "kolee" -> "kole".
  let word = token.replace(/(.)\1{2,}/g, '$1');
  if (word !== token) extras.push(word);
  if (word.length >= 5 && VOWEL.test(word.slice(-1)) && word.slice(-1) === word.slice(-2, -1)) {
    word = word.slice(0, -1);
    extras.push(word);
  }
  // Class 6 plural of borrowed nouns: "maofficer" -> "officer".
  if (word.startsWith(rules.nounPlural) && word.length - rules.nounPlural.length >= rules.minNounRest) {
    extras.push(word.slice(rules.nounPlural.length));
  }
  extras.push(...parseVerb(word, rules));
  return extras.filter((extra) => extra !== token && extra.length >= 3);
}

// New tokens for text already normalised by kenyaLingo.normalizeText (lower
// case, single spaces), in first-seen order and without repeats.
function morphologyExtras(normalizedText, rawRules) {
  const rules = compileRules(rawRules);
  const seen = new Set();
  const extras = [];
  for (const token of normalizedText.split(' ')) {
    for (const extra of analyseToken(token, rules)) {
      if (!seen.has(extra)) {
        seen.add(extra);
        extras.push(extra);
      }
    }
  }
  return extras;
}

// Concept layer: formal and street wording of the same idea ("unofficial
// payment", "kitu kidogo" -> rushwa bribe) share one token such as
// concept_bribe, so what the model learns from formal reports carries over to
// informal ones. Phrases match whole words only.
const conceptCache = new WeakMap();

function conceptPatterns(concepts) {
  if (conceptCache.has(concepts)) return conceptCache.get(concepts);
  const patterns = Object.keys(concepts.concepts).sort().map((name) => {
    const phrases = [...new Set(concepts.concepts[name])]
      .sort((a, b) => b.length - a.length || (a < b ? -1 : a > b ? 1 : 0));
    const alternation = phrases.map((phrase) => phrase.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
    return { name, pattern: new RegExp(`(?<![\\p{L}\\p{N}])(?:${alternation})(?![\\p{L}\\p{N}])`, 'u') };
  });
  conceptCache.set(concepts, patterns);
  return patterns;
}

function conceptTokens(text, concepts) {
  return conceptPatterns(concepts).filter(({ pattern }) => pattern.test(text)).map(({ name }) => `concept_${name}`);
}

function loadDefaultConcepts(conceptsPath = path.resolve(__dirname, '../../ml/lingo/concepts.json')) {
  return JSON.parse(fs.readFileSync(conceptsPath, 'utf8'));
}

// The text the category model sees: the glossary-expanded report, then the
// street-language stems (glossary-expanded on their own so that a phrase can
// never be matched across the join), then one token per concept expressed.
function prepareModelText(rawText, rawRules, concepts = null) {
  const { normalizeText, expandLingoTerms } = require('./kenyaLingo');
  const normalized = normalizeText(rawText);
  let text = expandLingoTerms(normalized);
  if (rawRules) {
    const extras = morphologyExtras(normalized, rawRules);
    if (extras.length) text = `${text} ${expandLingoTerms(extras.join(' '))}`.trim();
  }
  if (concepts) {
    const tokens = conceptTokens(text, concepts);
    if (tokens.length) text = `${text} ${tokens.join(' ')}`;
  }
  return text;
}

module.exports = {
  analyseToken: (token, rules) => analyseToken(token, compileRules(rules)),
  morphologyExtras,
  conceptTokens,
  prepareModelText,
  loadDefaultRules,
  loadDefaultConcepts
};
