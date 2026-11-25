// Same float32 model artifact as Python. No precomputed classification replies.
export function tokenize(text) {
  const words = (text.toLowerCase().match(/[\p{L}\p{N}_]+/gu) || []).filter(word => word.length >= 2);
  return [...words, ...words.slice(0, -1).map((word, i) => `${word} ${words[i + 1]}`)];
}

export function predict(text, meta, values) {
  const clean = text.trim();
  if (clean.length < 3 || clean.length > 2000) throw new Error('Use de 3 a 2000 caracteres.');
  const n = Object.keys(meta.vocabulary).length;
  const k = meta.classes.length;
  if (values.length !== k * n + k + n) throw new Error('Artefato de modelo inválido.');
  const counts = new Map();
  for (const term of tokenize(clean)) counts.set(term, (counts.get(term) || 0) + 1);
  const known = [...counts.entries()].filter(([term]) => Object.hasOwn(meta.vocabulary, term));
  if (!known.length) return {accepted:false, intent:null, reason:'no_known_terms', top:[]};
  const features = known.map(([term, count]) => {
    const index = meta.vocabulary[term];
    return [index, (1 + Math.log(count)) * values[k * n + k + index]];
  });
  const norm = Math.sqrt(features.reduce((sum, [, value]) => sum + value * value, 0));
  const logits = meta.classes.map((_, c) => {
    let sum = values[k * n + c];
    for (const [i, value] of features) sum += values[c * n + i] * value / norm;
    return sum;
  });
  const max = Math.max(...logits);
  const exponentials = logits.map(x => Math.exp(x - max));
  const denominator = exponentials.reduce((a, b) => a + b, 0);
  const top = meta.classes.map((intent, i) => ({intent, score:exponentials[i] / denominator})).sort((a, b) => b.score - a.score).slice(0, 3);
  const accepted = top[0].score >= meta.threshold;
  return {accepted, intent:accepted ? top[0].intent : null, reason:accepted ? 'accepted' : 'low_score', top, threshold:meta.threshold};
}

function searchTokens(text) {
  return tokenize(text.normalize('NFD').replace(/\p{M}/gu, '')).filter(term => !term.includes(' '));
}

// BM25: a genuine lexical retrieval baseline, explicitly distinct from local embeddings.
export function search(query, docs, topK=3) {
  const corpus = docs.map(doc => searchTokens(doc.title + ' ' + doc.text));
  const avgLength = corpus.reduce((sum, terms) => sum + terms.length, 0) / corpus.length;
  const queryTerms = [...new Set(searchTokens(query))];
  const df = new Map(queryTerms.map(term => [term, corpus.filter(terms => terms.includes(term)).length]));
  return docs.map((doc, i) => {
    let score = 0;
    for (const term of queryTerms) {
      const tf = corpus[i].filter(word => word === term).length;
      const idf = Math.log(1 + (docs.length - df.get(term) + .5) / (df.get(term) + .5));
      score += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * corpus[i].length / avgLength));
    }
    return {...doc, score, retrieval:'BM25 lexical'};
  }).filter(doc => doc.score > 0).sort((a, b) => b.score - a.score).slice(0, topK);
}

export function citationCheck(answer, sources) {
  const cited = [...answer.matchAll(/\[(RB-\d{2})\]/g)].map(m => m[1]);
  return cited.length > 0 && cited.every(id => sources.some(doc => doc.id === id));
}
