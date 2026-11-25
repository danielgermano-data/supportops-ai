"""Portable TF-IDF + logistic regression inference, without loading pickle files."""
import hashlib
import json
import re
from collections import Counter
from functools import lru_cache

import numpy as np

from .paths import WEB

TOKEN = re.compile(r'(?u)\b\w\w+\b')


def validate_text(text: str) -> str:
    if not isinstance(text, str) or not 3 <= len(text.strip()) <= 2000:
        raise ValueError('Use de 3 a 2000 caracteres, sem dados pessoais.')
    return text.strip()


@lru_cache(maxsize=1)
def load_model():
    meta = json.loads((WEB / 'model' / 'model.json').read_text(encoding='utf-8'))
    raw = (WEB / 'model' / 'weights.bin').read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta['weights_sha256']:
        raise ValueError('Model artifact checksum mismatch')
    n, k = len(meta['vocabulary']), len(meta['classes'])
    values = np.frombuffer(raw, dtype='<f4')
    if len(values) != k * n + k + n:
        raise ValueError('Invalid model shape')
    return meta, values[:k*n].reshape(k, n), values[k*n:k*n+k], values[k*n+k:]


def predict(text: str) -> dict:
    text = validate_text(text)
    meta, coef, bias, idf = load_model()
    words = TOKEN.findall(text.lower())
    terms = words + [' '.join(pair) for pair in zip(words, words[1:])]
    counts = Counter(terms)
    pairs = [(meta['vocabulary'][term], count) for term, count in counts.items() if term in meta['vocabulary']]
    if not pairs:
        return {'intent': None, 'accepted': False, 'reason': 'no_known_terms', 'top': [], 'language': 'en'}
    indices = np.array([i for i, _ in pairs])
    tf = (1 + np.log([count for _, count in pairs])) * idf[indices]
    tf /= np.linalg.norm(tf)
    logits = coef[:, indices] @ tf + bias
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    order = probs.argsort()[::-1][:3]
    top = [{'intent': meta['classes'][int(i)], 'score': round(float(probs[i]), 6)} for i in order]
    accepted = bool(probs[order[0]] >= meta['threshold'])
    return {'intent': top[0]['intent'] if accepted else None, 'accepted': accepted, 'reason': 'accepted' if accepted else 'low_score', 'top': top, 'threshold': meta['threshold'], 'language': 'en', 'model_version': meta['version']}
