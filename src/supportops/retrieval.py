import hashlib
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .classifier import validate_text
from .llm import Ollama
from .paths import DATA, ARTIFACTS


class Retriever:
    def __init__(self, semantic=False):
        self.docs = json.loads((DATA / 'knowledge.json').read_text(encoding='utf-8'))
        self.by_id = {d['id']: d for d in self.docs}
        self.semantic = semantic
        self.texts = [d['title'] + '\n' + d['text'] for d in self.docs]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents='unicode', sublinear_tf=True)
        self.lexical = self.vectorizer.fit_transform(self.texts)
        if semantic:
            path = ARTIFACTS / 'knowledge_embeddings.npy'
            manifest_path = ARTIFACTS / 'index_manifest.json'
            if not path.exists() or not manifest_path.exists():
                raise ValueError('Execute python scripts/build_index.py antes da recuperação semântica.')
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            digest = hashlib.sha256((DATA / 'knowledge.json').read_bytes()).hexdigest()
            if manifest['knowledge_sha256'] != digest or manifest['embedding_model'] != Ollama().embed_model:
                raise ValueError('Índice desatualizado; reconstrua com scripts/build_index.py.')
            self.vectors = np.load(path, allow_pickle=False)
            self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)
            if self.vectors.shape[0] != len(self.docs):
                raise ValueError('Invalid index shape')

    def search(self, query, top_k=3):
        query = validate_text(query)
        if not isinstance(top_k, int) or not 1 <= top_k <= 5:
            raise ValueError('top_k deve estar entre 1 e 5')
        if self.semantic:
            q = np.array(Ollama().embed(['task: search result | query: ' + query])[0])
            q /= np.linalg.norm(q)
            scores = self.vectors @ q
        else:
            scores = (self.lexical @ self.vectorizer.transform([query]).T).toarray().ravel()
        # Semantic matches alone do not prove the question is answerable.
        order = scores.argsort()[::-1][:top_k]
        return [{'id': self.docs[i]['id'], 'title': self.docs[i]['title'], 'text': self.docs[i]['text'], 'score': round(float(scores[i]), 6), 'retrieval': 'semantic' if self.semantic else 'lexical'} for i in order if scores[i] > 0]
