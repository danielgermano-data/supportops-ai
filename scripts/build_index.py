import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supportops.llm import Ollama
from supportops.paths import DATA, ARTIFACTS

docs = json.loads((DATA / 'knowledge.json').read_text(encoding='utf-8'))
texts = ['title: ' + d['title'] + ' | text: ' + d['text'] for d in docs]
model = Ollama()
vectors = np.asarray(model.embed(texts), dtype=np.float32)
if vectors.shape[0] != len(docs) or not np.isfinite(vectors).all():
    raise ValueError('Embedding generation failed validation')
ARTIFACTS.mkdir(exist_ok=True)
(ROOT / 'reports').mkdir(exist_ok=True)
np.save(ARTIFACTS / 'knowledge_embeddings.npy', vectors, allow_pickle=False)
manifest = {'embedding_model': model.embed_model, 'documents': len(docs), 'dimensions': vectors.shape[1], 'knowledge_sha256': hashlib.sha256((DATA / 'knowledge.json').read_bytes()).hexdigest(), 'source': 'actual local Ollama embeddings', 'chunking': 'One short self-contained bilingual runbook per chunk'}
(ARTIFACTS / 'index_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
(ROOT / 'reports' / 'index.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest))
