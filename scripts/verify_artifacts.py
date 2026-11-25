import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
meta=json.loads((root/'web/model/model.json').read_text(encoding='utf-8'))
assert hashlib.sha256((root/'web/model/weights.bin').read_bytes()).hexdigest()==meta['weights_sha256']
assert (root/'web/knowledge.json').read_bytes()==(root/'data/knowledge.json').read_bytes()
report=json.loads((root/'reports/training.json').read_text(encoding='utf-8'))
assert report==json.loads((root/'web/training.json').read_text(encoding='utf-8'))
assert report['portable_export_parity']['top1_match']
assert report['split']['fit']+report['split']['validation']==9999
assert report['test']['macro_f1']>.5
live=json.loads((root/'reports/live_evaluation.json').read_text(encoding='utf-8'))
assert live==json.loads((root/'web/live_evaluation.json').read_text(encoding='utf-8'))
assert live['rag'] and live['agents']
assert any(a['trace'] for a in live['agents'])
assert 'actual local inference' in live['source']
print('Artifact integrity, report parity and real-inference records verified.')
