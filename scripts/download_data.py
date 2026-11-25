"""Download the original BANKING77 splits with checksums; no customer files."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'train.csv': 'b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b',
    'test.csv': 'd12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d',
}
base = 'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/'
target = ROOT / 'data' / 'raw'
target.mkdir(parents=True, exist_ok=True)
manifest = {'dataset': 'BANKING77', 'owner': 'PolyAI', 'license': 'CC-BY-4.0', 'source': 'https://huggingface.co/datasets/PolyAI/banking77', 'files': {}}
for name, expected in FILES.items():
    body = urllib.request.urlopen(base + name, timeout=30).read()
    digest = hashlib.sha256(body).hexdigest()
    if digest != expected:
        raise ValueError(f'Upstream data changed: {name}; review rather than silently retrain.')
    (target / name).write_bytes(body)
    manifest['files'][name] = {'url': base + name, 'sha256': digest, 'bytes': len(body)}
(ROOT / 'data' / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps({'downloaded': list(FILES), 'checksum_validation': True}))
