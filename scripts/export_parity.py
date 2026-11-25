import csv
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supportops.classifier import predict

with (ROOT/'data'/'raw'/'test.csv').open(encoding='utf-8',newline='') as f:
    texts=[row['text'] for row in csv.DictReader(f)]
selected=texts[:30]+['zxqv zxqvzxqv','My card was charged twice for the same purchase.','Perdi meu cartão e preciso de ajuda.']
(ROOT/'tests'/'parity_cases.json').write_text(json.dumps([{'text':text,'expected':predict(text)} for text in selected],ensure_ascii=False,indent=2),encoding='utf-8')
print('Exported',len(selected),'cross-language runtime parity cases; Portuguese is a parity check, not a validation claim.')
