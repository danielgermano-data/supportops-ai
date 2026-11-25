"""Select hyperparameter and rejection threshold on validation; test once."""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
START = time.perf_counter()


def read(name):
    with (ROOT / 'data' / 'raw' / name).open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    return [r['text'] for r in rows], [r['category'] for r in rows]


def normalized(text):
    return ' '.join(text.lower().split())


texts, labels = read('train.csv')
test_text, test_labels = read('test.csv')
texts, labels = np.array(texts), np.array(labels)
norm = np.array([normalized(x) for x in texts])
# Keep the first original annotation when an identical query is repeated.
_, unique = np.unique(norm, return_index=True)
unique = np.sort(unique)
dedup_text, dedup_labels = texts[unique], labels[unique]
dedup_groups = norm[unique]
split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fit_idx, val_idx = next(split.split(dedup_text, dedup_labels, dedup_groups))
fit_x, fit_y = dedup_text[fit_idx], dedup_labels[fit_idx]
val_x, val_y = dedup_text[val_idx], dedup_labels[val_idx]
train_vocab = set(dedup_groups)
overlap_indices = [i for i, x in enumerate(test_text) if normalized(x) in train_vocab]
# Remove exact train/test overlap from the held-out evaluation, record exclusions.
keep = [i for i in range(len(test_text)) if i not in set(overlap_indices)]
test_x, test_y = np.array(test_text)[keep], np.array(test_labels)[keep]
candidates, models = [], []
for c in [0.5, 2.0, 8.0]:
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=16000, min_df=2, sublinear_tf=True)),
        ('lr', LogisticRegression(C=c, max_iter=500, solver='lbfgs', random_state=42)),
    ])
    pipe.fit(fit_x, fit_y)
    predictions = pipe.predict(val_x)
    candidates.append({'C': c, 'validation_macro_f1': float(f1_score(val_y, predictions, average='macro')), 'validation_accuracy': float(accuracy_score(val_y, predictions))})
    models.append(pipe)
best_idx = max(range(len(candidates)), key=lambda i: candidates[i]['validation_macro_f1'])
model = models[best_idx]
# Do not refit after threshold selection: same trained model is exported and tested.
val_probs = model.predict_proba(val_x)
val_conf = val_probs.max(axis=1)
val_correct = model.classes_[val_probs.argmax(axis=1)] == val_y
thresholds = []
for threshold in np.arange(0.20, 0.91, 0.01):
    mask = val_conf >= threshold
    if mask.sum():
        thresholds.append({'threshold': round(float(threshold), 2), 'accuracy': float(val_correct[mask].mean()), 'coverage': float(mask.mean()), 'accepted': int(mask.sum())})
eligible = [r for r in thresholds if r['accuracy'] >= .90 and r['accepted'] >= 50]
selected_threshold = max(eligible, key=lambda r: r['coverage']) if eligible else max(thresholds, key=lambda r: r['accuracy'])
probs = model.predict_proba(test_x)
prediction = model.classes_[probs.argmax(axis=1)]
confidence = probs.max(axis=1)
accepted = confidence >= selected_threshold['threshold']
classes = model.classes_.tolist()
baseline_label = max(set(fit_y), key=lambda label: int((fit_y == label).sum()))
baseline = np.repeat(baseline_label, len(test_y))
report = {
    'dataset': 'BANKING77', 'seed': 42, 'language': 'English', 'classes': len(classes),
    'split': {'original_train': len(texts), 'duplicates_removed_from_train': len(texts)-len(dedup_text), 'fit': len(fit_y), 'validation': len(val_y), 'original_test': len(test_text), 'test': len(test_y), 'exact_train_test_overlap_removed': len(overlap_indices)},
    'method': 'TF-IDF word unigrams/bigrams + multinomial logistic regression; C selected on validation macro-F1; no fitting or tuning on test',
    'hyperparameter_search': candidates, 'selected_C': candidates[best_idx]['C'],
    'abstention': {'selected_on': 'validation', 'target_accuracy': .90, 'validation': selected_threshold, 'test_coverage': float(accepted.mean()), 'test_accepted_accuracy': float((prediction[accepted] == test_y[accepted]).mean()) if accepted.any() else None},
    'test': {'accuracy': float(accuracy_score(test_y, prediction)), 'macro_f1': float(f1_score(test_y, prediction, average='macro')), 'top3_accuracy': float(np.mean([y in model.classes_[row.argsort()[-3:]] for y, row in zip(test_y, probs)]))},
    'baseline_most_frequent': {'accuracy': float(accuracy_score(test_y, baseline)), 'macro_f1': float(f1_score(test_y, baseline, average='macro')), 'label': baseline_label},
    'per_class': classification_report(test_y, prediction, output_dict=True, zero_division=0),
    'confusion_matrix': confusion_matrix(test_y, prediction, labels=classes).tolist(), 'class_order': classes,
    'limitations': ['English only classifier', 'Public benchmark; not a real customer deployment', 'No calibrated probability guarantee', 'Exact deduplication does not remove all near-duplicate semantic overlap', 'Out-of-domain queries require abstention and human review', 'No LLM fine-tuning performed'],
    'environment': {'sklearn': sklearn.__version__, 'numpy': np.__version__},
}
out = ROOT / 'web' / 'model'
out.mkdir(parents=True, exist_ok=True)
vectorizer, lr = model['tfidf'], model['lr']
arrays = [lr.coef_.astype('<f4').ravel(), lr.intercept_.astype('<f4'), vectorizer.idf_.astype('<f4')]
binary = np.concatenate(arrays).astype('<f4').tobytes()
(out / 'weights.bin').write_bytes(binary)
meta = {'version': 'banking77-tfidf-lr-v1', 'classes': classes, 'vocabulary': {term: int(index) for term, index in vectorizer.vocabulary_.items()}, 'threshold': selected_threshold['threshold'], 'dtype': 'float32-little-endian', 'weights_sha256': hashlib.sha256(binary).hexdigest(), 'token_pattern': r'(?u)\b\w\w+\b', 'ngram_range': [1,2], 'sublinear_tf': True, 'language': 'en'}
(out / 'model.json').write_text(json.dumps(meta, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
report['training_seconds'] = round(time.perf_counter() - START, 3)
report['artifact_weights_sha256'] = meta['weights_sha256']
report['corpus_sha256'] = hashlib.sha256('\n'.join(texts.tolist()).encode()).hexdigest()
(ROOT / 'reports').mkdir(exist_ok=True)
(ROOT / 'reports' / 'training.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
(ROOT / 'web' / 'training.json').write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
split_manifest = {'fit_original_row_ids': unique[fit_idx].tolist(), 'validation_original_row_ids': unique[val_idx].tolist(), 'test_original_row_ids': keep, 'test_excluded_row_ids': overlap_indices}
(ROOT / 'reports' / 'splits.json').write_text(json.dumps(split_manifest), encoding='utf-8')
with (ROOT / 'reports' / 'test_predictions.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['original_test_row', 'expected', 'predicted', 'score', 'accepted'])
    writer.writerows((keep[i], test_y[i], prediction[i], round(float(confidence[i]),6), bool(accepted[i])) for i in range(len(test_y)))
# Portable float32 export parity, including probability scores.
import sys
sys.path.insert(0, str(ROOT / 'src'))
from supportops.classifier import predict
max_diff = 0.
for i in range(min(100, len(test_x))):
    got = predict(str(test_x[i]))
    assert got['top'][0]['intent'] == prediction[i]
    max_diff = max(max_diff, abs(got['top'][0]['score'] - confidence[i]))
report['portable_export_parity'] = {'samples': min(100, len(test_x)), 'top1_match': True, 'max_score_error': float(max_diff)}
(ROOT / 'reports' / 'training.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
(ROOT / 'web' / 'training.json').write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
print(json.dumps({k: report[k] for k in ['split','selected_C','test','baseline_most_frequent','abstention','training_seconds','portable_export_parity']}, indent=2))
