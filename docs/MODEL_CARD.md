# Model card

## Intended use
English banking-support intent classification for learning and portfolio evaluation. Human-reviewed suggestions only. No account access or automatic customer decision.

## Training and evaluation
`scripts/train.py` is the executable protocol. Original public dataset: BANKING77 by PolyAI, CC BY 4.0. After four exact repeats were removed from the original training set, 8,000 rows fit the model and 1,999 rows selected C and the rejection threshold. Seven exact train/test overlaps were excluded from the original test, leaving 3,073 rows. Near-duplicates are not exhaustively eliminated.

The model uses word unigrams/bigrams, sublinear TF, smooth IDF, 16,000 maximum features and logistic regression. C candidates are 0.5, 2 and 8; validation macro-F1 selects 8. No refit after threshold selection. The shipped weights correspond to this same fit.

## Observed metrics
See machine-readable `reports/training.json`, per-class report, confusion matrix and predictions. Reported test accuracy is about 87.96 percent and macro-F1 about 0.8800. Majority baseline accuracy is about 1.30 percent. At validation-selected threshold 0.20, test accepted accuracy is about 91.24 percent with about 94.40 percent coverage. These are benchmark measurements, not approval rates or production service guarantees.

## Limitations
No Portuguese validation, probability calibration, fairness study or production drift evaluation. Low scores can trigger rejection, but high scores do not prove the query is in scope. Translation by an LLM can change intent. No LLM was trained or fine-tuned. Ollama and browser generation use independent pretrained models. Browser BM25 and local embedding retrieval are distinct implementations and must be evaluated separately.

## Reproducibility
Dataset file checksums, row indices and feature/weight metadata are committed. Binary float32 weights are verified by SHA-256 and checked against Python predictions. The exact runtime environment is in `requirements.lock`; different numerical libraries or hardware may yield small differences on retraining. Running inference uses the committed weights rather than retraining.
