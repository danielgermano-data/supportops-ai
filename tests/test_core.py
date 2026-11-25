import json
import math
import pytest
from fastapi.testclient import TestClient

from supportops.agent import validate_answer
from supportops.classifier import load_model, predict
from supportops.paths import DATA, ROOT
from supportops.retrieval import Retriever
from supportops.api import app, hits


@pytest.fixture(autouse=True)
def reset_rate_limit():
    hits.clear()


@pytest.mark.parametrize('text', ['', 'a', 'x'*2001, None])
def test_rejects_invalid_input(text):
    with pytest.raises(ValueError):
        predict(text)


def test_artifact_schema_and_probability_order():
    meta, weights, bias, idf = load_model()
    assert len(meta['classes']) == 77
    assert weights.shape == (77,len(meta['vocabulary']))
    assert (idf > 0).all()
    result = predict('My card was stolen and I need to block it')
    assert result['top'][0]['intent'] == 'lost_or_stolen_card'
    assert result['top'][0]['score'] >= result['top'][1]['score']


def test_unknown_tokens_abstain():
    result = predict('zxqv zxqvzxqv')
    assert result['accepted'] is False
    assert result['reason'] == 'no_known_terms'


def test_lexical_search_has_grounded_sources():
    docs = Retriever().search('cobrança duplicada, duas entradas iguais')
    assert docs[0]['id'] == 'RB-03'
    assert all(d['retrieval'] == 'lexical' for d in docs)
    assert all(d['id'] in Retriever().by_id for d in docs)


def test_citations_cannot_use_unknown_ids():
    sources = [{'id':'RB-03'}]
    assert validate_answer('Verifique o status [RB-03].',sources)['valid']
    assert not validate_answer('Estorno garantido [RB-99].',sources)['valid']
    assert not validate_answer('Uma resposta sem fonte.',sources)['valid']


def test_splits_have_no_row_overlap():
    split = json.loads((ROOT/'reports'/'splits.json').read_text())
    assert not set(split['fit_original_row_ids']) & set(split['validation_original_row_ids'])
    assert len(split['test_original_row_ids']) > 3000


def test_api_validation_and_no_unexpected_tool_parameters():
    with TestClient(app) as client:
        assert client.get('/health').json()['financial_actions'] is False
        assert client.post('/api/classify',json={'text':'x'}).status_code == 422
        assert client.post('/api/classify',json={'text':'blocked card','execute':'transfer'}).status_code == 422
        response = client.post('/api/classify',json={'text':'My card was stolen'})
        assert response.status_code == 200
        assert response.json()['top']


def test_local_api_blocks_cross_origin_and_limits_requests():
    with TestClient(app) as client:
        assert client.post('/api/classify',headers={'Origin':'https://example.com'},json={'text':'My card was stolen'}).status_code == 403
        statuses = [client.post('/api/classify',json={'text':'My card was stolen'}).status_code for _ in range(41)]
        assert statuses[-1] == 429
