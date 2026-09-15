import threading

import pytest

from app import create_app
from src.model.predictors import BackgroundPredictor

PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


class KeywordPredictor:
    """Stands in for the real model: 'good' -> positive, 'bad' -> negative, anything else neutral."""

    name = 'stub'

    def predict(self, texts):
        return [1 if 'good' in t.lower() else -1 if 'bad' in t.lower() else 0 for t in texts]


def make_client(api_key='test-key'):
    return create_app(predictor=KeywordPredictor(), youtube_api_key=api_key).test_client()


@pytest.fixture
def client():
    return make_client()


def youtube_item(text, published_at='2026-01-01T00:00:00Z', author='abc'):
    snippet = {'textOriginal': text, 'publishedAt': published_at, 'authorChannelId': {'value': author}}
    return {'snippet': {'topLevelComment': {'snippet': snippet}}}


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


def test_home(client):
    assert client.get('/').status_code == 200


def test_predict_returns_a_sentiment_per_comment(client):
    response = client.post('/predict', json={'comments': ['This video is good', 'Very bad explanation', 'Uploaded Tuesday']})

    assert response.status_code == 200
    assert [item['sentiment'] for item in response.get_json()] == [1, -1, 0]


def test_predict_returns_original_text_not_preprocessed(client):
    response = client.post('/predict', json={'comments': ['This video is GOOD!']})

    assert response.get_json()[0]['comment'] == 'This video is GOOD!'


@pytest.mark.parametrize('payload', [{}, {'comments': []}, {'comments': 'not a list'}])
def test_predict_rejects_missing_comments(client, payload):
    assert client.post('/predict', json=payload).status_code == 400


def test_predict_rejects_oversized_batch(client):
    assert client.post('/predict', json={'comments': ['x'] * 1001}).status_code == 413


def test_predict_with_timestamps_keeps_timestamps(client):
    comments = [{'text': 'good stuff', 'timestamp': '2026-01-01T00:00:00Z'}]

    response = client.post('/predict_with_timestamps', json={'comments': comments})

    assert response.get_json() == [{'comment': 'good stuff', 'sentiment': 1, 'timestamp': '2026-01-01T00:00:00Z'}]


def test_predict_with_timestamps_rejects_items_without_text(client):
    response = client.post('/predict_with_timestamps', json={'comments': [{'timestamp': '2026-01-01'}]})

    assert response.status_code == 400


def test_generate_chart_returns_png(client):
    response = client.post('/generate_chart', json={'sentiment_counts': {'1': 5, '0': 3, '-1': 2}})

    assert response.status_code == 200
    assert response.mimetype == 'image/png'
    assert response.data.startswith(PNG_SIGNATURE)


def test_generate_chart_rejects_all_zero_counts(client):
    response = client.post('/generate_chart', json={'sentiment_counts': {'1': 0, '0': 0, '-1': 0}})

    assert response.status_code == 400


def test_generate_wordcloud_returns_png(client):
    response = client.post('/generate_wordcloud', json={'comments': ['great explanation', 'loved the examples']})

    assert response.status_code == 200
    assert response.data.startswith(PNG_SIGNATURE)


def test_generate_wordcloud_rejects_text_with_no_words(client):
    response = client.post('/generate_wordcloud', json={'comments': ['❤️❤️', 'the and of']})

    assert response.status_code == 400


@pytest.mark.parametrize('dates', [
    ['2026-01-01T10:00:00Z', '2026-01-02T10:00:00Z', '2026-01-05T10:00:00Z'],  # daily
    ['2026-01-01T10:00:00Z', '2026-02-10T10:00:00Z', '2026-03-15T10:00:00Z'],  # weekly
    ['2025-01-01T10:00:00Z', '2025-08-10T10:00:00Z', '2026-03-15T10:00:00Z'],  # monthly
])
def test_generate_trend_graph_returns_png(client, dates):
    data = [{'timestamp': d, 'sentiment': s} for d, s in zip(dates, [1, -1, 0])]

    response = client.post('/generate_trend_graph', json={'sentiment_data': data})

    assert response.status_code == 200
    assert response.data.startswith(PNG_SIGNATURE)


def test_generate_trend_graph_rejects_bad_timestamps(client):
    response = client.post('/generate_trend_graph', json={'sentiment_data': [{'timestamp': 'yesterday', 'sentiment': 1}]})

    assert response.status_code == 400


def test_comments_rejects_invalid_video_id(client):
    assert client.get('/comments?video_id=not-an-id').status_code == 400


def test_comments_requires_server_api_key():
    response = make_client(api_key='').get('/comments?video_id=gwNPV882tkc')

    assert response.status_code == 503


def test_comments_fetches_and_follows_pages(client, monkeypatch):
    pages = [
        FakeResponse({'items': [youtube_item('first')], 'nextPageToken': 'page-2'}),
        FakeResponse({'items': [youtube_item('second', author='xyz')]}),
    ]
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params)
        return pages[len(calls) - 1]

    monkeypatch.setattr('app.requests.get', fake_get)

    response = client.get('/comments?video_id=gwNPV882tkc')

    assert response.status_code == 200
    assert [c['text'] for c in response.get_json()['comments']] == ['first', 'second']
    assert calls[0]['key'] == 'test-key'
    assert calls[1]['pageToken'] == 'page-2'


def test_comments_disabled_returns_empty_list(client, monkeypatch):
    error = {'error': {'message': 'disabled', 'errors': [{'reason': 'commentsDisabled'}]}}
    monkeypatch.setattr('app.requests.get', lambda url, params, timeout: FakeResponse(error, status_code=403))

    response = client.get('/comments?video_id=gwNPV882tkc')

    assert response.status_code == 200
    assert response.get_json()['comments'] == []


def test_comments_surfaces_youtube_errors_as_bad_gateway(client, monkeypatch):
    error = {'error': {'message': 'API key not valid.', 'errors': [{'reason': 'badRequest'}]}}
    monkeypatch.setattr('app.requests.get', lambda url, params, timeout: FakeResponse(error, status_code=400))

    response = client.get('/comments?video_id=gwNPV882tkc')

    assert response.status_code == 502
    assert response.get_json()['error'] == 'API key not valid.'


def test_predictions_return_503_with_retry_after_while_the_model_loads():
    release = threading.Event()

    def slow_loader():
        release.wait(5)
        return KeywordPredictor()

    client = create_app(predictor=BackgroundPredictor(slow_loader, timeout=0.05), youtube_api_key='k').test_client()

    response = client.post('/predict', json={'comments': ['good']})
    health = client.get('/health').get_json()
    release.set()

    assert response.status_code == 503
    assert response.headers['Retry-After'] == '10'
    assert health['model_loaded'] is False


def test_health_reports_a_ready_model(client):
    assert client.get('/health').get_json()['model_loaded'] is True


def test_api_key_is_trimmed_of_spaces_and_line_breaks(monkeypatch):
    monkeypatch.setenv('YOUTUBE_API_KEY', '  test-key' + chr(10))
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params)
        return FakeResponse({'items': []})

    monkeypatch.setattr('app.requests.get', fake_get)

    create_app(predictor=KeywordPredictor()).test_client().get('/comments?video_id=gwNPV882tkc')

    assert calls[0]['key'] == 'test-key'
