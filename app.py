"""Flask API behind the Chrome extension: fetches YouTube comments, predicts sentiment, draws charts."""
import io
import os
import pickle
import re

# torch must be imported before pandas: on Windows, pandas loads DLLs that make torch's own
# initialisation fail ("DLL initialization routine failed" on c10.dll). Skipped when torch is not
# installed, which is the case for the CI test job and the LightGBM-only setups.
if os.getenv('MODEL_SOURCE', 'transformer') == 'transformer':
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

import matplotlib

matplotlib.use('Agg')  # Non-interactive backend: charts are rendered to PNG bytes only

import matplotlib.dates as mdates
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from matplotlib.figure import Figure
from wordcloud import WordCloud

from src.data.text_cleaning import preprocess_comment
from src.model.predictors import (
    DEFAULT_TRANSFORMER_MODEL,
    BackgroundPredictor,
    ModelLoading,
    SklearnPredictor,
    TransformerPredictor,
)

load_dotenv()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
YOUTUBE_API_URL = 'https://www.googleapis.com/youtube/v3/commentThreads'
VIDEO_ID_PATTERN = re.compile(r'^[\w-]{11}$')
MAX_COMMENTS = 1000

SENTIMENT_LABELS = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
SENTIMENT_COLORS = {1: '#22a06b', 0: '#3b82f6', -1: '#e5484d'}


class YouTubeAPIError(Exception):
    """The YouTube Data API returned an error or could not be reached."""


def load_local_model(model_path, vectorizer_path):
    """Load the pickled model and TF-IDF vectorizer produced by the DVC pipeline."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer


def load_registry_model(model_name, alias):
    """Load a registered model plus the vectorizer logged in the same MLflow run, so the two always match."""
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
    version = MlflowClient().get_model_version_by_alias(model_name, alias)
    model = mlflow.sklearn.load_model(f'models:/{model_name}@{alias}')
    vectorizer_path = mlflow.artifacts.download_artifacts(run_id=version.run_id, artifact_path='tfidf_vectorizer.pkl')
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer


def load_configured_predictor():
    """Pick the predictor from MODEL_SOURCE: 'transformer' (default), 'local' pickles, or MLflow 'registry'."""
    source = os.getenv('MODEL_SOURCE', 'transformer')
    if source == 'transformer':
        return TransformerPredictor(os.getenv('TRANSFORMER_MODEL', DEFAULT_TRANSFORMER_MODEL))
    if source == 'registry':
        return SklearnPredictor(*load_registry_model(
            os.getenv('MODEL_NAME', 'yt_chrome_plugin_model'),
            os.getenv('MODEL_ALIAS', 'staging'),
        ))
    return SklearnPredictor(*load_local_model(
        os.getenv('MODEL_PATH', os.path.join(APP_DIR, 'lgbm_model.pkl')),
        os.getenv('VECTORIZER_PATH', os.path.join(APP_DIR, 'tfidf_vectorizer.pkl')),
    ))


def fetch_youtube_comments(video_id, api_key, max_comments):
    """Fetch up to max_comments top-level comments (most relevant first) via the YouTube Data API v3."""
    comments, page_token = [], None
    while len(comments) < max_comments:
        params = {
            'part': 'snippet',
            'videoId': video_id,
            'maxResults': 100,
            'order': 'relevance',
            'textFormat': 'plainText',
            'key': api_key,
        }
        if page_token:
            params['pageToken'] = page_token

        try:
            response = requests.get(YOUTUBE_API_URL, params=params, timeout=15)
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            raise YouTubeAPIError(f'Could not reach the YouTube API: {e.__class__.__name__}') from e

        if response.status_code != 200:
            error = data.get('error', {})
            reason = (error.get('errors') or [{}])[0].get('reason', '')
            if reason == 'commentsDisabled':
                return []
            raise YouTubeAPIError(error.get('message', f'YouTube API returned HTTP {response.status_code}'))

        for item in data.get('items', []):
            snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'text': snippet['textOriginal'],
                'timestamp': snippet['publishedAt'],
                'authorId': snippet.get('authorChannelId', {}).get('value', 'Unknown'),
            })

        page_token = data.get('nextPageToken')
        if not page_token:
            break
    return comments[:max_comments]


def png_response(fig, **savefig_kwargs):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', **savefig_kwargs)
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')


def trend_frequency(timestamps):
    """Resample daily for recent videos, weekly for a few months of comments, monthly beyond that."""
    span_days = (timestamps.max() - timestamps.min()).days
    if span_days <= 14:
        return 'D', 'Daily', '%b %d'
    if span_days <= 180:
        return 'W', 'Weekly', '%b %d'
    return 'ME', 'Monthly', '%Y-%m'


def create_app(predictor=None, youtube_api_key=None):
    """Build the Flask app. Tests inject a predictor and API key; production loads them from config."""
    if predictor is None:
        # Load in the background so the server accepts connections immediately; predictions wait for it
        wait_seconds = float(os.getenv('MODEL_LOAD_WAIT_SECONDS', '120'))
        predictor = BackgroundPredictor(load_configured_predictor, timeout=wait_seconds)
    # Trimmed: a key pasted into Secret Manager or .env followed by Enter carries a line break YouTube rejects
    api_key = (youtube_api_key if youtube_api_key is not None else os.getenv('YOUTUBE_API_KEY', '')).strip()

    app = Flask(__name__)
    CORS(app)  # The extension calls from a chrome-extension:// origin; no cookies or auth are involved

    @app.errorhandler(ModelLoading)
    def model_still_loading(_error):
        response = jsonify({'error': 'The sentiment model is still loading. Try again in a few seconds.'})
        response.headers['Retry-After'] = '10'
        return response, 503

    def predict_sentiments(texts):
        return predictor.predict(texts)

    def json_list(key):
        """Return (items, None) for a non-empty JSON list under key, or (None, error_response)."""
        items = (request.get_json(silent=True) or {}).get(key)
        if not items or not isinstance(items, list):
            return None, (jsonify({'error': f'No {key} provided'}), 400)
        if len(items) > MAX_COMMENTS:
            return None, (jsonify({'error': f'At most {MAX_COMMENTS} {key} per request'}), 413)
        return items, None

    @app.route('/')
    def home():
        return "Welcome to our flask api"

    @app.route('/health')
    def health():
        return jsonify({
            'status': 'ok',
            'model': predictor.name,
            'model_loaded': getattr(predictor, 'loaded', True),
            'youtube_api_key_configured': bool(api_key),
        })

    @app.route('/comments')
    def comments():
        video_id = request.args.get('video_id', '')
        if not VIDEO_ID_PATTERN.match(video_id):
            return jsonify({'error': 'Invalid YouTube video id'}), 400
        if not api_key:
            return jsonify({'error': 'YOUTUBE_API_KEY is not set on the server'}), 503

        max_comments = min(request.args.get('max_comments', 500, type=int), MAX_COMMENTS)
        try:
            fetched = fetch_youtube_comments(video_id, api_key, max_comments)
        except YouTubeAPIError as e:
            return jsonify({'error': str(e)}), 502
        return jsonify({'video_id': video_id, 'comments': fetched})

    @app.route('/predict', methods=['POST'])
    def predict():
        texts, error = json_list('comments')
        if error:
            return error
        try:
            sentiments = predict_sentiments(texts)
        except ModelLoading:
            raise
        except Exception as e:
            app.logger.exception('Prediction failed')
            return jsonify({"error": f"Prediction failed: {e}"}), 500
        return jsonify([{'comment': text, 'sentiment': sentiment} for text, sentiment in zip(texts, sentiments)])

    @app.route('/predict_with_timestamps', methods=['POST'])
    def predict_with_timestamps():
        items, error = json_list('comments')
        if error:
            return error
        try:
            texts = [item['text'] for item in items]
            timestamps = [item['timestamp'] for item in items]
        except (KeyError, TypeError):
            return jsonify({'error': 'Each comment needs "text" and "timestamp"'}), 400
        try:
            sentiments = predict_sentiments(texts)
        except ModelLoading:
            raise
        except Exception as e:
            app.logger.exception('Prediction failed')
            return jsonify({"error": f"Prediction failed: {e}"}), 500
        return jsonify([
            {'comment': text, 'sentiment': sentiment, 'timestamp': timestamp}
            for text, sentiment, timestamp in zip(texts, sentiments, timestamps)
        ])

    @app.route('/generate_chart', methods=['POST'])
    def generate_chart():
        counts = (request.get_json(silent=True) or {}).get('sentiment_counts')
        if not counts or not isinstance(counts, dict):
            return jsonify({"error": "No sentiment counts provided"}), 400
        try:
            values = {sentiment: int(counts.get(str(sentiment), 0)) for sentiment in SENTIMENT_LABELS}
        except (TypeError, ValueError):
            return jsonify({'error': 'Sentiment counts must be integers'}), 400

        shown = [sentiment for sentiment in SENTIMENT_LABELS if values[sentiment] > 0]
        if not shown:
            return jsonify({'error': 'Sentiment counts sum to zero'}), 400

        fig = Figure(figsize=(6, 6))
        ax = fig.subplots()
        ax.pie(
            [values[s] for s in shown],
            labels=[SENTIMENT_LABELS[s] for s in shown],
            colors=[SENTIMENT_COLORS[s] for s in shown],
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'},
        )
        ax.axis('equal')  # Equal aspect ratio draws the pie as a circle
        return png_response(fig, transparent=True)

    @app.route('/generate_wordcloud', methods=['POST'])
    def generate_wordcloud():
        texts, error = json_list('comments')
        if error:
            return error
        text = ' '.join(preprocess_comment(t) for t in texts).strip()
        if not text:
            return jsonify({'error': 'No words left after preprocessing'}), 400

        wordcloud = WordCloud(
            width=800, height=400, background_color='black', colormap='Blues', collocations=False
        ).generate(text)
        buffer = io.BytesIO()
        wordcloud.to_image().save(buffer, format='PNG')
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png')

    @app.route('/generate_trend_graph', methods=['POST'])
    def generate_trend_graph():
        items, error = json_list('sentiment_data')
        if error:
            return error
        try:
            df = pd.DataFrame(items)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df['sentiment'] = df['sentiment'].astype(int)
        except (KeyError, TypeError, ValueError):
            return jsonify({'error': 'Each item needs a valid "timestamp" and integer "sentiment"'}), 400

        freq, period, date_format = trend_frequency(df['timestamp'])
        counts = (
            df.set_index('timestamp')
            .resample(freq)['sentiment']
            .value_counts()
            .unstack(fill_value=0)
            .reindex(columns=[-1, 0, 1], fill_value=0)
        )
        # Periods without comments divide by zero and are dropped rather than plotted as 0%
        percentages = counts.div(counts.sum(axis=1), axis=0).mul(100).dropna()

        fig = Figure(figsize=(12, 6))
        ax = fig.subplots()
        for sentiment in (-1, 0, 1):
            ax.plot(
                percentages.index,
                percentages[sentiment],
                marker='o',
                linestyle='-',
                label=SENTIMENT_LABELS[sentiment],
                color=SENTIMENT_COLORS[sentiment],
            )
        ax.set(title=f'{period} Sentiment Percentage Over Time', xlabel='Date', ylabel='Percentage of Comments (%)')
        ax.grid(True)
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        return png_response(fig)

    return app


if __name__ == '__main__':
    # Local development only; Docker runs gunicorn. Debug mode stays off: its console allows code execution.
    create_app().run(host=os.getenv('HOST', '127.0.0.1'), port=int(os.getenv('PORT', '8080')))
