"""Sentiment predictors behind the API.

Two interchangeable implementations, both mapping a comment to -1 negative, 0 neutral, 1 positive:
the course's TF-IDF + LightGBM pair, and a pretrained multilingual social-media transformer.
"""
import re
import threading

from src.data.text_cleaning import preprocess_comment

DEFAULT_TRANSFORMER_MODEL = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
SENTIMENT_BY_LABEL = {'negative': -1, 'neutral': 0, 'positive': 1}
MENTION = re.compile(r'^@\w')


class SklearnPredictor:
    """TF-IDF vectorizer + LightGBM classifier trained by the DVC pipeline."""

    def __init__(self, model, vectorizer, name='lightgbm-tfidf'):
        self.model = model
        self.vectorizer = vectorizer
        self.name = name

    def predict(self, texts):
        features = self.vectorizer.transform([preprocess_comment(text) for text in texts])
        return [int(prediction) for prediction in self.model.predict(features)]


class TransformerPredictor:
    """Pretrained multilingual sentiment model.

    Trained on social media text, so emoji, slang, hyperbole and code-mixed languages are signal
    rather than noise: the text is passed through almost unchanged instead of being stripped by
    `preprocess_comment`, which would delete all of it.
    """

    def __init__(self, model_name=DEFAULT_TRANSFORMER_MODEL, batch_size=32, max_length=128):
        # Imported here so the API starts without torch when another model source is configured.
        # app.py pre-imports torch before pandas; see the note there.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.sentiment_by_id = self._label_map(self.model.config.id2label)
        # One inference at a time: on a 2-vCPU instance parallel runs add no speed, only memory,
        # and concurrent popup and badge requests pushed Cloud Run past its 2 GiB limit.
        self._lock = threading.Lock()

    @staticmethod
    def _label_map(id2label):
        """Map the model's class ids to -1/0/1, falling back to class order for unnamed labels."""
        mapped = {i: SENTIMENT_BY_LABEL.get(str(label).lower()) for i, label in id2label.items()}
        if None in mapped.values():
            return {i: sentiment for i, sentiment in zip(sorted(id2label), (-1, 0, 1))}
        return mapped

    @staticmethod
    def _normalise(text):
        """The model was trained with usernames and links replaced by placeholders."""
        words = ('@user' if MENTION.match(w) else 'http' if w.startswith('http') else w for w in text.split())
        return ' '.join(words) or '.'  # An empty string would tokenize to nothing

    def predict(self, texts):
        sentiments = []
        with self._lock:
            for start in range(0, len(texts), self.batch_size):
                batch = [self._normalise(str(text)) for text in texts[start:start + self.batch_size]]
                encoded = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt'
                )
                with self.torch.inference_mode():
                    logits = self.model(**encoded).logits
                sentiments.extend(self.sentiment_by_id[int(i)] for i in logits.argmax(dim=-1))
        return sentiments


class ModelLoading(Exception):
    """The model is still loading in the background."""


class BackgroundPredictor:
    """Loads a predictor in a background thread so the API accepts connections straight away.

    A cold Cloud Run instance needs up to a minute to load the transformer. Loading it before the
    server opens its port makes Cloud Run reject requests for that minute; loading it afterwards lets
    requests that don't need the model (comments, charts) answer at once while predictions wait.
    """

    def __init__(self, loader, timeout=120.0):
        self.timeout = timeout
        self._predictor = None
        self._error = None
        self._ready = threading.Event()
        threading.Thread(target=self._load, args=(loader,), name='model-loader', daemon=True).start()

    def _load(self, loader):
        try:
            self._predictor = loader()
        except Exception as error:  # re-raised to callers of predict()
            self._error = error
        finally:
            self._ready.set()

    @property
    def loaded(self):
        return self._ready.is_set() and self._error is None

    @property
    def name(self):
        return self._predictor.name if self._predictor is not None else 'loading'

    def predict(self, texts):
        if not self._ready.wait(self.timeout):
            raise ModelLoading()
        if self._error is not None:
            raise self._error
        return self._predictor.predict(texts)
