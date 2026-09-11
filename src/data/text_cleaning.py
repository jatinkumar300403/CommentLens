import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Negations and contrast words carry sentiment, so they are kept in the text
SENTIMENT_WORDS = {'not', 'but', 'however', 'no', 'yet'}

NLTK_RESOURCES = {'stopwords': 'corpora/stopwords', 'wordnet': 'corpora/wordnet'}


def ensure_nltk_data() -> None:
    """Download the NLTK corpora used for preprocessing, only if missing."""
    for resource, path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


@lru_cache(maxsize=1)
def _load_resources():
    ensure_nltk_data()
    return set(stopwords.words('english')) - SENTIMENT_WORDS, WordNetLemmatizer()


def preprocess_comment(comment) -> str:
    """Lowercase, strip non-English characters, drop stopwords and lemmatize a comment.

    Shared by the training pipeline and the Flask API so both see identical text.
    """
    if not isinstance(comment, str):
        return ''

    stop_words, lemmatizer = _load_resources()

    comment = comment.lower().strip()
    comment = re.sub(r'\n', ' ', comment)
    comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

    words = [word for word in comment.split() if word not in stop_words]
    return ' '.join(lemmatizer.lemmatize(word) for word in words)
