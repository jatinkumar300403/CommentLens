import numpy as np

from src.model.predictors import SklearnPredictor, TransformerPredictor


class PassThroughVectorizer:
    """Stands in for TF-IDF: hands the preprocessed text straight to the model."""

    def transform(self, texts):
        self.seen = list(texts)
        return self.seen


class EchoModel:
    def predict(self, texts):
        return np.array([1 if 'video' in text else 0 for text in texts])


def test_sklearn_predictor_preprocesses_before_predicting():
    vectorizer = PassThroughVectorizer()
    predictor = SklearnPredictor(EchoModel(), vectorizer)

    # 'Videos' lemmatizes to 'video' and the stopwords around it are dropped
    assert predictor.predict(['The Videos were great']) == [1]
    assert vectorizer.seen == ['video great']


def test_sklearn_predictor_returns_plain_ints():
    predictor = SklearnPredictor(EchoModel(), PassThroughVectorizer())

    assert [type(s) for s in predictor.predict(['a video', 'nothing here'])] == [int, int]


def test_label_map_reads_model_label_names():
    mapping = TransformerPredictor._label_map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})

    assert mapping == {0: -1, 1: 0, 2: 1}


def test_label_map_falls_back_to_class_order_for_unnamed_labels():
    mapping = TransformerPredictor._label_map({0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2'})

    assert mapping == {0: -1, 1: 0, 2: 1}


def test_normalise_replaces_handles_and_links_but_keeps_emoji():
    text = TransformerPredictor._normalise('@someuser loved it \U0001F525 https://youtu.be/abc')

    assert text == '@user loved it \U0001F525 http'


def test_normalise_keeps_empty_comments_tokenizable():
    assert TransformerPredictor._normalise('   ') == '.'
