from src.data.text_cleaning import preprocess_comment


def test_keeps_negations_that_flip_sentiment():
    assert preprocess_comment('This is not good') == 'not good'


def test_lowercases_lemmatizes_and_drops_stopwords():
    assert preprocess_comment('The Videos were GREAT') == 'video great'


def test_strips_emoji_and_non_english_characters():
    assert preprocess_comment('nice ❤️ 好') == 'nice'


def test_joins_lines():
    assert preprocess_comment('first line\nsecond line') == 'first line second line'


def test_non_string_input_becomes_empty():
    assert preprocess_comment(None) == ''
    assert preprocess_comment(float('nan')) == ''
