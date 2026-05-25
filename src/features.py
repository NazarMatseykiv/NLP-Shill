from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=60_000,
        sublinear_tf=True,
    )


def word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30_000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w\w+\b",
    )


def combined_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            ("char", char_vectorizer()),
            ("word", word_vectorizer()),
        ]
    )
