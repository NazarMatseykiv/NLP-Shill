from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import char_vectorizer, combined_vectorizer, word_vectorizer


def get_model_candidates(feature_set: str = "combined") -> dict[str, Pipeline]:
    if feature_set == "char":
        vect = char_vectorizer()
    elif feature_set == "word":
        vect = word_vectorizer()
    else:
        vect = combined_vectorizer()

    return {
        "logistic_regression": Pipeline(
            [
                ("tfidf", vect),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        C=1.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "linear_svc": Pipeline(
            [
                ("tfidf", vect),
                (
                    "clf",
                    LinearSVC(
                        class_weight="balanced",
                        C=1.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "complement_nb": Pipeline(
            [
                ("tfidf", vect),
                ("clf", ComplementNB(alpha=0.1)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("tfidf", vect),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced_subsample",
                        max_depth=24,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
