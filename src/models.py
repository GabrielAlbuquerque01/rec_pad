import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .config import SEED
from .preprocessing import make_preprocessor


def make_models():
    estimators = [
        SVC(kernel="linear", C=1, probability=True, random_state=SEED),
        SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=SEED),
        MLPClassifier(hidden_layer_sizes=(32,), activation="tanh", solver="sgd",
                      momentum=0.9, learning_rate_init=0.01, alpha=0.0001,
                      max_iter=1000, random_state=SEED),
        GaussianNB(),
        KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean"),
    ]
    return {f"M{i}": Pipeline([("preprocess", make_preprocessor()), ("model", model)])
            for i, model in enumerate(estimators, 1)}


def aligned_probabilities(model, features):
    probabilities = model.predict_proba(features)
    order = [list(model.classes_).index(label) for label in (0, 1)]
    aligned = probabilities[:, order]
    if not np.isfinite(aligned).all() or (aligned < 0).any():
        raise ValueError("Invalid probabilities")
    np.testing.assert_allclose(aligned.sum(axis=1), 1, atol=1e-8)
    return aligned
