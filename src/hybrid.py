"""Original competence estimation and probability fusion (no ready-made combiner)."""
import numpy as np
from sklearn.cluster import KMeans

from .config import N_CLUSTERS, SEED
from .preprocessing import make_preprocessor


def normalize_competence(competence, fallback=None):
    competence = np.asarray(competence, dtype=float)
    if (competence < 0).any() or not np.isfinite(competence).all():
        raise ValueError("Competence must be finite and nonnegative")
    if competence.sum() > 0:
        return competence / competence.sum()
    return np.full(len(competence), 1 / len(competence)) if fallback is None else fallback.copy()


def estimate_competence(oof_probabilities, labels, clusters, n_clusters=N_CLUSTERS):
    # Shape: examples x models x classes. argmax consistently resolves ties to 0.
    correct = oof_probabilities.argmax(axis=2) == labels[:, None]
    global_competence = correct.mean(axis=0)
    global_weights = normalize_competence(global_competence)
    local_competence = np.full((n_clusters, correct.shape[1]), np.nan)
    local_weights = np.tile(global_weights, (n_clusters, 1))
    for cluster in range(n_clusters):
        mask = clusters == cluster
        if mask.any():
            local_competence[cluster] = correct[mask].mean(axis=0)
            local_weights[cluster] = normalize_competence(local_competence[cluster], global_weights)
    return global_competence, local_competence, global_weights, local_weights


def fuse(probabilities, weights):
    weights = np.asarray(weights)
    if (weights < 0).any() or not np.isfinite(weights).all():
        raise ValueError("Invalid weights")
    np.testing.assert_allclose(weights.sum(axis=-1), 1, atol=1e-12)
    if weights.ndim == 1:
        return np.einsum("nmj,m->nj", probabilities, weights)
    return np.einsum("nmj,nm->nj", probabilities, weights)


class CompetenceRouter:
    def __init__(self):
        self.preprocessor = make_preprocessor()
        self.kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=SEED)

    def fit(self, features):
        self.training_features_ = self.preprocessor.fit_transform(features)
        self.kmeans.fit(self.training_features_)
        return self

    def predict(self, features):
        return self.kmeans.predict(self.preprocessor.transform(features))

    def diagnostics(self):
        points = self.training_features_
        center = points.mean(axis=0)
        records = []
        for cluster, centroid in enumerate(self.kmeans.cluster_centers_):
            members = points[self.kmeans.labels_ == cluster]
            within = float(((members - centroid) ** 2).sum())
            between = float(len(members) * ((centroid - center) ** 2).sum())
            records.append({"cluster": cluster, "size": len(members), "within_ss": within,
                            "between_ss": between, "within_mean_sq": within / len(members) if len(members) else np.nan})
        return records
