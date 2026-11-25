"""Competence estimation, calibration and probability fusion, without ready-made combiners.

M6 and M7 keep the original formulation: weights proportional to raw accuracy (M7 per K-Means
cluster, K fixed). M8, M9 and M10 fuse calibrated probabilities with weights proportional to the
accuracy above the majority-class rate; M9 uses K-Means regions and M10 neighborhoods, both with
the number of regions (or neighbors) and the shrinkage chosen on the inner folds.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from .config import (CLUSTER_CANDIDATES, KMEANS_INIT, N_CLUSTERS, NEIGHBOR_CANDIDATES,
                     SHRINKAGE_CANDIDATES)
from .preprocessing import make_preprocessor, make_router_preprocessor


# Original formulation (M6, M7)

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


# Regions

def fit_kmeans(points, n_clusters, seed):
    if n_clusters == 1:
        return None, np.zeros(len(points), dtype=int)
    model = KMeans(n_clusters=n_clusters, n_init=KMEANS_INIT, random_state=seed).fit(points)
    return model, model.labels_.astype(int)


def assign_clusters(model, points):
    return np.zeros(len(points), dtype=int) if model is None else model.predict(points).astype(int)


def cluster_diagnostics(points, labels, n_clusters):
    """Within: squared distances to each centroid. Between: size-weighted squared distance of
    each centroid to the overall mean. Their sum is the total sum of squares."""
    center = points.mean(axis=0)
    records = []
    for cluster in range(n_clusters):
        members = points[labels == cluster]
        centroid = members.mean(axis=0) if len(members) else np.full(points.shape[1], np.nan)
        within = float(((members - centroid) ** 2).sum()) if len(members) else 0.0
        between = float(len(members) * ((centroid - center) ** 2).sum()) if len(members) else 0.0
        records.append({"cluster": cluster, "size": int(len(members)), "within_ss": within,
                        "between_ss": between, "within_mean_sq": within / len(members) if len(members) else np.nan})
    return records


class CompetenceRouter:
    """Router of the original local hybrid (M7): standardized attributes, K fixed."""

    def __init__(self, seed, n_clusters=N_CLUSTERS):
        self.seed = seed
        self.n_clusters = n_clusters
        self.preprocessor = make_preprocessor()

    def fit(self, features):
        self.training_features_ = self.preprocessor.fit_transform(features)
        self.kmeans, self.labels_ = fit_kmeans(self.training_features_, self.n_clusters, self.seed)
        return self

    def predict(self, features):
        return assign_clusters(self.kmeans, self.preprocessor.transform(features))

    def diagnostics(self):
        return cluster_diagnostics(self.training_features_, self.labels_, self.n_clusters)


# Calibration

def logit(probabilities):
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


def fit_platt(probabilities, labels):
    """Platt scaling on the logit of the positive-class probability. Platt's smoothed targets
    keep the parameters finite even when the scores separate the classes perfectly."""
    features = logit(probabilities)
    positives = labels.sum()
    negatives = len(labels) - positives
    targets = np.where(labels == 1, (positives + 1) / (positives + 2), 1 / (negatives + 2))

    def objective(params):
        logits = params[0] * features + params[1]
        residual = expit(logits) - targets
        loss = np.sum(np.logaddexp(0, logits) - targets * logits)
        return loss, np.array([residual @ features, residual.sum()])

    params = minimize(objective, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B").x
    if not np.isfinite(params).all():
        raise ValueError("Calibration did not produce finite parameters")
    return params


def apply_platt(params, probabilities):
    positive = expit(params[0] * logit(probabilities) + params[1])
    return np.stack([1 - positive, positive], axis=-1)


def calibrate(oof, labels, inner_assignment, test):
    """Returns cross-fitted calibrated OOF probabilities (each inner fold calibrated with
    parameters fitted on the other folds), the test probabilities calibrated with parameters
    fitted on all OOF predictions, and those parameters."""
    n_models = oof.shape[1]
    crossfit = np.empty_like(oof)
    for fold in np.unique(inner_assignment):
        held, rest = inner_assignment == fold, inner_assignment != fold
        for model in range(n_models):
            params = fit_platt(oof[rest, model, 1], labels[rest])
            crossfit[held, model] = apply_platt(params, oof[held, model, 1])
    params = np.array([fit_platt(oof[:, model, 1], labels) for model in range(n_models)])
    calibrated_test = np.stack([apply_platt(params[model], test[:, model, 1])
                                for model in range(n_models)], axis=1)
    return crossfit, calibrated_test, params


# Competence above the majority-class rate (M8, M9, M10)

def correct_predictions(probabilities, labels):
    return probabilities.argmax(axis=2) == labels[:, None]


def excess_weights(hits, counts, positives, global_accuracy, global_positive_rate, shrinkage, fallback):
    """Weights proportional to the accuracy above the majority-class rate of the region.

    Accuracy and class rate are shrunk toward their global values with `shrinkage`
    pseudo-observations, so small regions borrow strength from the whole training set.
    A model that does not beat the majority class of the region gets zero weight; when no
    model does, the region uses the fallback weights.
    """
    hits = np.atleast_2d(np.asarray(hits, dtype=float))
    counts = np.atleast_1d(np.asarray(counts, dtype=float))
    positives = np.atleast_1d(np.asarray(positives, dtype=float))
    total = counts + shrinkage
    valid = total > 0
    safe_total = np.where(valid, total, 1)
    accuracy = (hits + shrinkage * np.asarray(global_accuracy)) / safe_total[:, None]
    positive_rate = (positives + shrinkage * global_positive_rate) / safe_total
    baseline = np.maximum(positive_rate, 1 - positive_rate)
    excess = np.clip(accuracy - baseline[:, None], 0, None)
    sums = excess.sum(axis=1)
    usable = valid & (sums > 0)
    weights = np.tile(np.asarray(fallback, dtype=float), (len(hits), 1))
    weights[usable] = excess[usable] / sums[usable, None]
    return weights


def global_excess_weights(correct, labels):
    uniform = np.full(correct.shape[1], 1 / correct.shape[1])
    return excess_weights(correct.sum(axis=0), len(labels), labels.sum(), correct.mean(axis=0),
                          labels.mean(), 0, uniform)[0]


def local_excess_weights(correct, labels, hits, counts, positives, shrinkage):
    fallback = global_excess_weights(correct, labels)
    return excess_weights(hits, counts, positives, correct.mean(axis=0), labels.mean(), shrinkage, fallback)


def cluster_statistics(correct, labels, train_regions, query_regions, n_regions):
    hits = np.zeros((n_regions, correct.shape[1]))
    np.add.at(hits, train_regions, correct)
    counts = np.bincount(train_regions, minlength=n_regions)
    positives = np.bincount(train_regions, weights=labels, minlength=n_regions)
    return hits[query_regions], counts[query_regions], positives[query_regions]


def neighbor_statistics(correct, labels, neighbors):
    return (correct[neighbors].sum(axis=1), np.full(len(neighbors), neighbors.shape[1]),
            labels[neighbors].sum(axis=1))


def nearest_neighbors(reference, queries, k):
    model = NearestNeighbors(n_neighbors=min(k, len(reference))).fit(reference)
    return model.kneighbors(queries, return_distance=False)


class ClusterRegions:
    """Regions of M9: K-Means clusters in the revised router space, one fit per candidate K."""
    candidates = [(k, s) for k in CLUSTER_CANDIDATES for s in sorted(SHRINKAGE_CANDIDATES, reverse=True)]

    def __init__(self, points, seed):
        self.points = points
        self.models, self.labels = {}, {}
        for k in CLUSTER_CANDIDATES:
            self.models[k], self.labels[k] = fit_kmeans(points, k, seed)

    def prepare(self, rest, held):
        return rest, held

    def statistics(self, context, k, correct, labels):
        rest, held = context
        return cluster_statistics(correct, labels, self.labels[k][rest], self.labels[k][held], k)

    def query(self, k, correct, labels, test_points):
        assignment = assign_clusters(self.models[k], test_points)
        return cluster_statistics(correct, labels, self.labels[k], assignment, k), assignment


class NeighborRegions:
    """Regions of M10: the k nearest training examples of each query in the router space."""
    candidates = [(k, s) for k in sorted(NEIGHBOR_CANDIDATES, reverse=True)
                  for s in sorted(SHRINKAGE_CANDIDATES, reverse=True)]

    def __init__(self, points):
        self.points = points

    def prepare(self, rest, held):
        # Neighbors are sorted by distance, so smaller neighborhoods are prefixes of the largest.
        return nearest_neighbors(self.points[rest], self.points[held], max(NEIGHBOR_CANDIDATES))

    def statistics(self, context, k, correct, labels):
        return neighbor_statistics(correct, labels, context[:, :k])

    def query(self, k, correct, labels, test_points):
        neighbors = nearest_neighbors(self.points, test_points, k)
        return neighbor_statistics(correct, labels, neighbors), neighbors


def select_and_fuse(regions, calibrated_oof, labels, inner_assignment, calibrated_test, test_points):
    """Chooses (regions, shrinkage) by the ROC-AUC of OOF scores fused with competences
    estimated only on the other inner folds, then fuses the test probabilities with the
    competences of the whole external training set.

    Candidates are ordered from the smoothest configuration, and ties keep the first one;
    for K-Means, K=1 reproduces the global weights.
    """
    correct = correct_predictions(calibrated_oof, labels)
    candidates = regions.candidates
    scores = np.empty((len(candidates), len(labels)))
    for fold in np.unique(inner_assignment):
        rest = np.flatnonzero(inner_assignment != fold)
        held = np.flatnonzero(inner_assignment == fold)
        context = regions.prepare(rest, held)
        for index, (parameter, shrinkage) in enumerate(candidates):
            statistics = regions.statistics(context, parameter, correct[rest], labels[rest])
            weights = local_excess_weights(correct[rest], labels[rest], *statistics, shrinkage)
            scores[index, held] = fuse(calibrated_oof[held], weights)[:, 1]
    inner_auc = np.array([roc_auc_score(labels, row) for row in scores])
    best = int(np.argmax(inner_auc))
    parameter, shrinkage = candidates[best]
    statistics, assignment = regions.query(parameter, correct, labels, test_points)
    weights = local_excess_weights(correct, labels, *statistics, shrinkage)
    selection = [{"parameter": p, "shrinkage": s, "inner_auc": float(a), "selected": i == best}
                 for i, ((p, s), a) in enumerate(zip(candidates, inner_auc))]
    return fuse(calibrated_test, weights), weights, (parameter, shrinkage), assignment, selection


class RevisedRouter:
    """Router space shared by M9 and M10, fitted on the external training set without labels."""

    def __init__(self, seed):
        self.seed = seed
        self.preprocessor = make_router_preprocessor()

    def fit(self, features):
        self.training_features_ = self.preprocessor.fit_transform(features)
        self.clusters = ClusterRegions(self.training_features_, self.seed)
        self.neighbors = NeighborRegions(self.training_features_)
        return self

    def transform(self, features):
        return self.preprocessor.transform(features)

    def diagnostics(self):
        return [{"k": k, **record} for k in CLUSTER_CANDIDATES
                for record in cluster_diagnostics(self.training_features_, self.clusters.labels[k], k)]
