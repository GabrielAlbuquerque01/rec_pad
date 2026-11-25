"""Repeated nested validation, audited partitions and external test metrics."""
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .config import BASE_MODELS, INNER_FOLDS, MODEL_IDS, OUTER_FOLDS
from .hybrid import (CompetenceRouter, RevisedRouter, calibrate, cluster_statistics, correct_predictions,
                     estimate_competence, fuse, global_excess_weights, local_excess_weights, select_and_fuse)
from .models import aligned_probabilities, make_models


def make_splits(labels, groups, n_splits, seed):
    grouped = len(np.unique(groups)) != len(groups)
    splitter_class = StratifiedGroupKFold if grouped else StratifiedKFold
    splitter = splitter_class(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(labels)), labels, groups if grouped else None))
    coverage = np.zeros(len(labels), dtype=int)
    for train, test in splits:
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Group overlap")
        if len(np.unique(labels[train])) != 2 or len(np.unique(labels[test])) != 2:
            raise ValueError("Partition missing a class")
        coverage[test] += 1
    np.testing.assert_array_equal(coverage, 1)
    return splits


def metrics(labels, probabilities):
    prediction = probabilities.argmax(axis=1)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    undefined = []

    def ratio(numerator, denominator, name):
        if denominator == 0:
            undefined.append(f"{name}: denominator zero")
            return np.nan
        return float(numerator / denominator)

    auc = float(roc_auc_score(labels, probabilities[:, 1])) if len(np.unique(labels)) == 2 else np.nan
    if np.isnan(auc):
        undefined.append("roc_auc: only one true class")
    sensitivity = ratio(tp, tp + fn, "sensitivity")
    specificity = ratio(tn, tn + fp, "specificity")
    return {"accuracy": float((tp + tn) / len(labels)),
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "precision": ratio(tp, tp + fp, "precision"),
            "sensitivity": sensitivity, "specificity": specificity,
            "roc_auc": auc, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "undefined": "; ".join(undefined)}


def diversity(probabilities, labels):
    """Oracle accuracy, pairwise disagreement and Yule's Q among the base models."""
    predictions = probabilities.argmax(axis=2)
    correct = predictions == labels[:, None]
    disagreement, q_values = [], []
    for i in range(correct.shape[1]):
        for j in range(i + 1, correct.shape[1]):
            disagreement.append(np.mean(predictions[:, i] != predictions[:, j]))
            both, neither = np.sum(correct[:, i] & correct[:, j]), np.sum(~correct[:, i] & ~correct[:, j])
            only_i, only_j = np.sum(correct[:, i] & ~correct[:, j]), np.sum(~correct[:, i] & correct[:, j])
            denominator = both * neither + only_i * only_j
            q_values.append((both * neither - only_i * only_j) / denominator if denominator else np.nan)
    return {"oracle_accuracy": float(correct.any(axis=1).mean()),
            "best_single_accuracy": float(correct.mean(axis=0).max()),
            "mean_disagreement": float(np.mean(disagreement)),
            "mean_q": float(np.nanmean(q_values)) if np.isfinite(q_values).any() else np.nan}


def fit_predict(train_x, train_y, test_x, seed, context, warning_records):
    outputs = []
    for name, model in make_models(seed).items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(train_x, train_y)
            outputs.append(aligned_probabilities(model, test_x))
        warning_records.extend({**context, "model": name, "category": w.category.__name__,
                                "message": str(w.message)} for w in caught)
    return np.stack(outputs, axis=1)


def weight_rows(context, hybrid, region, weights, competence=None):
    return [{**context, "hybrid": hybrid, "region": str(region), "model": model,
             "competence": np.nan if competence is None else float(competence[index]),
             "weight": float(weights[index])} for index, model in enumerate(BASE_MODELS)]


def evaluate_dataset(name, features, labels, groups, seed, repetition, output=Path("results"), plot=False):
    from .presentation import plot_clusters

    destination = output / "folds" / name
    destination.mkdir(parents=True, exist_ok=True)
    collected = {key: [] for key in ["metrics", "weights", "clusters", "selection", "diagnostics", "warnings"]}
    for fold, (train, test) in enumerate(make_splits(labels, groups, OUTER_FOLDS, seed), 1):
        print(f"{name}: repetition {repetition}, external fold {fold}/{OUTER_FOLDS}", flush=True)
        context = {"dataset": name, "repetition": repetition, "fold": fold}
        train_x, train_y, test_x = features.iloc[train], labels[train], features.iloc[test]
        warning_records = []
        oof = np.full((len(train), len(BASE_MODELS), 2), np.nan)
        inner_assignment = np.full(len(train), -1)
        inner_splits = make_splits(train_y, groups[train], INNER_FOLDS, seed)
        for inner_fold, (inner_train, validation) in enumerate(inner_splits, 1):
            oof[validation] = fit_predict(
                train_x.iloc[inner_train], train_y[inner_train], train_x.iloc[validation], seed,
                {**context, "stage": f"inner_{inner_fold}"}, warning_records)
            inner_assignment[validation] = inner_fold
        if not np.isfinite(oof).all() or (inner_assignment < 1).any():
            raise ValueError("Incomplete out-of-fold predictions")
        test_probabilities = fit_predict(train_x, train_y, test_x, seed, {**context, "stage": "outer"},
                                         warning_records)

        # M6 and M7: raw probabilities, accuracy-proportional weights, K-Means with K fixed.
        router = CompetenceRouter(seed).fit(train_x)
        global_c, local_c, global_w, local_w = estimate_competence(oof, train_y, router.labels_)
        test_clusters = router.predict(test_x)
        m6 = fuse(test_probabilities, global_w)
        m7 = fuse(test_probabilities, local_w[test_clusters])

        # M8, M9 and M10: calibrated probabilities and competence above the majority-class rate.
        calibrated_oof, calibrated_test, calibration = calibrate(oof, train_y, inner_assignment, test_probabilities)
        correct = correct_predictions(calibrated_oof, train_y)
        m8_weights = global_excess_weights(correct, train_y)
        m8 = fuse(calibrated_test, m8_weights)
        revised = RevisedRouter(seed).fit(train_x)
        test_points = revised.transform(test_x)
        m9, m9_weights, (m9_k, m9_shrinkage), m9_clusters, m9_selection = select_and_fuse(
            revised.clusters, calibrated_oof, train_y, inner_assignment, calibrated_test, test_points)
        m10, m10_weights, (m10_k, m10_shrinkage), m10_neighbors, m10_selection = select_and_fuse(
            revised.neighbors, calibrated_oof, train_y, inner_assignment, calibrated_test, test_points)

        scores = np.concatenate([test_probabilities, np.stack([m6, m7, m8, m9, m10], axis=1)], axis=1)
        collected["metrics"].extend({**context, "model": model, **metrics(labels[test], scores[:, index])}
                                    for index, model in enumerate(MODEL_IDS))

        collected["weights"].extend(weight_rows(context, "M6", "global", global_w, global_c))
        for cluster in range(len(local_w)):
            collected["weights"].extend(weight_rows(context, "M7", cluster, local_w[cluster], local_c[cluster]))
        collected["weights"].extend(weight_rows(context, "M8", "global", m8_weights, correct.mean(axis=0)))
        region_weights = local_excess_weights(correct, train_y, *cluster_statistics(
            correct, train_y, revised.clusters.labels[m9_k], np.arange(m9_k), m9_k), m9_shrinkage)
        for cluster in range(m9_k):
            collected["weights"].extend(weight_rows(context, "M9", cluster, region_weights[cluster]))
        collected["weights"].extend(weight_rows(context, "M10", "test_mean", m10_weights.mean(axis=0)))

        collected["clusters"].extend({**context, "router": "M7", "k": router.n_clusters, "selected": True, **r}
                                     for r in router.diagnostics())
        collected["clusters"].extend({**context, "router": "v2", "selected": r["k"] == m9_k, **r}
                                     for r in revised.diagnostics())
        for hybrid, parameter_name, selection in [("M9", "K", m9_selection), ("M10", "k", m10_selection)]:
            collected["selection"].extend({**context, "hybrid": hybrid, "parameter_name": parameter_name, **r}
                                          for r in selection)
        decisions = scores.argmax(axis=2)
        collected["diagnostics"].append({
            **context, **diversity(oof, train_y),
            "m7_changes_m6": float(np.mean(decisions[:, 6] != decisions[:, 5])),
            "m9_changes_m8": float(np.mean(decisions[:, 8] != decisions[:, 7])),
            "m10_changes_m8": float(np.mean(decisions[:, 9] != decisions[:, 7])),
            "m6_weight_range": float(np.ptp(global_w)), "m8_weight_range": float(np.ptp(m8_weights)),
            "m9_weight_shift": float(np.abs(m9_weights - m8_weights).sum(axis=1).mean() / 2),
            "m10_weight_shift": float(np.abs(m10_weights - m8_weights).sum(axis=1).mean() / 2),
            "m9_k": m9_k, "m9_shrinkage": m9_shrinkage, "m10_k": m10_k, "m10_shrinkage": m10_shrinkage,
            "m9_min_cluster": int(np.bincount(revised.clusters.labels[m9_k], minlength=m9_k).min()),
        })
        collected["warnings"].extend(warning_records)

        np.savez_compressed(
            destination / f"r{repetition}_f{fold}.npz", train=train, test=test, groups=groups,
            inner_assignment=inner_assignment, y_train=train_y, y_test=labels[test], oof=oof,
            scores=scores, train_clusters=router.labels_, test_clusters=test_clusters,
            global_competence=global_c, local_competence=local_c, global_weights=global_w, local_weights=local_w,
            calibrated_oof=calibrated_oof, calibrated_test=calibrated_test, calibration=calibration,
            m8_weights=m8_weights, m9_weights=m9_weights, m10_weights=m10_weights,
            m9_choice=np.array([m9_k, m9_shrinkage]), m10_choice=np.array([m10_k, m10_shrinkage]),
            m9_train_clusters=revised.clusters.labels[m9_k], m9_test_clusters=m9_clusters,
            m10_neighbors=m10_neighbors)
        if plot:
            plot_clusters(revised.training_features_, revised.clusters.labels[m9_k], name, fold, m9_k, output)
    return collected
