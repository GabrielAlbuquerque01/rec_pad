"""Nested validation, audited partitions, and external test metrics."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .config import INNER_FOLDS, METRICS, MODEL_IDS, OUTER_FOLDS, SEED
from .hybrid import CompetenceRouter, estimate_competence, fuse
from .models import aligned_probabilities, make_models


def make_splits(labels, groups, n_splits):
    grouped = len(np.unique(groups)) != len(groups)
    splitter_class = StratifiedGroupKFold if grouped else StratifiedKFold
    splitter = splitter_class(n_splits=n_splits, shuffle=True, random_state=SEED)
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
    return {"accuracy": float((tp + tn) / len(labels)),
            "precision": ratio(tp, tp + fp, "precision"),
            "sensitivity": ratio(tp, tp + fn, "sensitivity"),
            "specificity": ratio(tn, tn + fp, "specificity"),
            "roc_auc": auc, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "undefined": "; ".join(undefined)}


def fit_predict(train_x, train_y, test_x, context, warning_records):
    outputs = []
    for name, model in make_models().items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(train_x, train_y)
            outputs.append(aligned_probabilities(model, test_x))
        warning_records.extend({**context, "model": name, "category": w.category.__name__,
                                "message": str(w.message)} for w in caught)
    return np.stack(outputs, axis=1)


def evaluate_dataset(name, features, labels, groups, output=Path("results")):
    from .presentation import plot_clusters

    destination = output / "folds" / name
    destination.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for fold, (train, test) in enumerate(make_splits(labels, groups, OUTER_FOLDS), 1):
        print(f"{name}: external fold {fold}/{OUTER_FOLDS}", flush=True)
        context = {"dataset": name, "fold": fold}
        train_x, train_y = features.iloc[train], labels[train]
        warning_records = []
        oof = np.full((len(train), 5, 2), np.nan)
        inner_assignment = np.full(len(train), -1)
        for inner_fold, (inner_train, validation) in enumerate(make_splits(train_y, groups[train], INNER_FOLDS), 1):
            oof[validation] = fit_predict(
                train_x.iloc[inner_train], train_y[inner_train], train_x.iloc[validation],
                {**context, "stage": f"inner_{inner_fold}"}, warning_records)
            inner_assignment[validation] = inner_fold
        if not np.isfinite(oof).all() or (inner_assignment < 1).any():
            raise ValueError("Incomplete out-of-fold predictions")
        router = CompetenceRouter().fit(train_x)
        global_c, local_c, global_w, local_w = estimate_competence(oof, train_y, router.kmeans.labels_)
        test_probabilities = fit_predict(train_x, train_y, features.iloc[test],
                                         {**context, "stage": "outer"}, warning_records)
        test_clusters = router.predict(features.iloc[test])
        global_scores = fuse(test_probabilities, global_w)
        local_scores = fuse(test_probabilities, local_w[test_clusters])
        scores = np.concatenate([test_probabilities, global_scores[:, None], local_scores[:, None]], axis=1)
        fold_records = [{**context, "model": model, **metrics(labels[test], scores[:, i])}
                        for i, model in enumerate(MODEL_IDS)]
        all_metrics.extend(fold_records)
        prefix = destination / str(fold)
        pd.DataFrame(fold_records).to_csv(f"{prefix}_metrics.csv", index=False)
        np.savez_compressed(f"{prefix}_predictions.npz", train=train, test=test,
                            groups=groups, inner_assignment=inner_assignment, y_train=train_y,
                            y_test=labels[test], oof=oof, scores=scores,
                            train_clusters=router.kmeans.labels_, test_clusters=test_clusters,
                            global_competence=global_c, local_competence=local_c,
                            global_weights=global_w, local_weights=local_w)
        weights = []
        for index, model in enumerate(MODEL_IDS[:5]):
            weights.append({**context, "model": model, "region": "global", "competence": global_c[index], "weight": global_w[index]})
            for cluster in range(len(local_w)):
                weights.append({**context, "model": model, "region": str(cluster),
                                "competence": local_c[cluster, index], "weight": local_w[cluster, index]})
        pd.DataFrame(weights).to_csv(f"{prefix}_weights.csv", index=False)
        pd.DataFrame([{**context, **r} for r in router.diagnostics()]).to_csv(f"{prefix}_clusters.csv", index=False)
        Path(f"{prefix}_warnings.json").write_text(json.dumps(warning_records, indent=2), encoding="utf-8")
        plot_clusters(router, name, fold, output)
    return all_metrics


def aggregate(records, output=Path("results")):
    frame = pd.DataFrame(records)
    frame.to_csv(output / "metrics_by_fold.csv", index=False)
    summary = frame.groupby(["dataset", "model"])[METRICS].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / "summary.csv", index=False)
    return frame, summary.reset_index()
