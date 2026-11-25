"""Independently recompute saved metrics and verify every nested partition."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from src.config import DATASETS, METRICS, MODEL_IDS
from src.datasets import load_dataset
from src.evaluation import metrics
from src.hybrid import estimate_competence, fuse
from scipy.stats import friedmanchisquare, rankdata, studentized_range


def validate(output=Path("results")):
    frame = pd.read_csv(output / "metrics_by_fold.csv")
    if len(frame) != 350 or frame.duplicated(["dataset", "model", "fold"]).any():
        raise AssertionError("Expected 350 unique records")
    checked = 0
    for name in DATASETS:
        features, labels, groups, _ = load_dataset(name)
        test_coverage = np.zeros(len(labels), dtype=int)
        for fold in range(1, 6):
            saved = np.load(output / "folds" / name / f"{fold}_predictions.npz")
            train, test = saved["train"], saved["test"]
            assert_array_equal(np.sort(np.concatenate([train, test])), np.arange(len(labels)))
            assert_array_equal(saved["groups"], groups)
            assert_array_equal(saved["y_train"], labels[train])
            assert_array_equal(saved["y_test"], labels[test])
            assert not (set(groups[train]) & set(groups[test]))
            test_coverage[test] += 1
            assignment = saved["inner_assignment"]
            assert set(assignment) == {1, 2, 3}
            for inner in range(1, 4):
                inner_train, validation = train[assignment != inner], train[assignment == inner]
                assert not (set(groups[inner_train]) & set(groups[validation]))
                assert not (set(validation) & set(test))
            gc, lc, gw, lw = estimate_competence(saved["oof"], labels[train], saved["train_clusters"])
            for key, expected in [("global_competence", gc), ("local_competence", lc),
                                  ("global_weights", gw), ("local_weights", lw)]:
                assert_allclose(saved[key], expected, equal_nan=True)
            scores = saved["scores"]
            assert_allclose(scores.sum(axis=2), 1, atol=1e-8)
            assert_allclose(scores[:, 5], fuse(scores[:, :5], gw))
            assert_allclose(scores[:, 6], fuse(scores[:, :5], lw[saved["test_clusters"]]))
            for index, model in enumerate(MODEL_IDS):
                row = frame[(frame.dataset == name) & (frame.fold == fold) & (frame.model == model)].iloc[0]
                recomputed = metrics(labels[test], scores[:, index])
                assert_allclose(row[METRICS].to_numpy(dtype=float), [recomputed[m] for m in METRICS], equal_nan=True)
                assert sum(row[m] for m in ["tn", "fp", "fn", "tp"]) == len(test)
                checked += 1
        assert_array_equal(test_coverage, 1)
    summary = pd.read_csv(output / "summary.csv").set_index(["dataset", "model"])
    assert len(summary) == 70
    for key, subset in frame.groupby(["dataset", "model"]):
        for metric in METRICS:
            assert_allclose(summary.loc[key, f"{metric}_mean"], subset[metric].mean(), equal_nan=True)
            assert_allclose(summary.loc[key, f"{metric}_std"], subset[metric].std(ddof=1), equal_nan=True)
    matrix = frame.groupby(["dataset", "model"]).roc_auc.mean().unstack().loc[list(DATASETS), MODEL_IDS]
    saved_matrix = pd.read_csv(output / "auc_matrix.csv", index_col=0).loc[list(DATASETS), MODEL_IDS]
    assert_allclose(matrix, saved_matrix)
    statistics = json.loads((output / "statistics.json").read_text())
    statistic, pvalue = friedmanchisquare(*matrix.to_numpy().T)
    assert_allclose([statistic, pvalue], [statistics["statistic"], statistics["pvalue"]])
    mean_ranks = rankdata(-matrix.to_numpy(), axis=1).mean(axis=0)
    saved_ranks = pd.read_csv(output / "mean_ranks.csv", index_col=0).loc[MODEL_IDS, "mean_rank"]
    assert_allclose(mean_ranks, saved_ranks)
    if statistics["reject_null"]:
        # Independent studentized-range calculation checks package convention.
        standard_error = np.sqrt(7 * 8 / (6 * 10))
        q = np.abs(mean_ranks[:, None] - mean_ranks[None, :]) / standard_error * np.sqrt(2)
        pvalues = studentized_range.sf(q, 7, np.inf)
        saved_pvalues = pd.read_csv(output / "nemenyi_pvalues.csv", index_col=0).loc[MODEL_IDS, MODEL_IDS]
        assert_allclose(pvalues, saved_pvalues, atol=1e-10)
        assert_allclose(statistics["critical_difference"], studentized_range.ppf(.95, 7, np.inf) / np.sqrt(2) * standard_error)
    report = {"status": "passed", "datasets": 10, "external_folds": 50,
              "recomputed_metric_records": checked, "checks": ["outer group separation", "inner group separation",
              "external and internal coverage", "label and group identity", "probability sums",
              "competence and weights", "global and local fusion", "all test metrics", "70 means and standard deviations",
              "Friedman and ranks", "independent Nemenyi convention and critical difference"]}
    (output / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    validate()
