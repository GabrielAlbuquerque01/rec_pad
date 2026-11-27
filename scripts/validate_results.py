"""Recompute saved metrics and fusions and verify every nested partition.

Run as python -m scripts.validate_results.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import friedmanchisquare, rankdata, studentized_range

from src.config import (ABLATION_PAIRS, ALPHA, BASE_MODELS, CONTROL_MODEL, DATASETS, INNER_FOLDS, METRICS,
                        MODEL_IDS, OUTER_FOLDS, PRIMARY_MODELS, REPETITION_SEEDS)
from src.datasets import load_dataset
from src.evaluation import metrics
from src.hybrid import (apply_platt, calibrate, cluster_statistics, correct_predictions, estimate_competence, fuse,
                        global_excess_weights, local_excess_weights, neighbor_statistics)
from src.statistical_tests import expected_records, holm, paired_test


def check_fold(saved, labels, groups):
    train, test = saved["train"], saved["test"]
    assert_array_equal(np.sort(np.concatenate([train, test])), np.arange(len(labels)))
    assert_array_equal(saved["groups"], groups)
    assert_array_equal(saved["y_train"], labels[train])
    assert_array_equal(saved["y_test"], labels[test])
    assert not set(groups[train]) & set(groups[test])
    assignment = saved["inner_assignment"]
    assert set(assignment) == set(range(1, INNER_FOLDS + 1))
    for inner in range(1, INNER_FOLDS + 1):
        assert not set(groups[train[assignment != inner]]) & set(groups[train[assignment == inner]])
    y_train, oof, scores = saved["y_train"], saved["oof"], saved["scores"]
    raw_test = scores[:, :len(BASE_MODELS)]
    assert np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all()
    assert_allclose(scores.sum(axis=2), 1, atol=1e-8)

    gc, lc, gw, lw = estimate_competence(oof, y_train, saved["train_clusters"])
    for key, expected in [("global_competence", gc), ("local_competence", lc),
                          ("global_weights", gw), ("local_weights", lw)]:
        assert_allclose(saved[key], expected, equal_nan=True)
    assert_allclose(scores[:, 5], fuse(raw_test, gw))
    assert_allclose(scores[:, 6], fuse(raw_test, lw[saved["test_clusters"]]))

    calibrated_oof, calibrated_test, params = calibrate(oof, y_train, assignment, raw_test)
    assert_allclose(saved["calibration"], params, atol=1e-8)
    assert_allclose(saved["calibrated_oof"], calibrated_oof, atol=1e-8)
    assert_allclose(saved["calibrated_test"], calibrated_test, atol=1e-8)
    for model in range(len(BASE_MODELS)):
        assert_allclose(apply_platt(saved["calibration"][model], raw_test[:, model, 1]), saved["calibrated_test"][:, model])
    correct = correct_predictions(saved["calibrated_oof"], y_train)
    m8 = global_excess_weights(correct, y_train)
    assert_allclose(saved["m8_weights"], m8)
    assert_allclose(scores[:, 7], fuse(saved["calibrated_test"], m8))

    k9, shrink9 = saved["m9_choice"]
    m9 = local_excess_weights(correct, y_train, *cluster_statistics(
        correct, y_train, saved["m9_train_clusters"], saved["m9_test_clusters"], int(k9)), shrink9)
    assert_allclose(saved["m9_weights"], m9)
    k10, shrink10 = saved["m10_choice"]
    assert saved["m10_neighbors"].shape == (len(test), int(k10))
    assert saved["m10_neighbors"].min() >= 0
    assert saved["m10_neighbors"].max() < len(train)
    m10 = local_excess_weights(correct, y_train, *neighbor_statistics(correct, y_train, saved["m10_neighbors"]), shrink10)
    assert_allclose(saved["m10_weights"], m10)
    for index, weights in [(8, m9), (9, m10)]:
        assert (weights >= 0).all()
        assert_allclose(weights.sum(axis=1), 1)
        assert_allclose(scores[:, index], fuse(saved["calibrated_test"], weights))
    if int(k9) == 1:
        assert_allclose(scores[:, 8], scores[:, 7])
    return test


def check_pairwise(statistics, matrix):
    """Check family membership as well as raw and multiplicity-adjusted inference."""
    expected = {("control", CONTROL_MODEL, model) for model in PRIMARY_MODELS if model != CONTROL_MODEL}
    expected.update(("ablation", first, second) for first, second in ABLATION_PAIRS)
    actual = [(row["family"], row["model"], row["reference"]) for row in statistics["pairwise"]]
    assert len(actual) == len(expected) and set(actual) == expected
    for row in statistics["pairwise"]:
        recomputed = paired_test(matrix[row["model"]], matrix[row["reference"]])
        keys = ["mean_difference", "wilcoxon_p", "sign_p", "wins", "losses", "ties"]
        assert_allclose([recomputed[k] for k in keys], [row[k] for k in keys])
    for family in ("control", "ablation"):
        rows = [row for row in statistics["pairwise"] if row["family"] == family]
        adjusted = holm([row["wilcoxon_p"] for row in rows])
        assert_allclose(adjusted, [row["wilcoxon_p_holm"] for row in rows])
        assert_array_equal(adjusted < ALPHA, [row["significant"] for row in rows])


def validate(output=Path("results")):
    frame = pd.read_csv(output / "metrics_by_fold.csv")
    expected = len(DATASETS) * len(MODEL_IDS) * len(REPETITION_SEEDS) * OUTER_FOLDS
    actual = list(frame[["dataset", "model", "repetition", "fold"]].itertuples(index=False, name=None))
    if len(actual) != expected or set(actual) != expected_records():
        raise AssertionError(f"Expected {expected} unique records")
    checked = 0
    for name in DATASETS:
        _, labels, groups, _ = load_dataset(name)
        for repetition in range(1, len(REPETITION_SEEDS) + 1):
            coverage = np.zeros(len(labels), dtype=int)
            for fold in range(1, OUTER_FOLDS + 1):
                with np.load(output / "folds" / name / f"r{repetition}_f{fold}.npz") as saved:
                    test = check_fold(saved, labels, groups)
                    coverage[test] += 1
                    rows = frame[(frame.dataset == name) & (frame.repetition == repetition) & (frame.fold == fold)]
                    for index, model in enumerate(MODEL_IDS):
                        row = rows[rows.model == model].iloc[0]
                        recomputed = metrics(labels[test], saved["scores"][:, index])
                        assert_allclose(row[METRICS].to_numpy(dtype=float), [recomputed[m] for m in METRICS], equal_nan=True)
                        assert sum(row[m] for m in ["tn", "fp", "fn", "tp"]) == len(test)
                        checked += 1
            assert_array_equal(coverage, 1)

    summary = pd.read_csv(output / "summary.csv").set_index(["dataset", "model"])
    assert len(summary) == len(DATASETS) * len(MODEL_IDS)
    for key, subset in frame.groupby(["dataset", "model"]):
        for metric in METRICS:
            assert_allclose(summary.loc[key, f"{metric}_mean"], subset[metric].mean(), equal_nan=True)
            assert_allclose(summary.loc[key, f"{metric}_std"], subset[metric].std(ddof=1), equal_nan=True)
            assert summary.loc[key, f"{metric}_count"] == subset[metric].count()
    matrix = frame.groupby(["dataset", "model"]).roc_auc.mean().unstack().loc[list(DATASETS), MODEL_IDS]
    assert_allclose(matrix, pd.read_csv(output / "auc_matrix.csv", index_col=0).loc[list(DATASETS), MODEL_IDS])
    statistics = json.loads((output / "statistics.json").read_text(encoding="utf-8"))
    primary = matrix[PRIMARY_MODELS].to_numpy()
    statistic, pvalue = friedmanchisquare(*primary.T)
    assert_allclose([statistic, pvalue], [statistics["friedman"]["statistic"], statistics["friedman"]["pvalue"]])
    mean_ranks = rankdata(-primary, axis=1).mean(axis=0)
    saved_ranks = pd.read_csv(output / "mean_ranks.csv", index_col=0).loc[PRIMARY_MODELS, "mean_rank"]
    assert_allclose(mean_ranks, saved_ranks)
    if statistics["friedman"]["reject_null"]:
        # Independent studentized-range computation checks the package convention.
        count, blocks = len(PRIMARY_MODELS), len(DATASETS)
        standard_error = np.sqrt(count * (count + 1) / (6 * blocks))
        q = np.abs(mean_ranks[:, None] - mean_ranks[None, :]) / standard_error * np.sqrt(2)
        saved_pvalues = pd.read_csv(output / "nemenyi_pvalues.csv", index_col=0).loc[PRIMARY_MODELS, PRIMARY_MODELS]
        assert_allclose(studentized_range.sf(q, count, np.inf), saved_pvalues, atol=1e-10)
        assert_allclose(statistics["friedman"]["critical_difference"],
                        studentized_range.ppf(.95, count, np.inf) / np.sqrt(2) * standard_error)
    check_pairwise(statistics, matrix)

    report = {"status": "passed", "datasets": len(DATASETS), "repetitions": len(REPETITION_SEEDS),
              "external_folds": len(DATASETS) * len(REPETITION_SEEDS) * OUTER_FOLDS,
              "recomputed_metric_records": checked,
              "checks": ["outer and inner group separation", "external coverage per repetition",
                         "label and group identity", "probability sums", "M6/M7 competence, weights and fusion",
                         "cross-fitted calibration", "M8/M9/M10 weights and fusion", "K=1 equivalence of M9 and M8",
                         "all test metrics", "means and standard deviations", "Friedman and ranks",
                         "independent Nemenyi convention and critical difference", "paired tests"]}
    (output / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    validate()
