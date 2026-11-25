"""Aggregation and hypothesis tests with datasets as blocks; folds are never treated as datasets."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import numpy as np
import pandas as pd
import scikit_posthocs as posthocs
from scipy.stats import binomtest, friedmanchisquare, rankdata, studentized_range, wilcoxon

from .config import (ABLATION_PAIRS, ALPHA, CONTROL_MODEL, DATASETS, METRICS, MODEL_IDS, OUTER_FOLDS,
                     PRIMARY_MODELS, REPETITION_SEEDS)

TIE_TOLERANCE = 1e-12


def aggregate(frame, output=Path("results")):
    summary = frame.groupby(["dataset", "model"])[METRICS].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    return summary


def expected_records():
    return {(dataset, model, repetition, fold) for dataset in DATASETS for model in MODEL_IDS
            for repetition in range(1, len(REPETITION_SEEDS) + 1) for fold in range(1, OUTER_FOLDS + 1)}


def auc_matrix(frame):
    return frame.groupby(["dataset", "model"]).roc_auc.mean().unstack().loc[list(DATASETS), MODEL_IDS]


def rank_matrix(matrix):
    return pd.DataFrame(rankdata(-matrix.to_numpy(), axis=1, method="average"),
                        index=matrix.index, columns=matrix.columns)


def critical_difference(n_models, n_datasets, alpha=ALPHA):
    # Nemenyi q = Tukey studentized range / sqrt(2), infinite degrees of freedom.
    q_alpha = studentized_range.ppf(1 - alpha, n_models, np.inf) / np.sqrt(2)
    return float(q_alpha), float(q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets)))


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for position, index in enumerate(np.argsort(pvalues)):
        running = max(running, min(1.0, (len(pvalues) - position) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def paired_test(first, second):
    """Wilcoxon signed-rank (zero differences split between signs) and sign test over datasets."""
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    difference[np.abs(difference) <= TIE_TOLERANCE] = 0
    wins, losses = int((difference > 0).sum()), int((difference < 0).sum())
    if wins + losses == 0:
        wilcoxon_p = sign_p = 1.0
    else:
        wilcoxon_p = float(wilcoxon(difference, zero_method="zsplit").pvalue)
        sign_p = float(binomtest(wins, wins + losses, 0.5).pvalue)
    return {"mean_difference": float(difference.mean()), "wins": wins, "losses": losses,
            "ties": len(difference) - wins - losses, "wilcoxon_p": wilcoxon_p, "sign_p": sign_p}


def pairwise_family(matrix, pairs, family):
    rows = [{"family": family, "model": first, "reference": second, **paired_test(matrix[first], matrix[second])}
            for first, second in pairs]
    for row, adjusted in zip(rows, holm([row["wilcoxon_p"] for row in rows])):
        row["wilcoxon_p_holm"] = float(adjusted)
        row["significant"] = bool(adjusted < ALPHA)
    return rows


def compare(frame, output=Path("results")):
    actual = list(frame[["dataset", "model", "repetition", "fold"]].itertuples(index=False, name=None))
    expected = expected_records()
    if len(actual) != len(expected) or set(actual) != expected or not np.isfinite(frame.roc_auc).all():
        raise ValueError(f"Final inference requires all {len(expected)} unique and finite ROC-AUC records")
    matrix = auc_matrix(frame)
    matrix.to_csv(output / "auc_matrix.csv")
    rank_matrix(matrix).mean().rename("mean_rank").to_csv(output / "mean_ranks_all.csv")

    primary = matrix[PRIMARY_MODELS]
    ranks = rank_matrix(primary)
    ranks.to_csv(output / "ranks_by_dataset.csv")
    mean_ranks = ranks.mean()
    mean_ranks.rename("mean_rank").sort_values().to_csv(output / "mean_ranks.csv")
    statistic, pvalue = friedmanchisquare(*[primary[model].values for model in PRIMARY_MODELS])
    friedman = {"statistic": float(statistic), "pvalue": float(pvalue), "alpha": ALPHA, "metric": "roc_auc",
                "blocks": len(primary), "algorithms": len(PRIMARY_MODELS), "models": PRIMARY_MODELS,
                "reject_null": bool(pvalue < ALPHA), "nemenyi_executed": bool(pvalue < ALPHA)}
    if friedman["reject_null"]:
        pvalues = posthocs.posthoc_nemenyi_friedman(primary.to_numpy())
        pvalues.index = pvalues.columns = PRIMARY_MODELS
        pvalues.to_csv(output / "nemenyi_pvalues.csv")
        friedman["q_alpha"], friedman["critical_difference"] = critical_difference(len(PRIMARY_MODELS), len(primary))
    else:
        (output / "nemenyi_pvalues.csv").unlink(missing_ok=True)

    control_pairs = [(CONTROL_MODEL, model) for model in PRIMARY_MODELS if model != CONTROL_MODEL]
    pairwise = pairwise_family(matrix, control_pairs, "control") + pairwise_family(matrix, ABLATION_PAIRS, "ablation")
    pd.DataFrame(pairwise).to_csv(output / "pairwise_tests.csv", index=False)
    statistics = {"friedman": friedman, "mean_ranks": mean_ranks.sort_values().to_dict(), "pairwise": pairwise}
    (output / "statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    return statistics
