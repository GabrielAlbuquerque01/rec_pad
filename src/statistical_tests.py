import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import numpy as np
import pandas as pd
import scikit_posthocs as posthocs
from scipy.stats import friedmanchisquare, rankdata, studentized_range

from .config import DATASETS, MODEL_IDS, OUTER_FOLDS


def compare(frame, output=Path("results")):
    expected = {(dataset, model, fold) for dataset in DATASETS for model in MODEL_IDS
                for fold in range(1, OUTER_FOLDS + 1)}
    actual = list(frame[["dataset", "model", "fold"]].itertuples(index=False, name=None))
    if len(actual) != len(expected) or set(actual) != expected or not np.isfinite(frame.roc_auc).all():
        raise ValueError("Final inference requires all 350 unique and finite ROC-AUC records")
    matrix = frame.groupby(["dataset", "model"]).roc_auc.mean().unstack().loc[list(DATASETS), MODEL_IDS]
    matrix.to_csv(output / "auc_matrix.csv")
    ranks = pd.DataFrame(rankdata(-matrix.to_numpy(), axis=1, method="average"),
                         index=matrix.index, columns=matrix.columns)
    ranks.to_csv(output / "ranks_by_dataset.csv")
    ranks.mean().rename("mean_rank").sort_values().to_csv(output / "mean_ranks.csv")
    statistic, pvalue = friedmanchisquare(*[matrix[model].values for model in MODEL_IDS])
    result = {"statistic": float(statistic), "pvalue": float(pvalue), "alpha": 0.05,
              "blocks": len(matrix), "algorithms": len(MODEL_IDS), "metric": "roc_auc",
              "reject_null": bool(pvalue < 0.05), "nemenyi_executed": bool(pvalue < 0.05)}
    if pvalue < 0.05:
        pvalues = posthocs.posthoc_nemenyi_friedman(matrix.to_numpy())
        pvalues.index = pvalues.columns = MODEL_IDS
        pvalues.to_csv(output / "nemenyi_pvalues.csv")
        # Nemenyi q = Tukey studentized range / sqrt(2), infinite df.
        q_alpha = studentized_range.ppf(0.95, len(MODEL_IDS), np.inf) / np.sqrt(2)
        result["q_alpha"] = float(q_alpha)
        result["critical_difference"] = float(q_alpha * np.sqrt(7 * 8 / (6 * 10)))
    (output / "statistics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return matrix, ranks.mean(), result
