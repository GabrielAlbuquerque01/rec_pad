"""Exploratory recombination of the version 1 predictions; no model is retrained.

Run as python -m scripts.explore_combiners <folds directory> [output.csv].

The input is the results/folds directory of commit b1b9b49 (one partition, seed 42,
files <dataset>/<fold>_predictions.npz with models M1-M7), which can be extracted with
git archive b1b9b49 results/folds | tar -x -C <destination>.

Every variant uses only out-of-fold (OOF) information from the external training
set to build its weights, so the external test set is never used for fitting.
The results are exploratory: the external folds were already observed, so a
confirmatory comparison requires a new run with different partitions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.config import DATASETS, N_CLUSTERS, OUTER_FOLDS

BASE_MODELS = 5
NAIVE_BAYES = 3
# Pseudo-count that pulls local accuracy toward global accuracy in small regions.
SHRINKAGE = 20


def normalize(values, fallback):
    values = np.clip(np.asarray(values, dtype=float), 0, None)
    return values / values.sum() if values.sum() > 0 else fallback.copy()


def majority_rate(labels):
    return max(labels.mean(), 1 - labels.mean()) if len(labels) else np.nan


def logit(probabilities):
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def calibrate(oof, labels, test):
    """Platt scaling of each base model, fitted on OOF scores only."""
    calibrated = np.empty((test.shape[0], BASE_MODELS))
    for model in range(BASE_MODELS):
        scaler = LogisticRegression(C=1e6).fit(logit(oof[:, model, 1])[:, None], labels)
        calibrated[:, model] = scaler.predict_proba(logit(test[:, model, 1])[:, None])[:, 1]
    return calibrated


def variants(saved):
    oof, labels, train_clusters = saved["oof"], saved["y_train"], saved["train_clusters"]
    test, test_clusters = saved["scores"][:, :BASE_MODELS], saved["test_clusters"]
    positive = test[:, :, 1]
    correct = oof.argmax(axis=2) == labels[:, None]
    global_accuracy = correct.mean(axis=0)
    uniform = np.full(BASE_MODELS, 1 / BASE_MODELS)
    # Competence above the majority-class baseline: a model at that level adds no information.
    global_excess = normalize(global_accuracy - majority_rate(labels), uniform)

    scores = {
        "M6": saved["scores"][:, 5, 1],
        "M7": saved["scores"][:, 6, 1],
        "MEDIA_SIMPLES": positive.mean(axis=1),
        "G_EXCESSO": positive @ global_excess,
        "G_CAL_EXCESSO": calibrate(oof, labels, test) @ global_excess,
    }
    local_excess = np.zeros(len(test))
    selection = np.zeros(len(test))
    top3 = np.zeros(len(test))
    for cluster in range(N_CLUSTERS):
        members, queries = train_clusters == cluster, test_clusters == cluster
        if not queries.any():
            continue
        shrunk = (correct[members].sum(axis=0) + SHRINKAGE * global_accuracy) / (members.sum() + SHRINKAGE)
        baseline = majority_rate(labels[members]) if members.any() else majority_rate(labels)
        weights = normalize(shrunk - baseline, global_excess)
        local_excess[queries] = positive[queries] @ weights
        order = np.lexsort((-global_accuracy, -shrunk))
        selection[queries] = positive[queries, order[0]]
        top3[queries] = positive[queries][:, order[:3]].mean(axis=1)
    scores["L_EXCESSO_SUAVIZADO"] = local_excess
    scores["L_SELECAO_1"] = selection
    scores["L_SELECAO_TOP3"] = top3
    return scores


def diagnostics(saved):
    test, labels = saved["scores"][:, :BASE_MODELS], saved["y_test"]
    predictions = test.argmax(axis=2)
    correct = predictions == labels[:, None]
    pairs = [(i, j) for i in range(BASE_MODELS) for j in range(i + 1, BASE_MODELS)]
    m6, m7 = saved["scores"][:, 5], saved["scores"][:, 6]
    others = np.delete(predictions, NAIVE_BAYES, axis=1)
    others_majority = others.sum(axis=1) * 2 > others.shape[1]
    nb_alone = predictions[:, NAIVE_BAYES] != others_majority
    return {
        "faixa_pesos_globais": np.ptp(saved["global_weights"]),
        "max_dif_local_global": np.abs(saved["local_weights"] - saved["global_weights"]).max(),
        "m7_muda_decisao_m6": np.mean(m6.argmax(axis=1) != m7.argmax(axis=1)),
        "oraculo_acc": correct.any(axis=1).mean(),
        "melhor_individual_acc": correct.mean(axis=0).max(),
        "discordancia_media": np.mean([np.mean(predictions[:, i] != predictions[:, j]) for i, j in pairs]),
        "nb_prob_extrema": np.mean(test[:, NAIVE_BAYES].max(axis=1) > 0.99),
        "m6_segue_nb_contra_maioria": np.mean(nb_alone & (m6.argmax(axis=1) == predictions[:, NAIVE_BAYES])),
        "menor_cluster_treino": np.bincount(saved["train_clusters"], minlength=N_CLUSTERS).min(),
    }


def main():
    folds = Path(sys.argv[1])
    records, diagnostic_records = [], []
    for name in DATASETS:
        for fold in range(1, OUTER_FOLDS + 1):
            saved = np.load(folds / name / f"{fold}_predictions.npz")
            labels = saved["y_test"]
            for variant, score in variants(saved).items():
                # Threshold 0.5 on the positive score matches argmax with ties resolved to class 0.
                predicted = (score > 0.5).astype(int)
                records.append({"dataset": name, "fold": fold, "variant": variant,
                                "roc_auc": roc_auc_score(labels, score),
                                "accuracy": float((predicted == labels).mean()),
                                "balanced_accuracy": balanced_accuracy_score(labels, predicted)})
            diagnostic_records.append({"dataset": name, **diagnostics(saved)})
    frame = pd.DataFrame(records)
    matrix = frame.groupby(["dataset", "variant"]).roc_auc.mean().unstack().loc[list(DATASETS)]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("ROC-AUC média por base (exploratório):")
    print(matrix.round(4).to_string())
    ranks = pd.DataFrame(rankdata(-matrix.to_numpy(), axis=1), index=matrix.index, columns=matrix.columns)
    print("\nRanking médio entre as variantes:")
    print(ranks.mean().sort_values().round(2).to_string())
    print("\nComparação pareada por base com M6 (Wilcoxon exploratório, sem correção):")
    for variant in matrix.columns.drop("M6"):
        difference = matrix[variant] - matrix["M6"]
        nonzero = difference[difference.abs() > 1e-12]
        pvalue = wilcoxon(nonzero).pvalue if len(nonzero) else np.nan
        print(f"{variant:22s} dif. média={difference.mean():+.4f} vitórias={int((difference > 1e-12).sum())} "
              f"derrotas={int((difference < -1e-12).sum())} p={pvalue:.4f}")
    for metric in ["accuracy", "balanced_accuracy"]:
        means = frame.groupby(["dataset", "variant"])[metric].mean().unstack().loc[list(DATASETS)]
        print(f"\n{metric}: média por base e diferença média em relação a M6")
        print(means.round(4).to_string())
        print(means.sub(means["M6"], axis=0).mean().round(4).to_string())
    summary = pd.DataFrame(diagnostic_records).groupby("dataset").mean().loc[list(DATASETS)]
    print("\nDiagnósticos (média dos folds externos):")
    print(summary.round(4).to_string())
    if len(sys.argv) > 2:
        frame.to_csv(sys.argv[2], index=False)


if __name__ == "__main__":
    main()
