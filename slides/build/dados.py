# -*- coding: utf-8 -*-
"""Leitura dos resultados do experimento (pasta results/ do repositorio).

Os slides tiram numeros daqui em vez de copia-los a mao, para que a apresentacao
acompanhe uma nova execucao do experimento.
"""
import json
import pathlib
from functools import lru_cache

import pandas as pd

RESULTS = pathlib.Path(__file__).resolve().parents[2] / "results"
TIE = 1e-12

BASES = ["sonar", "ionosphere", "banknote", "blood", "climate", "qsar", "raisin", "rice",
         "spambase", "retinopathy"]
NOME_BASE = {"sonar": "Sonar", "ionosphere": "Ionosphere", "banknote": "Banknote",
             "blood": "Blood", "climate": "Climate", "qsar": "QSAR", "raisin": "Raisin",
             "rice": "Rice", "spambase": "Spambase", "retinopathy": "Retinopathy"}
MODELOS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"]
NOME_MODELO = {"M1": "SVM linear", "M2": "SVM-RBF", "M3": "MLP", "M4": "Naive Bayes",
               "M5": "KNN", "M6": "Global (aula)", "M7": "Local original",
               "M8": "Global v2", "M9": "Local K-Means v2", "M10": "Local k-NN v2"}


@lru_cache
def summary():
    return pd.read_csv(RESULTS / "summary.csv").set_index(["dataset", "model"])


@lru_cache
def auc():
    return pd.read_csv(RESULTS / "auc_matrix.csv", index_col=0).loc[BASES, MODELOS]


@lru_cache
def estatistica():
    return json.loads((RESULTS / "statistics.json").read_text(encoding="utf-8"))


@lru_cache
def diagnosticos():
    return pd.read_csv(RESULTS / "diagnostics.csv")


@lru_cache
def pesos():
    return pd.read_csv(RESULTS / "weights.csv")


@lru_cache
def bases():
    return pd.read_csv(RESULTS / "datasets.csv").set_index("dataset").loc[BASES]


def par(modelo, referencia):
    """Linha dos testes pareados para o par pedido (familia controle ou ablacao)."""
    for row in estatistica()["pairwise"]:
        if row["model"] == modelo and row["reference"] == referencia:
            return row
    raise KeyError((modelo, referencia))


def resultado_por_base(modelo, referencia):
    """+1, -1 ou 0 por base, comparando a ROC-AUC media dos dois modelos."""
    diff = auc()[modelo] - auc()[referencia]
    return [0 if abs(d) <= TIE else (1 if d > 0 else -1) for d in diff]


def pesos_globais(base, hibrido):
    w = pesos()
    sel = w[(w.dataset == base) & (w.hybrid == hibrido) & (w.region == "global")]
    return sel.groupby("model").weight.mean().loc[MODELOS[:5]]


@lru_cache
def probabilidade_extrema(base, modelo_indice, limite=0.99):
    """Fracao media das instancias de teste em que o modelo atribui probabilidade acima do
    limite a uma classe, antes e depois da calibracao."""
    import numpy as np
    antes, depois = [], []
    for arquivo in sorted((RESULTS / "folds" / base).glob("*.npz")):
        salvo = np.load(arquivo)
        antes.append(np.mean(salvo["scores"][:, modelo_indice].max(axis=1) > limite))
        depois.append(np.mean(salvo["calibrated_test"][:, modelo_indice].max(axis=1) > limite))
    return float(np.mean(antes)), float(np.mean(depois))


def fracao_k_global():
    """Fracao dos folds externos em que a validacao interna escolheu K = 1 para M9."""
    return diagnosticos().groupby("dataset").m9_k.agg(lambda s: float((s == 1).mean())).loc[BASES]


def media(base, modelo, metrica):
    return float(summary().loc[(base, modelo), f"{metrica}_mean"])


def desvio(base, modelo, metrica):
    return float(summary().loc[(base, modelo), f"{metrica}_std"])
