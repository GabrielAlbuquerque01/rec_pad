"""Static scientific figures using only prescribed metrics and PCA."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def plot_clusters(router, name, fold, output):
    directory = output / "figures" / "pca"
    directory.mkdir(parents=True, exist_ok=True)
    pca = PCA(n_components=2, svd_solver="full")
    projection = pca.fit_transform(router.training_features_)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(projection[:, 0], projection[:, 1], c=router.kmeans.labels_, cmap="viridis", s=10, alpha=0.65)
    ax.set(title=f"{name}, fold {fold}: treino externo", xlabel="PC1", ylabel="PC2")
    fig.text(0.5, 0.01, f"Projeção 2D; variância exibida: {pca.explained_variance_ratio_.sum():.1%}. Não representa toda a geometria.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(directory / f"{name}_{fold}.png", dpi=140)
    plt.close(fig)


def present(frame, summary, matrix, ranks, statistics, output=Path("results")):
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for metric in ["roc_auc", "accuracy", "sensitivity", "specificity"]:
        values = summary.pivot(index="dataset", columns="model", values=f"{metric}_mean")
        fig, ax = plt.subplots(figsize=(9, 5))
        image = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(7), values.columns)
        ax.set_yticks(range(10), values.index)
        for i in range(10):
            for j in range(7):
                ax.text(j, i, f"{values.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if values.iloc[i, j] > 0.65 else "black")
        ax.set_title(f"{metric}: média dos cinco folds externos")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(figures / f"{metric}.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3))
    ordered = ranks.sort_values()
    ax.scatter(ordered.values, np.arange(7))
    ax.set_yticks(np.arange(7), ordered.index)
    ax.set(xlabel="Ranking médio de ROC-AUC (menor é melhor)", xlim=(0.8, 7.2))
    ax.invert_yaxis()
    if "critical_difference" in statistics:
        ax.set_title(f"Nemenyi: diferença crítica = {statistics['critical_difference']:.3f}")
    fig.tight_layout()
    fig.savefig(figures / "rankings.png", dpi=160)
    plt.close(fig)

    for artifact in ["weights", "clusters"]:
        tables = [pd.read_csv(path) for path in sorted((output / "folds").glob(f"*/*_{artifact}.csv"))]
        pd.concat(tables, ignore_index=True).to_csv(output / f"{artifact}.csv", index=False)
    warning_records = []
    for path in sorted((output / "folds").glob("*/*_warnings.json")):
        warning_records.extend(json.loads(path.read_text(encoding="utf-8")))
    (output / "warnings.json").write_text(json.dumps(warning_records, indent=2), encoding="utf-8")

    lines = ["# Resultados observados", "", "Um experimento 5×3, semente 42, sem otimização. M1: SVM linear; M2: SVM-RBF; M3: MLP; M4: GaussianNB; M5: KNN; M6: híbrido global; M7: híbrido local.", "",
             "350 avaliações de teste e 70 agregados. Desvios-padrão amostrais entre cinco folds (ddof=1), não intervalos de confiança. Veja summary.csv para todas as métricas e contagens válidas.", "",
             "| Base | M1 | M2 | M3 | M4 | M5 | M6 | M7 |", "|---|---|---|---|---|---|---|---|"]
    for dataset, row in matrix.iterrows():
        lines.append("| " + dataset + " | " + " | ".join(f"{v:.4f}" for v in row) + " |")
    lines += ["", "Tabela: ROC-AUC média por base.", "", f"Friedman: estatística {statistics['statistic']:.6f}; p={statistics['pvalue']:.8g}; alfa=0,05.", "",
              "| Modelo | Ranking médio |", "|---|---|"]
    lines += [f"| {model} | {value:.3f} |" for model, value in ordered.items()]
    if statistics["reject_null"]:
        lines += ["", f"H0 rejeitada. Nemenyi: q={statistics['q_alpha']:.6f} (amplitude studentizada / √2); CD={statistics['critical_difference']:.6f}. P-valores completos em nemenyi_pvalues.csv."]
        pvalues = pd.read_csv(output / "nemenyi_pvalues.csv", index_col=0)
        for model in matrix.columns[:-1]:
            difference = matrix.M7 - matrix[model]
            significant = pvalues.loc["M7", model] < 0.05
            lines += [f"- M7 versus {model}: ROC-AUC maior em {(difference > 0).sum()}/10 bases, menor em {(difference < 0).sum()}/10; p de Nemenyi={pvalues.loc['M7', model]:.6g}. " +
                      ("Diferença significativa nos rankings." if significant else "Sem evidência de diferença pelo pós-teste; isso não demonstra equivalência.")]
    else:
        lines += ["", "Friedman não rejeitou H0: Nemenyi não foi executado. Não se demonstrou equivalência nem superioridade."]
    lines += ["", "Comparação descritiva M7 − M6 por base (sem testes adicionais):", "", "| Base | Acurácia | Sensibilidade | Especificidade |", "|---|---|---|---|"]
    for dataset in matrix.index:
        subset = summary[summary.dataset == dataset].set_index("model")
        differences = [subset.loc["M7", f"{metric}_mean"] - subset.loc["M6", f"{metric}_mean"] for metric in ["accuracy", "sensitivity", "specificity"]]
        lines.append(f"| {dataset} | " + " | ".join(f"{v:+.4f}" for v in differences) + " |")
    clusters = pd.read_csv(output / "clusters.csv")
    lines += ["", "Agrupamentos no treino externo: within é a soma dos quadrados das distâncias aos centroides; between é a soma ponderada dos quadrados das distâncias dos centroides à média geral. Não são índices de seleção de K.", "", "| Base | Menor–maior cluster | Within médio por fold | Between médio por fold |", "|---|---|---|---|"]
    for dataset, subset in clusters.groupby("dataset"):
        totals = subset.groupby("fold")[["within_ss", "between_ss"]].sum().mean()
        lines.append(f"| {dataset} | {subset['size'].min()}–{subset['size'].max()} | {totals.within_ss:.2f} | {totals.between_ss:.2f} |")
    undefined = frame[frame.undefined.fillna("") != ""]
    lines += ["", f"Avisos capturados: {len(warning_records)} (warnings.json). Registros com métricas indefinidas: {len(undefined)}; causas em metrics_by_fold.csv. Não foram alterados parâmetros após observar resultados.", "",
              "Limitações: dez bases, uma semente e hiperparâmetros fixos limitam a generalização. Os folds são dependentes e não foram tratados como bases independentes. A projeção PCA perde dimensões; clusters não equivalem a classes. Probabilidades são estimativas sem garantia de calibração perfeita. Dependências não identificáveis nos dados públicos permanecem uma limitação (docs/BASES.md)."]
    (output / "ANALISE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
