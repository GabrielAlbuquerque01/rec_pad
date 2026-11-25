"""Static figures and the generated results summary (results/ANALISE.md)."""
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

from .config import (ALPHA, BASE_MODELS, CLUSTER_CANDIDATES, CONTROL_MODEL, DATASETS, INNER_FOLDS, MODEL_IDS,
                     MODEL_NAMES, OUTER_FOLDS, PRIMARY_MODELS, REPETITION_SEEDS)


def plot_clusters(points, labels, name, fold, n_clusters, output):
    directory = output / "figures" / "pca"
    directory.mkdir(parents=True, exist_ok=True)
    pca = PCA(n_components=2, svd_solver="full")
    projection = pca.fit_transform(points)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(projection[:, 0], projection[:, 1], c=labels, cmap="viridis", s=10, alpha=0.65)
    ax.set(title=f"{name}, fold {fold}: roteador v2, K={n_clusters}", xlabel="PC1", ylabel="PC2")
    fig.text(0.5, 0.01, f"Projeção 2D; variância exibida: {pca.explained_variance_ratio_.sum():.1%}. "
             "Não representa toda a geometria.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(directory / f"{name}_{fold}.png", dpi=140)
    plt.close(fig)


def plot_heatmap(summary, metric, path):
    values = summary.pivot(index="dataset", columns="model", values=f"{metric}_mean").loc[list(DATASETS), MODEL_IDS]
    fig, ax = plt.subplots(figsize=(11, 5))
    image = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(MODEL_IDS)), MODEL_IDS)
    ax.set_yticks(range(len(values)), values.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values.iloc[i, j]
            ax.text(j, i, "—" if np.isnan(value) else f"{value:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if value > 0.65 else "black")
    ax.axvline(len(BASE_MODELS) - 0.5, color="black", linewidth=1)
    ax.set_title(f"{metric}: média de {len(REPETITION_SEEDS) * OUTER_FOLDS} folds externos")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_critical_difference(mean_ranks, critical_difference, path):
    """Demšar-style diagram: methods joined by a bar are not distinguished by Nemenyi."""
    ordered = mean_ranks.sort_values()
    names, values = list(ordered.index), ordered.to_numpy()
    count = len(values)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0.4, count + 0.6)
    ax.set_ylim(-(count // 2 + 2) * 0.32, 0.6)
    ax.axis("off")
    ax.plot([1, count], [0, 0], color="black", linewidth=1)
    for tick in range(1, count + 1):
        ax.plot([tick, tick], [0, 0.08], color="black", linewidth=1)
        ax.text(tick, 0.14, str(tick), ha="center", va="bottom", fontsize=9)
    if critical_difference is not None:
        ax.plot([1, 1 + critical_difference], [0.45, 0.45], color="black", linewidth=1.5)
        ax.text(1 + critical_difference / 2, 0.5, f"CD = {critical_difference:.2f}", ha="center", va="bottom", fontsize=9)
    half = (count + 1) // 2
    for position, (name, value) in enumerate(zip(names, values)):
        left = position < half
        level = position if left else count - 1 - position
        y = -(level + 1) * 0.32 - 0.25
        x_text = 0.45 if left else count + 0.55
        ax.plot([value, value, x_text], [0, y, y], color="black", linewidth=0.8)
        label = f"{name} {MODEL_NAMES[name]} ({value:.2f})"
        ax.text(x_text, y + 0.03, label, ha="left" if left else "right", va="bottom", fontsize=8)
    if critical_difference is not None:
        groups = []
        for start in range(count):
            end = start
            while end + 1 < count and values[end + 1] - values[start] < critical_difference:
                end += 1
            if end > start and not any(s <= start and end <= e for s, e in groups):
                groups.append((start, end))
        for level, (start, end) in enumerate(groups):
            y = -0.1 - level * 0.07
            ax.plot([values[start] - 0.04, values[end] + 0.04], [y, y], color="tab:red", linewidth=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def markdown_table(frame, float_format="{:.4f}"):
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    lines = [header, "|" + "---|" * len(frame.columns)]
    for _, row in frame.iterrows():
        cells = [float_format.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v)
                 else ("—" if isinstance(v, (float, np.floating)) else str(v)) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def distribution(series):
    counts = series.value_counts().sort_index()
    return ", ".join(f"{int(k) if float(k).is_integer() else k}: {v}" for k, v in counts.items())


def present(frame, summary, statistics, output=Path("results")):
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for metric in ["roc_auc", "balanced_accuracy", "accuracy", "sensitivity", "specificity"]:
        plot_heatmap(summary, metric, figures / f"{metric}.png")
    friedman = statistics["friedman"]
    mean_ranks = pd.Series(statistics["mean_ranks"])
    plot_critical_difference(mean_ranks, friedman.get("critical_difference"), figures / "critical_difference.png")

    matrix = pd.read_csv(output / "auc_matrix.csv", index_col=0)
    pairwise = pd.DataFrame(statistics["pairwise"])
    diagnostics = pd.read_csv(output / "diagnostics.csv")
    clusters = pd.read_csv(output / "clusters.csv")
    warnings = json.loads((output / "warnings.json").read_text(encoding="utf-8"))
    n_folds = len(REPETITION_SEEDS) * OUTER_FOLDS

    legend = "; ".join(f"{m}: {MODEL_NAMES[m]}" for m in MODEL_IDS)
    lines = ["# Resultados observados", "",
             f"Protocolo: {len(REPETITION_SEEDS)} repetições (sementes {', '.join(map(str, REPETITION_SEEDS))}) de "
             f"{OUTER_FOLDS} folds externos × {INNER_FOLDS} internos; {len(frame)} registros e {len(summary)} agregados. "
             f"Médias e desvios usam os {n_folds} folds externos de cada base; os folds de repetições diferentes "
             "se sobrepõem e não são amostras independentes.", "", legend + ".", "",
             f"## ROC-AUC média ({n_folds} folds)", ""]
    lines += markdown_table(matrix.reset_index().rename(columns={"index": "Base", "dataset": "Base"}))
    lines += ["", "Médias, desvios e contagens de todas as métricas em summary.csv; mapas em figures/.", "",
              "## Friedman e Nemenyi (comparação principal)", "",
              f"Modelos: {', '.join(PRIMARY_MODELS)}; {friedman['blocks']} bases como blocos; α = {ALPHA}.",
              f"Friedman χ² = {friedman['statistic']:.4f}; p = {friedman['pvalue']:.6g}.", ""]
    lines += ["| Modelo | Ranking médio |", "|---|---|"]
    lines += [f"| {m} {MODEL_NAMES[m]} | {v:.2f} |" for m, v in mean_ranks.items()]
    if friedman["reject_null"]:
        nemenyi = pd.read_csv(output / "nemenyi_pvalues.csv", index_col=0)
        significant = [f"{a} × {b} (p = {nemenyi.loc[a, b]:.4g})" for i, a in enumerate(PRIMARY_MODELS)
                       for b in PRIMARY_MODELS[i + 1:] if nemenyi.loc[a, b] < ALPHA]
        lines += ["", f"H0 rejeitada. Nemenyi: q = {friedman['q_alpha']:.4f}; CD = {friedman['critical_difference']:.4f} "
                  "(figures/critical_difference.png).", "",
                  "Pares com diferença significativa: " + ("; ".join(significant) if significant else "nenhum") + "."]
    else:
        lines += ["", "Friedman não rejeitou H0; Nemenyi não foi executado. Isso não demonstra equivalência."]

    def pairwise_lines(family, title):
        subset = pairwise[pairwise.family == family]
        rows = ["", title, "", "| Modelo | Referência | Δ AUC média | V / D / E | p Wilcoxon | p Holm | p sinal |",
                "|---|---|---|---|---|---|---|"]
        for _, r in subset.iterrows():
            rows.append(f"| {r.model} | {r.reference} | {r.mean_difference:+.4f} | {r.wins} / {r.losses} / {r.ties} | "
                        f"{r.wilcoxon_p:.4g} | {r.wilcoxon_p_holm:.4g}{' *' if r.significant else ''} | {r.sign_p:.4g} |")
        return rows

    lines += pairwise_lines("control", f"## {CONTROL_MODEL} contra cada modelo da comparação principal")
    lines += pairwise_lines("ablation", "## Ablação: efeito de cada mudança")
    lines += ["", "Wilcoxon pareado por base (diferenças nulas divididas entre os sinais), correção de Holm dentro de "
              "cada família; * indica p ajustado < α. V/D/E: vitórias, derrotas e empates do primeiro modelo."]

    selection = diagnostics.groupby("dataset").agg(
        k9=("m9_k", distribution), s9=("m9_shrinkage", distribution),
        k10=("m10_k", distribution), s10=("m10_shrinkage", distribution)).loc[list(DATASETS)]
    lines += ["", "## Escolhas feitas na validação interna", "",
              f"Contagem de folds externos ({n_folds} por base) por valor escolhido. K = 1 em M9 equivale à ponderação global.",
              "", "| Base | M9: K | M9: λ | M10: k | M10: λ |", "|---|---|---|---|---|"]
    lines += [f"| {d} | {r.k9} | {r.s9} | {r.k10} | {r.s10} |" for d, r in selection.iterrows()]

    effect = diagnostics.groupby("dataset")[["m6_weight_range", "m8_weight_range", "m7_changes_m6", "m9_changes_m8",
                                             "m10_changes_m8", "m9_weight_shift", "m10_weight_shift"]].mean()
    lines += ["", "## Quanto os pesos locais se afastam dos globais", "",
              "Faixa dos pesos globais (máximo − mínimo); fração das instâncias de teste em que a decisão local difere "
              "da global correspondente; deslocamento médio dos pesos locais em relação aos globais (metade da soma "
              "das diferenças absolutas, de 0 a 1).", ""]
    lines += markdown_table(effect.loc[list(DATASETS)].reset_index().rename(columns={
        "dataset": "Base", "m6_weight_range": "faixa M6", "m8_weight_range": "faixa M8",
        "m7_changes_m6": "M7≠M6", "m9_changes_m8": "M9≠M8", "m10_changes_m8": "M10≠M8",
        "m9_weight_shift": "desloc. M9", "m10_weight_shift": "desloc. M10"}))

    diversity = diagnostics.groupby("dataset")[["oracle_accuracy", "best_single_accuracy", "mean_disagreement",
                                                "mean_q"]].mean().loc[list(DATASETS)]
    lines += ["", "## Diversidade dos modelos base (previsões OOF)", "",
              "Oráculo: fração em que algum modelo acerta. Q de Yule médio entre pares: próximo de 1 indica erros "
              "coincidentes; próximo de 0, independentes.", ""]
    lines += markdown_table(diversity.reset_index().rename(columns={
        "dataset": "Base", "oracle_accuracy": "oráculo", "best_single_accuracy": "melhor individual",
        "mean_disagreement": "discordância", "mean_q": "Q médio"}))

    v2 = clusters[clusters.router == "v2"]
    per_k = v2.groupby(["dataset", "repetition", "fold", "k"]).agg(
        within=("within_ss", "sum"), between=("between_ss", "sum"), smallest=("size", "min")).reset_index()
    per_k["between_share"] = per_k.between / (per_k.between + per_k.within)
    share = per_k.groupby(["dataset", "k"]).between_share.mean().unstack().loc[list(DATASETS)]
    smallest = per_k.groupby(["dataset", "k"]).smallest.min().unstack().loc[list(DATASETS)]
    original = clusters[clusters.router == "M7"].groupby("dataset")["size"].agg(["min", "max"]).loc[list(DATASETS)]
    lines += ["", "## Agrupamentos", "",
              "Roteador v2 (Yeo-Johnson + padronização): fração between/(within + between) média por K e, entre "
              "parênteses, o menor cluster observado. Roteador de M7 (padronização, K = 3): menor e maior cluster.", "",
              "| Base | " + " | ".join(f"K={k}" for k in CLUSTER_CANDIDATES if k > 1) + " | M7: menor–maior |",
              "|---|" + "---|" * (len(CLUSTER_CANDIDATES) - 1) + "---|"]
    for dataset in DATASETS:
        cells = [f"{share.loc[dataset, k]:.2f} ({int(smallest.loc[dataset, k])})" for k in CLUSTER_CANDIDATES if k > 1]
        lines.append(f"| {dataset} | " + " | ".join(cells) + f" | {original.loc[dataset, 'min']}–{original.loc[dataset, 'max']} |")

    undefined = frame[frame.undefined.fillna("") != ""]
    categories = pd.Series([w["category"] for w in warnings]).value_counts().to_dict() if warnings else {}
    lines += ["", "## Avisos e métricas indefinidas", "",
              f"Avisos capturados: {len(warnings)} {categories} (warnings.json). Registros com métricas indefinidas: "
              f"{len(undefined)}; causas na coluna undefined de metrics_by_fold.csv.", "",
              "## Limitações", "",
              "Dez bases limitam o poder dos testes entre bases. As repetições reduzem a variância de cada média, "
              "mas não criam novas bases. Hiperparâmetros dos modelos base são fixos. Calibração de Platt é uma "
              "aproximação paramétrica. A PCA perde dimensões; clusters não equivalem a classes. Dependências não "
              "identificáveis nos dados públicos permanecem uma limitação (docs/BASES.md)."]
    (output / "ANALISE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
