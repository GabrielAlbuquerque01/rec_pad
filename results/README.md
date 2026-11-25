# Artefatos do experimento

- `datasets.csv`: dimensões, classes, duplicatas, tratamentos e hashes dos dados originais.
- `metrics_by_fold.csv`: 350 registros; precisão indefinida é célula vazia, com causa na coluna `undefined`.
- `summary.csv`: 70 linhas; média, desvio amostral e contagem válida de cada métrica, incluindo contagens da matriz de confusão.
- `auc_matrix.csv`: médias de ROC-AUC, matriz 10 × 7 usada em Friedman.
- `statistics.json`, `mean_ranks.csv`, `ranks_by_dataset.csv`, `nemenyi_pvalues.csv`: estatística, rankings e pós-teste.
- `weights.csv`: competências e pesos por fold, modelo e região; `global` indica M6, e 0–2 indicam os clusters de M7.
- `clusters.csv`: tamanho e dispersões no treinamento externo. Identificadores de cluster só têm significado dentro de cada fold.
- `folds/<base>/<fold>_predictions.npz`: índices de treino/teste (base zero), grupos de todas as linhas, fold interno de cada linha de treino, rótulos, probabilidades OOF `(n_treino,5,2)`, scores externos `(n_teste,7,2)`, clusters, competências e pesos. Ordem dos modelos M1–M7, classes 0/1.
- `folds/`: também contém métricas, pesos, clusters e avisos por fold.
- `figures/`: quatro mapas de médias, rankings e 50 projeções PCA em `pca/`.
- `warnings.json`: avisos de treinamento por etapa; `execution.log`: progresso da execução.
- `environment.json`: versões e hashes dos módulos usados; `validation.json`: conferência posterior dos artefatos.
- `ANALISE.md`: resultados tabulados e interpretação estatística; complementos na [apresentação](../docs/APRESENTACAO.md).

`python main.py --report-only` regenera os relatórios a partir de registros completos. `python -m scripts.validate_results` verifica partições e recalcula os resultados usando as previsões salvas.
