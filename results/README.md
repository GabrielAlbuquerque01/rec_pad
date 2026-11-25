# Artefatos do experimento

Ordem dos modelos em todos os arquivos: M1–M10. Classes 0/1. Identificadores de cluster só têm significado
dentro de cada fold.

- `datasets.csv`: dimensões, classes, duplicatas, tratamentos e hashes dos dados originais.
- `metrics_by_fold.csv`: 1500 registros (base, modelo, repetição, fold). Métrica indefinida é célula vazia, com
  a causa na coluna `undefined`.
- `summary.csv`: 100 linhas com média, desvio amostral e contagem válida de cada métrica sobre os 15 folds
  externos, incluindo a matriz de confusão.
- `auc_matrix.csv`: ROC-AUC média, 10 bases × 10 modelos.
- Estatística:
  - `ranks_by_dataset.csv` e `mean_ranks.csv`: rankings da comparação principal (M1–M5, M8–M10);
  - `mean_ranks_all.csv`: rankings descritivos dos dez modelos;
  - `nemenyi_pvalues.csv`: pós-teste de Nemenyi;
  - `pairwise_tests.csv`: Wilcoxon, Holm e teste do sinal (famílias controle e ablação);
  - `statistics.json`: tudo isso reunido.
- `weights.csv`: pesos por fold.
  - M6 e M8: região `global`, com a acurácia OOF na coluna `competence`;
  - M7: clusters 0–2;
  - M9: clusters do K escolhido;
  - M10: média dos pesos sobre as instâncias de teste (`test_mean`).
- `clusters.csv`: tamanho, within e between dos roteadores de M7 (`router=M7`) e da versão 2 (`router=v2`, um
  bloco por K candidato; `selected` marca o K escolhido para M9).
- `selection.csv`: ROC-AUC nas dobras internas de cada candidato (K ou k, λ) de M9 e M10, e o escolhido.
- `diagnostics.csv`, por fold:
  - diversidade das OOF (oráculo, discordância, Q);
  - faixa dos pesos globais;
  - fração de decisões em que o local difere do global;
  - deslocamento dos pesos locais;
  - parâmetros escolhidos.
- `folds/<base>/r<repetição>_f<fold>.npz`:
  - índices de treino e teste (base zero), grupos e fold interno de cada linha de treino;
  - rótulos;
  - OOF brutas `(n_treino,5,2)` e calibradas em ajuste cruzado;
  - parâmetros de Platt;
  - scores de teste `(n_teste,10,2)` e probabilidades de teste calibradas;
  - clusters e pesos de M7;
  - pesos por instância de M8–M10, escolhas, clusters de M9 e vizinhos de M10.
- `figures/`:
  - mapas de médias (ROC-AUC, acurácia balanceada, acurácia, sensibilidade, especificidade);
  - `critical_difference.png`;
  - em `pca/`, projeções do roteador v2 com o K escolhido (repetição 1).
- `warnings.json`: avisos de treinamento por etapa. `execution.log`: progresso da execução.
- `environment.json`: versões, sementes e hashes dos módulos (calculados com quebras de linha LF).
- `validation.json`: conferência posterior dos artefatos.
- `ANALISE.md`: resultados tabulados, gerados automaticamente.

`python main.py --report-only` regenera relatórios e figuras a partir dos registros salvos.
`python -m scripts.validate_results` verifica as partições e recalcula calibração, pesos, fusões, métricas e
testes a partir das previsões salvas.
