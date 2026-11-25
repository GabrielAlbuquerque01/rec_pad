# Classificação por competência local

Experimento em dez bases UCI. Cinco classificadores individuais (SVM linear, SVM-RBF, MLP, Gaussian Naive
Bayes e KNN) são comparados com cinco híbridos autorais:

- M6 e M7: formulação original (pesos proporcionais à acurácia, global ou por cluster do K-Means);
- M8, M9 e M10: versão 2. As probabilidades são calibradas e o peso de cada modelo vem da acurácia acima da
  classe majoritária: no treino inteiro (M8), por cluster do K-Means (M9) ou na vizinhança da instância (M10).
  K, k e a suavização são escolhidos na validação interna.

Especificação em [docs/AGENTS.md](docs/AGENTS.md).

## Instalação e execução

```powershell
conda create -n rec_pad python=3.12
conda activate rec_pad
python -m pip install -r requirements-lock.txt
python main.py --acquire-only
python -m unittest discover -s tests -v
python main.py
python -m scripts.validate_results
```

- **Execução:** CPU, 3 repetições (sementes 11, 22 e 33) de 5 folds externos × 3 internos, sem ajuste de
  hiperparâmetros dos modelos base. Leva cerca de 8 minutos.
- **Dados:** a primeira aquisição precisa de internet; com os dados em cache, roda offline.
- **Relatórios:** `python main.py --report-only` recompõe tabelas e figuras a partir dos registros salvos. Uma
  nova execução de `main.py` recalcula tudo e sobrescreve os artefatos.
- **Ambiente:** Python 3.12.16. Se houver pacotes instalados no diretório do usuário, defina
  `PYTHONNOUSERSITE=1` para isolar o ambiente.

## Estrutura

- `src/`: aquisição, pré-processamento, modelos, híbridos, avaliação, estatística e figuras. Parâmetros em
  `src/config.py`.
- `scripts/`:
  - `acquire.py`: download e auditoria;
  - `validate_results.py`: conferência independente dos artefatos;
  - `explore_combiners.py`: reanálise exploratória da versão 1.
- `tests/`: correção dos combinadores, calibração, isolamento das partições e estatística.
- `slides/`: gerador da apresentação no template do PPGEC; o `.pptx` editável sai em `slides/exports/` ([instruções](slides/README.md)).
- `data/raw/`: dados UCI em cache, ignorados pelo Git.
- `results/`: métricas, pesos, escolhas, clusters, diagnósticos, previsões, figuras e
  [análise](results/ANALISE.md) ([guia dos artefatos](results/README.md)).
- `docs/`:
  - [plano e desvios](docs/PLANO.md);
  - [fontes e tratamentos](docs/BASES.md);
  - [formulação e arquitetura](docs/METODO.md);
  - [estudo de fragilidades](docs/ESTUDO_MELHORIAS.md);
  - [apresentação](docs/APRESENTACAO.md) e [guia do diagrama para o Lucidchart](docs/DIAGRAMA_ARQUITETURA.md).

## Resultado desta execução

São 1500 registros e 100 agregados; os testes passaram e a validação dos artefatos foi concluída.

**Comparação principal** (Friedman na ROC-AUC, 10 bases × 8 modelos): χ² = 37,46, p = 3,8·10⁻⁶.

| Modelos | Ranking médio |
|---|---|
| M8 Global v2 | 2,55 |
| M9 Local K-Means v2 | 2,75 |
| M10 Local k-NN v2 | 2,75 |
| M3 MLP | 4,40 |
| M2 SVM-RBF | 4,75 |
| M1 SVM linear | 4,90 |
| M4 GaussianNB | 6,80 |
| M5 KNN | 7,10 |

- **Nemenyi (CD = 3,32):** separa os três híbridos de GaussianNB e KNN, mas não de SVM linear, SVM-RBF e MLP.
- **M9 contra as SVMs e a MLP (Wilcoxon):** vence 8 de 10 bases contra cada uma (p bruto entre 0,02 e 0,04),
  mas nenhuma diferença sobrevive à correção de Holm.
- **A correção da competência funciona.** M8 supera M6 (a fórmula da aula) em 9 de 10 bases, com 1 empate:
  +0,0058 de AUC, p de Holm = 0,012.
- **A ponderação local não melhorou sobre a global corrigida.** M9 × M8 terminou com 4 vitórias, 5 derrotas
  e 1 empate; M10 × M8, com 3, 6 e 1. Ao contrário da versão 1, o mecanismo local agora atua: os pesos locais
  se deslocam até 28% em relação aos globais, e a decisão muda em até 4% das instâncias. Mesmo assim, não há
  ganho, e a validação interna escolhe K = 1 (global) em Banknote e Spambase na maior parte dos folds. Nestas
  bases, a conclusão é que não há evidência de vantagem local. Isso não demonstra equivalência.

A primeira rodada confirmatória usou early stopping na MLP, o que causou subajuste. A MLP voltou à
configuração da versão 1, e o desvio está registrado em [docs/PLANO.md](docs/PLANO.md).
