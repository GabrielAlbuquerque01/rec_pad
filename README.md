# Classificação por competência local

Experimento em dez bases UCI: SVM linear, SVM-RBF, MLP, Gaussian Naive Bayes e KNN, comparados com dois híbridos autorais. O híbrido global pondera probabilidades pela acurácia de validação interna; o local utiliza a acurácia na região definida por K-Means. Requisitos em [docs/AGENTS.md](docs/AGENTS.md).

## Instalação e execução

```powershell
conda activate rec_pad
python -m pip install -r requirements.txt
python main.py --acquire-only
python -m unittest discover -s tests -v
python main.py
python -m scripts.validate_results
```

CPU; uma execução 5 folds externos × 3 internos, semente 42, sem ajuste de hiperparâmetros. A primeira aquisição precisa de internet; os dados completos em cache permitem executar offline. `python main.py --report-only` recompõe tabelas e gráficos a partir dos 350 registros já existentes. Uma nova execução de `main.py` recalcula os experimentos e sobrescreve seus artefatos. Python usado: 3.12.16; `requirements-lock.txt` registra também as dependências transitivas e pode ser instalado com `pip install -r requirements-lock.txt`.

## Estrutura

- `src/`: aquisição, pré-processamento, modelos, híbridos, avaliação, estatística e figuras.
- `scripts/acquire.py`: download e auditoria, sem experimentos.
- `tests/`: testes de correção e isolamento das partições.
- `data/raw/`: dados UCI e metadados em cache, ignorados pelo Git.
- `results/`: auditoria, métricas por fold, médias/desvios, pesos, clusters, previsões, gráficos e análise.
- `docs/`: [plano](docs/PLANO.md), [fontes e tratamentos](docs/BASES.md), [formulação e arquitetura](docs/METODO.md) e [apresentação](docs/APRESENTACAO.md).

Friedman usa ROC-AUC média, dez bases como blocos e alfa 5%; Nemenyi é executado somente se Friedman rejeitar H0. As previsões individuais e os híbridos compartilham as mesmas probabilidades. Não se presume superioridade do híbrido.

## Resultado desta execução

Dez bases concluídas: 350 registros, 70 agregados, nove testes aprovados e validação dos artefatos concluída. Friedman p=0,0002465; híbridos global e local tiveram ranking médio 2,35 e Nemenyi p=1 entre si. Não houve evidência de vantagem local sobre global. A [análise](results/ANALISE.md) e a [apresentação](docs/APRESENTACAO.md) detalham comparações, desempenho por classe e limitações, incluindo avisos de convergência da MLP.
