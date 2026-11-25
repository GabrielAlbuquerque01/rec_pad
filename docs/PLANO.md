# Plano e decisões anteriores à execução

Fonte normativa: AGENTS.md. Um experimento, CPU, semente 42, cinco folds externos e três internos; ROC-AUC como desfecho principal, alfa 0,05. Sem busca de hiperparâmetros.

1. Adquirir as dez bases UCI e preservar arquivos, metadados, variáveis e hashes. Conferir dimensões e classes. Auditar ausências, infinitos, constantes, duplicatas e dependências.
2. Implementar imputação por mediana, remoção de constantes e padronização ajustadas em cada treino; cinco classificadores especificados.
3. Implementar roteador K-Means, competências com probabilidades fora da amostra, pesos normalizados e fallbacks explícitos.
4. Validar alinhamento de classes, fusão, empates, isolamento das partições e cobertura interna.
5. Executar 50 folds externos, salvar previsões, partições, pesos, agrupamentos e 350 registros de métricas.
6. Agregar 70 resultados; Friedman sobre dez bases e Nemenyi somente se significativo. Produzir tabelas, gráficos e apresentação breve.

Dependências: NumPy/pandas para dados; ucimlrepo para aquisição; scikit-learn para transformações, modelos individuais e K-Means/PCA; SciPy/scikit-posthocs para inferência estatística; matplotlib para figuras. Testes com unittest da biblioteca padrão. Ambiente solicitado: Conda rec_pad.

Duplicatas de atributos, inclusive com rótulos conflitantes, serão preservadas e agrupadas nas divisões externas e internas. A identificação do grupo usa apenas igualdade de atributos originais, sem estimar transformações. Sem duplicatas, StratifiedKFold; com duplicatas, StratifiedGroupKFold. Não remover observações nem alterar rótulos para melhorar resultados.

Dependências sem identificador observável serão registradas como limitação. Na Climate, os índices de desenho/simulação não entram nos atributos; verificar sua semântica antes de decidir agrupamento.
