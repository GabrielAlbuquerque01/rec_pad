# Sistema híbrido de classificação por competência local

## 1. Objetivo

Desenvolver em Python um sistema híbrido que combine classificadores supervisionados com diferentes características de aprendizagem. Regiões do espaço de atributos (clusters do K-Means ou vizinhanças) são identificadas sem rótulos, e o desempenho dos classificadores em cada região determina sua contribuição para a decisão final.

Investigar se a ponderação local melhora o desempenho em relação aos classificadores individuais e à ponderação global. Não presumir que o híbrido será superior: verificar a hipótese experimentalmente.

O agente deve implementar o projeto completo, obter as bases, executar os experimentos, analisar os resultados e preparar um diagrama da arquitetura. Não é necessário elaborar relatório formal.

Histórico: a versão 1 (commit `b1b9b49`) ponderava pela acurácia bruta, e os pesos ficaram quase uniformes; o diagnóstico está em [ESTUDO_MELHORIAS.md](ESTUDO_MELHORIAS.md). Esta especificação descreve a versão 2, que mantém os híbridos originais como ablação. As decisões fixadas antes da execução e os desvios ficam em [PLANO.md](PLANO.md).

## 2. Bases de dados

Utilizar as dez bases públicas de classificação binária abaixo, disponíveis no UCI. A aquisição pode ser feita com `ucimlrepo`. Conferir atributos, alvo, dimensões e significado das classes na fonte antes de utilizar cada base.

| Base | ID UCI | Amostras | Atributos |
|---|---:|---:|---:|
| Sonar | [151](https://archive.ics.uci.edu/dataset/151) | 208 | 60 |
| Ionosphere | [52](https://archive.ics.uci.edu/dataset/52) | 351 | 34 |
| Banknote Authentication | [267](https://archive.ics.uci.edu/dataset/267) | 1372 | 4 |
| Blood Transfusion | [176](https://archive.ics.uci.edu/dataset/176) | 748 | 4 |
| Climate Model Simulation Crashes | [252](https://archive.ics.uci.edu/dataset/252) | 540 | 18 |
| QSAR Biodegradation | [254](https://archive.ics.uci.edu/dataset/254) | 1055 | 41 |
| Raisin | [850](https://archive.ics.uci.edu/dataset/850) | 900 | 7 |
| Rice (Cammeo and Osmancik) | [545](https://archive.ics.uci.edu/dataset/545) | 3810 | 7 |
| Spambase | [94](https://archive.ics.uci.edu/dataset/94) | 4601 | 57 |
| Diabetic Retinopathy Debrecen | [329](https://archive.ics.uci.edu/dataset/329) | 1151 | 19 |

As dimensões são referências anteriores ao pré-processamento. Registrar os valores efetivamente encontrados. Excluir identificadores (inclusive os de simulação na Climate), verificar se registros relacionados ou duplicados exigem permanecer na mesma partição e documentar qualquer tratamento. Não substituir bases sem informar.

## 3. Pré-processamento

- Separar atributos `X` e alvo `y`; codificar as classes como 0 e 1 e documentar seu significado.
- Investigar valores ausentes, inconsistentes, duplicados, atributos constantes e desbalanceamento.
- Imputar atributos numéricos ausentes pela mediana quando necessário; remover atributos constantes e identificadores.
- Classificadores: padronizar atributos com `StandardScaler`: `z = (x - média) / desvio_padrão`.
- Roteador dos híbridos da versão 2: Yeo-Johnson seguido de padronização (`PowerTransformer`), para que caudas pesadas não formem clusters próprios nem dominem as distâncias.
- Usar PCA para visualizar os agrupamentos em duas dimensões, sem substituir os atributos do experimento principal. Não interpretar a projeção como a geometria completa dos dados.

Ajustar as transformações exclusivamente nos dados de treinamento de cada divisão. Não balancear ou transformar o conjunto de teste a partir de suas próprias estatísticas.

## 4. Classificadores

Comparar cinco classificadores individuais e cinco estratégias híbridas. Os hiperparâmetros abaixo são valores fixados para manter o experimento simples e reproduzível, não resultados de otimização.

| ID | Modelo | Configuração |
|---|---|---|
| M1 | SVM linear | `CalibratedClassifierCV(SVC(kernel="linear", C=1), method="sigmoid", cv=5, ensemble=False)` |
| M2 | SVM-RBF | `CalibratedClassifierCV(SVC(kernel="rbf", C=1, gamma="scale"), method="sigmoid", cv=5, ensemble=False)` |
| M3 | MLP | Uma camada oculta de 32 neurônios, tanh, SGD, backpropagation, momentum=0.9, learning_rate_init=0.01, alpha=0.0001, max_iter=1000 |
| M4 | Gaussian Naive Bayes | `GaussianNB()` |
| M5 | KNN | `KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")` |
| M6 | Híbrido global (aula) | Pesos proporcionais à acurácia global; probabilidades brutas |
| M7 | Híbrido local original | Pesos proporcionais à acurácia em cada cluster (K = 3); probabilidades brutas |
| M8 | Híbrido global v2 | Pesos pela acurácia acima da classe majoritária; probabilidades calibradas |
| M9 | Híbrido local K-Means v2 | Idem, por cluster, com K e suavização escolhidos na validação interna |
| M10 | Híbrido local k-NN v2 | Idem, na vizinhança de cada instância, com k e suavização escolhidos na validação interna |

A SVM linear explora fronteiras lineares e margem máxima; a SVM-RBF e a MLP, relações não lineares; o Naive Bayes, probabilidades a posteriori (MAP); e o KNN, similaridade entre instâncias. A regularização e os parâmetros dos modelos controlam a complexidade e ajudam a discutir generalização e viés-variância. Nas SVMs, `CalibratedClassifierCV(..., ensemble=False)` aplica o mesmo procedimento de Platt que `probability=True`, que foi deprecado.

Utilizar as implementações individuais do scikit-learn. A lógica de agrupamento, vizinhança, calibração dos combinadores, estimação de competência, cálculo de pesos, escolha de parâmetros e fusão deve ser coordenada por código próprio. Não utilizar `VotingClassifier`, `StackingClassifier` ou outros combinadores prontos. Usar as mesmas saídas dos cinco modelos para construir todos os híbridos.

## 5. Sistema híbrido autoral

### 5.1. Formulação original (M6, M7)

Mantida como ablação e para mostrar o ponto de partida (ponderação de performance da aula 09).

- K-Means com `n_clusters=3` e `n_init=20` sobre os atributos padronizados do treino externo, sem rótulos.
- `competencia_global[m]` = acurácia das predições OOF; `competencia_local[m,c]` = acurácia no cluster `c`.
- `peso[m] = competencia[m] / soma(competencias)`.
- Cluster sem exemplos usa os pesos globais; soma global zero usa 1/5.

### 5.2. Calibração (M8–M10)

Antes da fusão, cada modelo é calibrado por Platt sobre o logit da probabilidade positiva, com os alvos suavizados de Platt. Os parâmetros são ajustados nas predições OOF:

- para estimar competências no treino externo, cada dobra interna é calibrada com parâmetros ajustados nas outras dobras (ajuste cruzado);
- para o teste externo, os parâmetros são ajustados em todas as OOF.

### 5.3. Competência acima da classe majoritária (M8–M10)

Um modelo só contribui se acertar mais do que a regra "classe majoritária" na região. Para o modelo `m` e a região `r` com `n_r` exemplos de treino:

```text
acuracia[m,r] = (acertos[m,r] + λ · acuracia_global[m]) / (n_r + λ)
taxa_pos[r]   = (positivos[r] + λ · taxa_pos_global) / (n_r + λ)
base[r]       = max(taxa_pos[r], 1 − taxa_pos[r])
peso[m,r]     = max(acuracia[m,r] − base[r], 0) / soma_m(max(acuracia[m,r] − base[r], 0))
```

`λ` é o número de pseudo-observações que puxam a região para os valores globais. Se nenhum modelo supera a base na região, usam-se os pesos globais de M8; se nenhum supera globalmente, 1/5. A diferença entre M8, M9 e M10 é somente a região:

- M8: todo o treino externo.
- M9: cluster do K-Means no espaço do roteador.
- M10: os k vizinhos mais próximos da instância no espaço do roteador.

### 5.4. Escolha das regiões e da suavização

- Candidatos: K ∈ {1, …, 6} para M9, k ∈ {7, 15, 31} para M10 e λ ∈ {0, 10, 30}.
- Para cada candidato, as competências de cada dobra interna são estimadas só com as outras dobras, e as OOF fundidas são avaliadas por ROC-AUC. Vence o maior valor.
- Empates mantêm o primeiro candidato da ordem K crescente, k decrescente e λ decrescente. K = 1 reproduz a ponderação global, de modo que M9 só se afasta de M8 quando há sinal local no treino.

Registrar os tamanhos dos grupos, within e between para cada K (aula 08), e as escolhas feitas em cada fold. Os clusters não são necessariamente equivalentes às classes.

### 5.5. Fusão e decisão

```text
score[j] = soma(peso[m](x) * probabilidade[m,j](x) para m em 1..5)
classe_predita = argmax(score)
```

Alinhar as colunas de `predict_proba` pela ordem das classes. Empates resolvidos para a classe 0. As saídas são estimativas; não afirmar que as probabilidades são perfeitamente calibradas.

## 6. Protocolo experimental

Executar 3 repetições (sementes fixadas em `src/config.py`) de validação cruzada estratificada com 5 folds externos e 3 internos. A semente da repetição define as partições, a MLP e o K-Means, e as mesmas divisões valem para todos os modelos. Se houver grupos de observações dependentes, mantê-los na mesma partição e documentar a divisão.

Para cada base, repetição e fold externo:

1. Separar treinamento e teste externos. Reservar o teste exclusivamente para a avaliação final.
2. Dividir o treinamento externo em três folds internos. Para cada um, ajustar o pré-processamento e os cinco classificadores nos outros dois e predizer o fold restante, de forma que cada exemplo tenha uma predição de cada modelo que não o utilizou no ajuste.
3. Ajustar os roteadores (M7 e versão 2) sobre todo o treinamento externo, sem rótulos.
4. Calcular competências e pesos (seção 5), incluindo a calibração e a escolha de K, k e λ, usando apenas as predições internas.
5. Treinar novamente os cinco classificadores, com seus pré-processadores, sobre todo o treinamento externo.
6. Predizer o teste externo com os cinco modelos e os cinco híbridos, sem recalcular nada no teste.
7. Calcular e salvar as métricas dos dez modelos.

O protocolo produz 1500 registros (10 bases × 10 modelos × 3 repetições × 5 folds), agregados em 100 resultados (média e desvio dos 15 folds por par base/modelo).

## 7. Métricas e análise estatística

Calcular a partir dos resultados de teste:

- Acurácia: `(VP + VN) / N`.
- Acurácia balanceada: `(sensibilidade + especificidade) / 2`.
- Precisão: `VP / (VP + FP)`.
- Sensibilidade: `VP / (VP + FN)`.
- Especificidade: `VN / (VN + FP)`.
- ROC-AUC: área sob a curva ROC, utilizando o score contínuo da classe positiva, não apenas a classe predita.
- Matriz de confusão: VP, VN, FP e FN.

Registrar média e desvio-padrão de cada métrica por base e modelo e identificar métricas indefinidas e sua causa. A ROC-AUC é a métrica principal da comparação estatística, definida antes da execução, considerando que algumas bases são desbalanceadas. Analisar também acurácia balanceada, sensibilidade e especificidade.

### Friedman-Nemenyi (comparação principal)

1. Construir uma matriz de 10 linhas (bases) e 8 colunas (M1–M5, M8, M9, M10) com a ROC-AUC média dos 15 folds externos.
2. Aplicar o teste de Friedman com nível de significância de 5%, considerando as bases como blocos. H0: não há diferença sistemática entre os rankings dos modelos; H1: existe diferença.
3. Calcular o ranking médio: rank 1 para o melhor resultado de cada base, utilizando ranks médios nos empates.
4. Se Friedman rejeitar H0, executar o pós-teste de Nemenyi e apresentar os p-valores entre pares. A diferença crítica é `CD = q_alpha * sqrt(A*(A+1)/(6*N))`, com `A=8` e `N=10`. Apresentar o diagrama de diferença crítica.

### Comparações pareadas

- M9 contra cada um dos outros sete da comparação principal: Wilcoxon pareado por base (diferenças nulas divididas entre os sinais), correção de Holm e teste do sinal (vitórias, derrotas e empates).
- Ablação, com Holm dentro da família: M8 × M6 (correção da competência e calibração), M9 × M7 (versão 2 × original) e M10 × M8 (local por vizinhança × global).

Se um teste não for significativo, não concluir que os modelos são equivalentes nem que há superioridade comprovada. Usar SciPy e scikit-posthocs. Não tratar folds como bases independentes. Não executar os testes finais com resultados incompletos nem inventar resultados.

## 8. Organização e implementação

```text
project/
├── README.md
├── requirements.txt
├── main.py
├── data/
├── docs/
├── src/
│   ├── config.py
│   ├── datasets.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── hybrid.py
│   ├── evaluation.py
│   ├── statistical_tests.py
│   └── presentation.py
├── scripts/
├── tests/
└── results/
```

Bibliotecas: NumPy, pandas, scikit-learn, SciPy, scikit-posthocs, matplotlib e ucimlrepo. Utilizar CPU, sem serviços ou frameworks adicionais. Centralizar os parâmetros em `src/config.py` e evitar contagens fixas no código.

Validar o código antes de executar as dez bases:

- pesos não negativos e com soma igual a 1;
- alinhamento das probabilidades e predições consistentes;
- separação entre treino e teste e cobertura da validação interna;
- calibração em ajuste cruzado sem usar os rótulos da própria dobra;
- igualdade entre o híbrido local e o global quando há uma única região;
- vizinhanças contidas no treinamento externo.

## 9. Entregáveis

- Código funcional e instruções curtas de instalação e execução no README.md.
- Tabela com características das dez bases e tratamentos realizados.
- CSV com métricas por fold e tabela consolidada de médias e desvios-padrão.
- Pesos, escolhas da validação interna, tamanhos dos clusters, within/between e medidas de diversidade.
- Resultados de Friedman, rankings médios, Nemenyi, diagrama de diferença crítica e comparações pareadas.
- Diagrama da arquitetura em Mermaid, UML ou ferramenta de diagramação (ex.: Lucidchart), com etapas delimitadas por linhas pontilhadas. Não utilizar ferramentas de geração de imagens.
- Apresentação breve com arquitetura, funcionamento do mecanismo autoral, comparação experimental e conclusões.

Executar as etapas até obter resultados reais. Fixar parâmetros e candidatos antes da execução (PLANO.md) e registrar qualquer desvio. Se houver impedimentos, registrar o que falhou e entregar o que foi efetivamente concluído.
