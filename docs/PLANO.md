# Plano e decisões anteriores à execução

Fonte normativa: [AGENTS.md](AGENTS.md). Diagnóstico que motivou a versão 2: [ESTUDO_MELHORIAS.md](ESTUDO_MELHORIAS.md).

## Histórico

A primeira execução (commit `b1b9b49`) usou uma única partição 5×3 com semente 42 e os híbridos M6 e M7. Ela
mostrou que, com pesos proporcionais à acurácia bruta, as ponderações local e global quase não diferem. As
variantes da versão 2 foram desenhadas depois de olhar esses folds. Por isso, a avaliação confirmatória usa
partições e sementes novas, e tudo o que está abaixo foi fixado antes de rodá-la.

## Dados e partições

1. Adquirir as dez bases UCI, preservar arquivos, metadados, variáveis e hashes, e conferir dimensões e classes.
   Auditar ausências, infinitos, constantes, duplicatas e dependências.
2. Duplicatas de atributos, inclusive com rótulos conflitantes, são preservadas e mantidas juntas nas divisões
   externas e internas (StratifiedGroupKFold). Sem duplicatas, usa-se StratifiedKFold. Nenhuma observação é
   removida e nenhum rótulo é alterado.
3. Na Climate, os índices de estudo e execução são excluídos dos atributos. A dependência entre desenhos fica
   registrada como limitação.

## Execução confirmatória (versão 2)

**Protocolo.** São 3 repetições com as sementes 11, 22 e 33, cada uma com 5 folds externos × 3 internos. A
semente da repetição define as partições, a inicialização da MLP e a do K-Means. A métrica principal é a
ROC-AUC média dos 15 folds externos de cada base, com α = 0,05.

**Modelos base.** Configuração da versão 1, com duas mudanças:

- SVMs: `CalibratedClassifierCV(SVC(...), method="sigmoid", cv=5, ensemble=False)` no lugar de
  `SVC(probability=True)`. É o mesmo procedimento de Platt, sem o parâmetro deprecado.
- MLP: `early_stopping=True`, com os valores padrão do scikit-learn (`validation_fraction=0.1`,
  `n_iter_no_change=10`), contra a falta de convergência observada em QSAR e Retinopathy. **Revertido**: ver
  "Desvio registrado" abaixo.

**Híbridos.**

| ID | Probabilidades | Competência | Regiões |
|---|---|---|---|
| M6 | brutas | acurácia OOF / soma (fórmula da aula 09) | global |
| M7 | brutas | idem, por cluster | K-Means, K = 3, padronização |
| M8 | calibradas (Platt, ajuste cruzado nas dobras internas) | acurácia acima da taxa da classe majoritária | global |
| M9 | calibradas | idem, suavizada com λ pseudo-observações | K-Means, K escolhido |
| M10 | calibradas | idem, suavizada com λ | k vizinhos mais próximos de cada instância |

- **Roteador de M9 e M10:** Yeo-Johnson seguido de padronização, ajustado no treino externo sem rótulos. A
  escolha foi feita comparando apenas o tamanho dos clusters, sem rótulos: padronização simples gera clusters
  de 13–34 exemplos em QSAR, Retinopathy e Spambase, e Yeo-Johnson, de 189 ou mais.
- **Candidatos:** K ∈ {1, …, 6}, k ∈ {7, 15, 31} e λ ∈ {0, 10, 30}.
- **Critério de escolha:** ROC-AUC das OOF fundidas com competências estimadas nas outras dobras internas.
- **Empates:** vence o primeiro candidato, na ordem K crescente, k decrescente e λ decrescente. Assim, empates
  favorecem a configuração mais suave, e K = 1 equivale à ponderação global.

**Métricas.** Acurácia, acurácia balanceada, precisão, sensibilidade, especificidade, ROC-AUC e matriz de
confusão.

**Estatística.** A unidade é a base (10 blocos), nunca o fold.

1. Comparação principal: Friedman com M1–M5, M8, M9 e M10 (A = 8, CD = 3,32). Nemenyi somente se p < 0,05.
2. Controle M9 contra os outros sete: Wilcoxon pareado (diferenças nulas divididas), correção de Holm e teste do
   sinal.
3. Ablação: M8 × M6, M9 × M7 e M10 × M8, com Wilcoxon e Holm dentro da família.

**Leitura dos resultados.** Superioridade somente com p ajustado < 0,05; sem significância, não se conclui
equivalência. A pergunta "a ponderação local ajuda?" é respondida por M9 × M8 e M10 × M8, e "a correção da
competência ajuda?", por M8 × M6.

**Sem ajustes posteriores.** Nenhum parâmetro é alterado depois de observar os resultados. Qualquer desvio é
registrado nesta seção, com a justificativa.

### Desvio registrado: MLP sem early stopping

A primeira rodada confirmatória mostrou a MLP muito abaixo da versão 1 (ROC-AUC média de 0,68 em Climate e
0,80 em Ionosphere, contra 0,94 e 0,92). Para conferir a causa, usamos só dados de treino (primeiro fold
externo da semente 11, validação cruzada de 3 dobras dentro dele):

- com `early_stopping=True` e paciência padrão, o treino para em 16–23 iterações, porque a reserva de 10% para
  validação tem poucas dezenas de exemplos e oscila demais;
- a perda de treino fica entre 0,25 e 0,46 (contra 0,03–0,06 sem early stopping);
- a AUC na validação interna cai para 0,60 (Climate), 0,72 (Ionosphere) e 0,75 (Sonar), contra 0,94, 0,94 e
  0,85 sem early stopping.

A MLP voltou à configuração da versão 1 (`max_iter=1000`, sem early stopping), já especificada antes de
qualquer resultado. Não se buscou outra paciência nem outro número de iterações. Os avisos de convergência
voltam a aparecer e ficam registrados como limitação. A correção fortalece um modelo individual, ou seja,
trabalha contra a vantagem dos híbridos, e não a favor. Nada mais mudou: as sementes, os candidatos e os
testes são os mesmos. Todos os resultados publicados vêm da segunda rodada.

## Dependências

NumPy/pandas para dados; ucimlrepo para aquisição; scikit-learn para transformações, modelos, K-Means, vizinhos
e PCA; SciPy (otimização da calibração, Friedman, Wilcoxon, sinal) e scikit-posthocs (Nemenyi); matplotlib
para figuras; unittest para testes. Ambiente Conda `rec_pad`, Python 3.12, versões em `requirements-lock.txt`.
