# Estudo: escopo, aulas e melhorias do sistema híbrido

> Este estudo analisa a **versão 1** do projeto. O que foi implementado a partir dele e o resultado da
> versão 2 estão na [seção 11](#11-o-que-foi-implementado-e-o-resultado).

Base analisada: commit `b1b9b49`, resultados em `results/` (execução 5×3, semente 42) e aulas 01–09 em `aulas/`.
Os hashes de `results/environment.json` conferem com o código atual (após normalizar CRLF → LF), então os
resultados correspondem ao código do repositório.

## 1. Resumo

1. **O projeto atende formalmente à atividade**: dez bases UCI, cinco classificadores individuais, dois
   híbridos, Friedman + Nemenyi e arquitetura em Mermaid. A engenharia de avaliação é sólida: validação
   cruzada aninhada, agrupamento de duplicatas, nenhuma estatística ajustada no teste e uma validação
   independente dos artefatos.
2. **Fragilidade central: o híbrido local (M7) é, na prática, o híbrido global (M6), e ambos quase coincidem
   com a média simples das probabilidades.** Todos os pesos ficam entre 0,136 e 0,232 (uniforme = 0,200).
   M7 altera a decisão de M6 em, no máximo, 0,57% das instâncias de teste de uma base (média dos folds), e em
   nenhuma instância em cinco das dez bases. Com isso, a pergunta de pesquisa ("ponderar localmente ajuda?")
   fica sem resposta por construção, e não por falta de evidência.
3. **Causa:** competência = acurácia bruta, normalizada linearmente. Um modelo que só acerta tanto quanto
   "chutar a classe majoritária" ainda recebe cerca de 20% do peso. A aula 03 dá o critério que falta (a
   acurácia só é aceitável quando supera a da classe majoritária), e a aula 09 mostra, com o próprio exemplo
   do professor, que a fusão colapsa quando as competências são parecidas.
4. **Segunda fragilidade: a fusão linear usa probabilidades não calibradas.** O GaussianNB atribui
   probabilidade > 0,99 a 74–99% das instâncias em seis bases e, em Retinopathy, define sozinho a decisão de
   M6 contra os outros quatro modelos em 9,4% das instâncias.
5. **Terceira: as regiões do K-Means com K = 3 fixo isolam outliers** (clusters de 7, 9 e 12 exemplos em
   Blood, Retinopathy e QSAR). Nessas regiões, a competência local estimada é ruído.
6. **Reanálise exploratória** das previsões já salvas, sem re-treinar: medir a competência acima da classe
   majoritária melhora M6 em 8 de 10 bases (+0,0049 de ROC-AUC média); somando a isso a calibração das
   probabilidades antes da fusão, o ganho chega a +0,0062. Isso é indício, não resultado: esses folds já foram observados. **Mesmo com a
   competência corrigida, a variante local ainda não superou a global**, então a melhoria precisa vir também
   da definição das regiões (seção 7).

## 2. O que o projeto propõe hoje

| ID | Modelo | Papel |
|---|---|---|
| M1 | SVM linear (C=1, Platt interno) | fronteira linear, margem máxima |
| M2 | SVM-RBF (C=1, gamma="scale") | não linear, kernel |
| M3 | MLP 32 tanh, SGD + momentum | não linear, backpropagation |
| M4 | GaussianNB | MAP com independência condicional |
| M5 | KNN k=5 ponderado pela distância | baseado em instâncias |
| M6 | Híbrido global | `w_m = a_m / Σ a_r`, com `a_m` = acurácia OOF |
| M7 | Híbrido local | mesmo cálculo, por cluster do K-Means (K=3) |

Fluxo por fold externo (5): três folds internos geram probabilidades fora da amostra (OOF) para os cinco
modelos → o K-Means é ajustado no treino externo sem rótulos → competências global e por cluster → os modelos
são reajustados no treino externo → fusão no teste externo. Friedman sobre a matriz 10 × 7 de ROC-AUC
média; Nemenyi se p < 0,05.

Resultado atual: Friedman p = 0,00025; M6 e M7 empatam no ranking médio (2,35); Nemenyi separa os híbridos
de NB (p = 0,027) e KNN (p = 0,002), mas não de SVM linear, SVM-RBF ou MLP; M7 × M6 tem p = 1.

## 3. Atendimento ao enunciado

| Requisito do enunciado | Situação | Observação |
|---|---|---|
| 10 bases | Atendido | UCI, binárias, auditadas (`docs/BASES.md`) |
| Pelo menos um modelo híbrido | Atendido | K-Means (não supervisionado) + 5 supervisionados + combinador próprio |
| Friedman-Nemenyi ou ANOVA-Tukey | Atendido | Friedman-Nemenyi; é a escolha certa para AUC entre bases (Demšar, 2006, citado na aula 06) |
| Apresentação + figura da arquitetura | Parcial | `APRESENTACAO.md` e Mermaid existem; faltam a versão sucinta e a figura no Lucidchart |

O `docs/AGENTS.md` impõe restrições que **o enunciado não exige**: proíbe suavização, expoentes e critério
para escolher K, além de vetar algoritmos ou experimentos extras. Essas restrições são a causa direta das
fragilidades F1 e F4 abaixo. Revisá-las é uma decisão sua (seção 10).

## 4. Integração com as aulas

| Aula | Conteúdo-chave | Onde aparece no projeto | Lacuna / oportunidade |
|---|---|---|---|
| 01 Introdução | generalização, viés-variância, validação, k-fold, matriz de confusão e métricas | CV aninhada; acurácia, precisão, sensibilidade, especificidade | Viés-variância aparece só como texto. O ensemble reduz variância, e isso pode ser mostrado com o desvio entre folds |
| 02 RNAs | perceptron, MLP, backpropagation, momentum, mínimos locais, "ensemble de RNAs" contra mínimos locais | M3 segue a aula (tanh, SGD, momentum 0,9) | A MLP não convergiu em **todos os 20 ajustes** de QSAR e em 5 de Retinopathy; nada foi feito a respeito |
| 03 Pré-processamento | imputação, constantes, redundância, inconsistência, **desbalanceamento** ("acurácia aceitável se maior que a da classe majoritária"), padronização, PCA | mediana, VarianceThreshold, z-score e PCA só para figuras; duplicatas e conflitos agrupados | O critério da classe majoritária **não** é usado na competência (F2). Climate (8,5% de falhas) e Blood (24% de doadores) não têm nenhum tratamento de custo/limiar: especificidade de 0,48 em Climate e sensibilidade de 0,14 em Blood |
| 04 Bayes | MAP, Naive Bayes, independência condicional | M4 | A suposição de independência é violada (Sonar: bandas adjacentes; Spambase: frequências correlacionadas). O resultado são probabilidades extremas que dominam a fusão (F3) |
| 05 SVM | margem, kernels, **sensibilidade a C e γ**, seleção de modelo | M1, M2 com C=1 e γ="scale" fixos | `SVC(probability=True)` gera 400 FutureWarnings (deprecado). Não há nenhuma justificativa experimental para C e γ |
| 06 Avaliação | k-fold, desvio-padrão, ROC/AUC, t-test, Wilcoxon, sinal, Friedman, Nemenyi, **CD**, "analisar em vez de descrever tabelas" | Friedman + Nemenyi com CD idêntico ao da aula; ROC-AUC como métrica principal | Não há diagrama CD no estilo da aula (`rankings.png` é só um scatter). Também faltam teste pareado contra controle (Wilcoxon/sinal), repetição da CV e análise de diversidade |
| 07 IBL / KNN | KNN ponderado pela distância, maldição da dimensionalidade, regressão local, RBF com centros via k-means | M5 segue a aula | O KNN fica em último no ranking (6,1), coerente com a aula nas bases de alta dimensão. A ideia de vizinhança como região de competência (OLA) ainda não foi explorada |
| 08 Não supervisionado | K-Means, inicialização, within/between, **fraco com grupos não globulares e de tamanhos/densidades diferentes** | roteador K-Means (n_init=20); within/between reportados | K = 3 fixo. O ponto fraco citado na aula aparece nos dados: clusters de outliers (F4) |
| 09 Sistemas híbridos | fusão (voto, **ponderação de performance `w_i = α_i/Σα_j`**), **seleção** (K-classificadores, clustering-and-selection, seleção dinâmica k-NN, a priori × a posteriori), stacking, **diversidade** | M6 = fórmula da aula; M7 = mesma fórmula por região | O exemplo da aula (três classificadores com precisão ≈ 0,5: a fusão ≈ D3, e a seleção por região erra menos) é exatamente o fenômeno M6 ≈ M7. O projeto não implementa seleção, não escolhe K por critério (a aula manda testar K = 2…Kmax), não mede diversidade e não tem baseline de stacking |

Leitura geral: o projeto cobre bem as aulas 01, 03, 06 e 08 do ponto de vista de **protocolo**, mas
aproveita pouco a aula 09, que é justamente o núcleo do trabalho. A fusão foi implementada; seleção,
diversidade e calibração, que a aula apresenta como condições para a combinação funcionar, ficaram de fora.

## 5. Fragilidades, em ordem de impacto

### F1. Os pesos são comprimidos, e por isso M7 ≈ M6 ≈ média simples

Com `w_m = a_m / Σ a_r`, a razão entre dois pesos é a razão entre as acurácias. Como as acurácias ficam entre
0,62 e 1,00, um modelo fraco recebe no mínimo ~60% do peso do melhor.

| Evidência | Valor |
|---|---|
| Faixa dos pesos globais (todas as bases e folds) | 0,158 – 0,218 |
| Faixa dos pesos locais | 0,136 – 0,232 |
| Maior diferença peso local − global (média por base) | 0,044 (Blood) |
| Instâncias de teste em que M7 muda a decisão de M6 | 0 a 0,57% (zero em 5 bases) |
| ROC-AUC média: média simples − M6 | −0,0008 |
| Desvio-padrão entre folds da AUC de M6 (Sonar, Blood) | 0,057; 0,046, contra \|M7 − M6\| ≤ 0,001 |

Consequência: o bom ranking dos híbridos (2,35) vem de **fazer a média de cinco modelos diversos**, e não da
ponderação por competência. Na apresentação, hoje, não há como sustentar que "a competência local" contribui.

### F2. A competência por acurácia ignora a classe majoritária e não conversa com a métrica principal

- Climate: a classe majoritária responde por 91,5%; os cinco modelos têm acurácia OOF entre 0,924 e 0,950 e
  recebem pesos entre 0,196 e 0,202. Medidos acima da classe majoritária, os excessos vão de 0,009 a 0,035, uma
  diferença de 4×.
- Blood: a classe majoritária responde por 76,2%. NB (0,750) e KNN (0,742) ficam **abaixo** desse patamar e
  mesmo assim recebem 0,197 e 0,194.
- A comparação final usa ROC-AUC (ordenação), mas a competência usa acurácia no limiar 0,5 (decisão).

### F3. A fusão linear de probabilidades não calibradas é dominada pelo modelo mais confiante, que não é o melhor

| Base | NB com prob. > 0,99 | M6 segue o NB contra a maioria dos outros 4 | Acurácia OOF do NB |
|---|---|---|---|
| Retinopathy | 29% | **9,4%** | 0,617 (pior modelo) |
| QSAR | 97% | 3,9% | 0,664 (pior modelo) |
| Sonar | 75% | 3,9% | 0,677 (pior modelo) |

Ao remover o NB de M6 (exploratório), a AUC sobe +0,013 em Retinopathy e +0,011 em Sonar. A aula 09
condiciona a confiança `C(D_i|x) = max_j P(ω_j|x, D_i)` a "estimativas bem calibradas"; o projeto
registra que as probabilidades não são calibradas, mas não trata isso.

### F4. As regiões do K-Means não representam regiões de competência

- K = 3 é fixo, sem relação com a estrutura dos dados. A aula 09 (K-classificadores) testa K = 2…Kmax e usa
  um critério para escolher.
- Clusters mínimos de 7 (Blood), 9 (Retinopathy), 12 (QSAR) e 23 (Spambase) exemplos. Em
  `results/figures/pca/retinopathy_1.png`, o terceiro cluster é formado por ~11 pontos dispersos: o K-Means,
  sobre atributos padronizados com caudas pesadas, gastou um centroide com outliers (exatamente o ponto fraco
  descrito na aula 08).
- A acurácia local nesses grupos é estimada com poucos exemplos e não tem nenhum controle de variância. O
  fallback só cobre cluster vazio.
- O K-Means agrupa por geometria dos atributos, não por onde os classificadores diferem. Nada garante que
  esses clusters separem regiões em que os modelos têm desempenhos diferentes.

### F5. Seleção × fusão: a aula foca seleção, e o projeto implementa só fusão

A fórmula da aula `P(correct) = Σ_j P(R_j) · P(D_i(j) | R_j)` é maximizada escolhendo, em cada região, o
classificador mais competente. Na reanálise (seção 6), a seleção pura por cluster **melhora a acurácia
balanceada** (+0,0085, a maior entre as variantes) mas **piora a ROC-AUC** (−0,0046), porque mistura escalas
de score de modelos diferentes entre regiões, o que prejudica uma métrica de ordenação global. Isso importa
para o desenho: um híbrido por seleção precisa de calibração, ou precisa ser avaliado também por uma métrica
de decisão.

### F6. Inferência estatística com pouco poder e uma única partição

- Nemenyi com 7 métodos e 10 bases exige uma diferença de ranking de 2,85; com 8 métodos, 3,32. Incluir dois
  híbridos quase idênticos (M6, M7) não acrescenta informação e divide posições de ranking.
- Uma única execução de 5 folds com semente 42: diferenças da ordem de 0,001 entre híbridos estão muito
  abaixo do desvio entre folds.
- A aula 06 lista Wilcoxon e o teste do sinal (vitórias/derrotas/empates). Para "híbrido × cada
  alternativa", Demšar (2006) recomenda comparação contra um controle (Wilcoxon ou Bonferroni-Dunn/Holm),
  que tem mais poder que o Nemenyi todos-contra-todos.
- `results/figures/rankings.png` não é um diagrama de diferença crítica: não mostra o CD nem os grupos não
  distinguíveis, como nos slides da aula 06.

### F7. Modelos base

- MLP: `ConvergenceWarning` em 20/20 ajustes de QSAR e 5 de Retinopathy. As probabilidades OOF de um modelo
  que não convergiu entram na competência.
- SVM: `SVC(probability=True)` está deprecado (400 avisos) e roda uma calibração de Platt interna com 5
  folds que o projeto não controla. O argmax dessas probabilidades pode divergir do sinal da função de
  decisão.
- Hiperparâmetros fixos. Isso é aceitável para manter o escopo, mas a aula 05 enfatiza a sensibilidade a C e γ.
  Vale pelo menos justificar, ou fazer uma busca pequena nos folds internos.

### F8. Engenharia e reprodutibilidade

- Contagens fixas no código impedem acrescentar modelos: `5` em
  [src/evaluation.py:77](../src/evaluation.py#L77) e [src/evaluation.py:106](../src/evaluation.py#L106);
  `7*8/(6*10)` em [src/statistical_tests.py:37](../src/statistical_tests.py#L37);
  `range(7)`/`range(10)` em [src/presentation.py:36-39](../src/presentation.py#L36-L39);
  `350`, `range(1, 6)` e `7*8/(6*10)` em [scripts/validate_results.py:18](../scripts/validate_results.py#L18),
  [scripts/validate_results.py:24](../scripts/validate_results.py#L24) e
  [scripts/validate_results.py:71](../scripts/validate_results.py#L71).
- O hash em `environment.json` é calculado sobre bytes com LF; com `core.autocrlf=true` no Windows, a
  conferência falha se for refeita sobre o checkout. Normalizar as quebras de linha antes do hash.
- O ambiente Conda `rec_pad` não existe nesta máquina e `data/raw/` não está no repositório. Uma nova
  execução exige recriar o ambiente (Python 3.12, `requirements-lock.txt`) e baixar as bases.

### F9. Restrições autoimpostas

"Sem suavização, expoentes ou outras estratégias de ponderação", "K = 3 sem índice de seleção" e "não
acrescentar algoritmos" (AGENTS.md §5.1, §5.3, §9) tornaram o experimento limpo, mas também fixaram
exatamente as escolhas que produzem F1 e F4. O enunciado da disciplina não pede nada disso.

## 6. Reanálise exploratória com as previsões salvas

Script: [scripts/explore_combiners.py](../scripts/explore_combiners.py)
(`python -m scripts.explore_combiners <pasta folds da versão 1>`). Ele lê os arquivos
`results/folds/*/*_predictions.npz` do commit `b1b9b49`, que podem ser extraídos com
`git archive b1b9b49 results/folds | tar -x -C <destino>`. Nenhum modelo é re-treinado. Todas as variantes
derivam os pesos apenas das previsões OOF do treino externo. O K-Means é o mesmo da execução (K=3); variantes
que exigem outro K ou vizinhança k-NN precisam de uma nova execução.

Variantes:

- `MEDIA_SIMPLES`: pesos 1/5.
- `G_EXCESSO`: `w_m ∝ max(a_m − b, 0)`, com `b` = taxa da classe majoritária no treino.
- `G_CAL_EXCESSO`: igual a `G_EXCESSO`, mas cada modelo é calibrado (Platt sobre logit) com as OOF antes da fusão.
- `L_EXCESSO_SUAVIZADO`: versão local de `G_EXCESSO`, com `â_mc = (acertos_mc + 20·a_m)/(n_c + 20)` e `b_c` do cluster.
- `L_SELECAO_1`: em cada cluster, usa só o modelo de maior `â_mc` (clustering-and-selection).
- `L_SELECAO_TOP3`: em cada cluster, média dos 3 modelos de maior `â_mc`.

ROC-AUC média por base:

| Base | M6 | M7 | Média simples | G excesso | G cal. excesso | L excesso suav. | L seleção 1 | L top-3 |
|---|---|---|---|---|---|---|---|---|
| sonar | 0,9141 | 0,9151 | 0,9109 | 0,9198 | 0,9233 | 0,9231 | 0,8938 | 0,9309 |
| ionosphere | 0,9830 | 0,9822 | 0,9820 | 0,9836 | 0,9819 | 0,9828 | 0,9812 | 0,9772 |
| banknote | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| blood | 0,7271 | 0,7269 | 0,7271 | 0,7551 | 0,7532 | 0,7407 | 0,7362 | 0,7405 |
| climate | 0,9543 | 0,9543 | 0,9541 | 0,9570 | 0,9598 | 0,9607 | 0,9497 | 0,9536 |
| qsar | 0,9333 | 0,9334 | 0,9327 | 0,9357 | 0,9351 | 0,9356 | 0,9136 | 0,9355 |
| raisin | 0,9338 | 0,9337 | 0,9337 | 0,9337 | 0,9339 | 0,9305 | 0,9241 | 0,9311 |
| rice | 0,9769 | 0,9769 | 0,9768 | 0,9769 | 0,9770 | 0,9775 | 0,9748 | 0,9781 |
| spambase | 0,9799 | 0,9800 | 0,9796 | 0,9801 | 0,9797 | 0,9794 | 0,9657 | 0,9778 |
| retinopathy | 0,7940 | 0,7944 | 0,7916 | 0,8032 | 0,8144 | 0,8073 | 0,8118 | 0,8183 |

Comparação pareada com M6 (Wilcoxon exploratório, sem correção para múltiplas comparações):

| Variante | Δ AUC média | Vitórias / derrotas | p | Δ acurácia balanceada |
|---|---|---|---|---|
| G cal. excesso | +0,0062 | 7 / 2 | 0,074 | +0,0059 |
| G excesso | +0,0049 | 8 / 1 | 0,012 | +0,0044 |
| L top-3 | +0,0047 | 5 / 4 | 0,43 | +0,0048 |
| L excesso suav. | +0,0041 | 6 / 3 | 0,098 | +0,0062 |
| M7 (atual) | +0,0000 | 4 / 4 | 0,84 | +0,0008 |
| Média simples | −0,0008 | 1 / 8 | 0,008 | +0,0005 |
| L seleção 1 | −0,0046 | 2 / 7 | 0,20 | **+0,0085** |

Diversidade e espaço de ganho (acurácia no teste):

| Base | Oráculo (algum modelo acerta) | Melhor individual | Discordância média entre pares |
|---|---|---|---|
| retinopathy | 0,912 | 0,731 | 0,276 |
| qsar | 0,954 | 0,880 | 0,171 |
| sonar | 0,952 | 0,875 | 0,196 |
| spambase | 0,984 | 0,936 | 0,115 |
| rice | 0,948 | 0,931 | 0,027 |
| climate | 0,970 | 0,961 | 0,037 |

O que se aprende:

1. **A ponderação atual equivale à média simples.** Corrigir a competência (excesso sobre a classe
   majoritária) é a mudança de maior efeito e mais consistente, e tem justificativa direta na aula 03.
2. **A calibração ajuda onde o NB atrapalha**: sobre `G excesso`, ela acrescenta +0,011 em Retinopathy e
   +0,004 em Sonar (no total, +0,020 e +0,009 sobre M6).
3. **Com a competência corrigida, local (K-Means, K=3) não superou global**: `L excesso suav.` × `G excesso` =
   4 vitórias, 5 derrotas. Pode ser que regiões K-Means simplesmente não sejam regiões de competência nessas
   bases. Esse seria um achado legítimo, desde que venha de um desenho que dê chance ao mecanismo local.
4. **Seleção e fusão otimizam coisas diferentes**: a seleção melhora a decisão e piora a ordenação. Se a
   apresentação incluir um híbrido por seleção, a métrica de decisão também precisa aparecer.
5. **O espaço de ganho está nas bases diversas** (Retinopathy, QSAR, Sonar, Spambase). Em Rice, Climate e
   Banknote os modelos concordam quase sempre, e nenhum combinador muda muito.

Cuidado: estes números foram obtidos depois de olhar os folds de teste. Servem para orientar o desenho, não
como resultado. O experimento confirmatório deve usar partições novas (seção 8).

## 7. Arquitetura proposta (v2)

Ideia para a apresentação: **partir da fórmula da aula (M6), mostrar por que ela colapsa e propor a
correção autoral**.

```text
Para cada fold externo:
  1. Pré-processamento por modelo (sem mudanças)
  2. Cinco modelos base → probabilidades OOF nos 3 folds internos
  3. Calibração de Platt por modelo, ajustada nas OOF            [novo]
  4. Roteador: padronização (+ log1p em atributos assimétricos)  [ajuste]
     K-Means com K ∈ {1, …, Kmax} escolhido por validação interna  [novo]
  5. Competência acima da classe majoritária, suavizada           [novo]
  6. Fusão ponderada das probabilidades calibradas
  7. Avaliação no teste externo
```

Formulação:

- Calibração: `p̃_m(x) = σ(α_m · logit p_m(x) + β_m)`, com `α_m, β_m` ajustados nas OOF. Para não estimar a
  competência com OOF calibradas na própria amostra, ajustar o calibrador em dobras cruzadas (cross-fitting
  sobre os 3 folds internos).
- Competência local suavizada: `â_mc = (acertos_mc + λ·a_m) / (n_c + λ)`. Em cluster pequeno, ela tende à
  global; em cluster grande, prevalece a local. Substitui a regra "cluster vazio → global".
- Peso: `w_mc = max(â_mc − b_c, 0) / Σ_r max(â_rc − b_c, 0)`, com `b_c` = taxa da classe majoritária no
  cluster (critério da aula 03). Se todos ficarem ≤ 0, usar os pesos globais; se esses também, 1/5.
- Escolha de K: para cada K ∈ {1, …, Kmax}, estimar as competências em 2 folds internos e medir a AUC do
  híbrido no 3º (rodízio). Escolher o K de maior média. **K = 1 recupera o global**, então o local só "liga"
  quando há sinal nos dados de treino. Reportar within/between de cada K, como na aula 08.

Híbridos a comparar (manter o número de métodos sob controle para não perder poder no Nemenyi):

| ID | Proposta | Relação com a aula 09 |
|---|---|---|
| M6′ | global, competência em excesso, probabilidades calibradas | ponderação de performance, corrigida |
| M7′ | local K-Means com K escolhido, competência suavizada | partição do espaço + ponderação dinâmica a priori |
| M8 | local por vizinhança: k vizinhos do ponto no espaço do roteador, acurácia OOF neles (OLA) | seleção dinâmica k-NN; Woods et al. (1997), referência [5] da aula |

Opcionais, como ablação fora do Friedman principal: M6 e M7 originais (fórmula da aula, para contar a
história), seleção top-3 por região e stacking com regressão logística escrita no próprio código sobre as OOF
(meta-learning da aula 09).

Friedman principal: M1–M5 + M6′ + M7′ + M8 = 8 métodos (CD = 3,32).

## 8. Protocolo e estatística

- **Execução confirmatória com partições novas**: por exemplo, 3 repetições de 5×3 com sementes diferentes de
  42; a AUC por base passa a ser a média de 15 folds. Isso estabiliza as estimativas e evita reaproveitar os
  folds usados na exploração.
- **Pré-registro**: fixar `λ`, `Kmax`, o `k` do OLA e as variantes antes de rodar, por escrito em `docs/`.
- Manter Friedman + Nemenyi (enunciado) e acrescentar:
  - diagrama de diferença crítica no estilo da aula 06;
  - Wilcoxon pareado M7′ × cada alternativa, com correção de Holm, e contagem vitórias/derrotas/empates
    (teste do sinal), como pedido nas "Considerações" da aula 06.
- Diversidade (aula 09): discordância entre pares, estatística Q de Yule e acurácia do oráculo por base,
  calculadas nas OOF. Isso ajuda a explicar em que bases a combinação tem espaço para ganhar.
- Opcional: acurácia balanceada como métrica complementar, por causa de Climate e Blood.

## 9. Modelos base e engenharia

- MLP: decidir antes da execução entre `early_stopping=True` (com `validation_fraction`) e `max_iter` maior /
  `learning_rate="adaptive"`, e registrar a escolha. A aula 02 permite discutir os dois caminhos.
- SVM: trocar `probability=True` por `decision_function` + a calibração do passo 3. Isso remove o
  componente deprecado e unifica a calibração de todos os modelos.
- Hiperparâmetros: manter fixos (escopo) e justificar na apresentação, ou fazer uma busca pequena (C, γ e k do
  KNN) nos folds internos. Recomendo manter fixos, porque o foco é o combinador.
- Generalizar as contagens fixas (F8) e cobrir com testes: pesos somam 1, K = 1 ⇒ M7′ ≡ M6′, cross-fitting
  sem vazamento, OLA com vizinhança contida no treino externo.
- Normalizar as quebras de linha no hash dos fontes.

## 10. Plano e decisões pendentes

Ordem sugerida:

1. **Decisões** (abaixo) e revisão do `AGENTS.md` para refletir o novo escopo.
2. **Refatoração sem mudar resultados**: remover as contagens fixas; rodar `--report-only` e
   `validate_results` para confirmar que os números atuais se mantêm.
3. **M6′ e M7′**: calibração, competência em excesso, suavização, escolha de K, com testes unitários.
4. **M8 (OLA)**.
5. **Protocolo**: repetições com sementes novas, diagrama CD, Wilcoxon-Holm, diversidade.
6. **Execução, validação, atualização** de `results/ANALISE.md` e do README.
7. **Apresentação sucinta e diagrama no Lucidchart** da arquitetura v2 (etapas delimitadas por linhas
   pontilhadas, conforme o AGENTS.md).

Decisões tomadas:

1. `AGENTS.md` revisado: suavização, escolha de K/k/λ e novos híbridos passaram a fazer parte da especificação.
2. M6 e M7 originais ficam como ablação.
3. 3 repetições, com sementes 11, 22 e 33.
4. MLP: o early stopping foi pré-registrado, causou subajuste e foi revertido (desvio registrado em
   [PLANO.md](PLANO.md)).
5. Stacking ficou de fora, para não diluir o Friedman.

## 11. O que foi implementado e o resultado

As etapas 1–6 do plano foram concluídas. Detalhes em [PLANO.md](PLANO.md), [METODO.md](METODO.md) e
[results/ANALISE.md](../results/ANALISE.md).

| Fragilidade | Tratamento | Efeito observado |
|---|---|---|
| F1 pesos comprimidos | competência acima da classe majoritária | faixa dos pesos globais de 0,003–0,05 (M6) para 0,01–0,50 (M8) |
| F2 baseline ignorado | idem | modelos abaixo da classe majoritária recebem peso zero |
| F3 probabilidades não calibradas | Platt por modelo, com ajuste cruzado nas dobras internas | M8 × M6: +0,0058 de AUC, 9 vitórias, 0 derrotas, 1 empate, p de Holm = 0,012 |
| F4 regiões do K-Means | roteador Yeo-Johnson; K ∈ 1–6 e λ escolhidos nas dobras internas; vizinhança k-NN (M10) | clusters de outliers desaparecem até K = 4; a validação interna escolhe K = 1 onde não há sinal local (Banknote, Spambase) |
| F5 seleção × fusão | fusão por pesos (seleção pura piora a AUC) | — |
| F6 estatística | 3 repetições, sementes novas, diagrama CD, Wilcoxon + Holm, teste do sinal | — |
| F7 modelos base | SVM com `CalibratedClassifierCV` (sem avisos de depreciação); MLP mantida | 73 avisos de convergência restantes (QSAR, Retinopathy) |
| F8 engenharia | contagens centralizadas em `src/config.py`; hash com LF; 20 testes; validação ampliada | — |

Resposta às duas perguntas:

- **A correção da competência melhora a fórmula da aula?** Sim. M8 supera M6 com significância, depois da
  correção de Holm.
- **A ponderação local melhora a global?** Não houve evidência. M9 × M8 terminou com 4 vitórias, 5 derrotas e
  1 empate (p = 0,96); M10 × M8, com 3, 6 e 1 (p = 0,35). Agora o mecanismo local de fato atua (os pesos se
  deslocam até 28%), então esse é um resultado do experimento, não um artefato do desenho.

Na comparação principal, os três híbridos v2 ocupam os três primeiros lugares do ranking. Diferem
significativamente de NB e KNN (Nemenyi), mas não das SVMs nem da MLP depois da correção de Holm.

## 12. Revisão de manutenção e reprodução em Linux (27/11/2026)

Esta revisão preserva o protocolo, os modelos, as sementes e os artefatos científicos versionados.
Os números das seções anteriores continuam descrevendo a execução original em Windows.

### Verificações realizadas

- 28 testes aprovados: os 20 existentes, cinco testes do gerador de slides e três testes de auditoria
  estatística. Os testes de slides percorrem 12 combinações de tamanho e presença de capa, de 1 a 40
  slides, e verificam XMLs, relacionamentos, notas, metadados e preservação da exportação após falha.
- Validação dos 150 folds externos e 1.500 registros originais: métricas, pesos, calibração, cobertura,
  separação de grupos e inferência estatística. A auditoria também passou a verificar as contagens das
  métricas, as famílias de testes, os p-valores de Holm e as decisões de significância.
- Reprodução integral, em uma saída separada, das dez bases e três repetições com Python 3.12.16 e as
  mesmas versões de pacotes. Os hashes de `main.py` e `src/*.py` coincidem com os originais.
- Renderização dos dez slides em PDF com LibreOffice e revisão visual; inspeção da versão revisada.

### Diferença de reprodutibilidade encontrada

Windows e Linux produziram 67 valores de ROC-AUC diferentes acima de 10⁻¹², todos em Blood. A maior
diferença absoluta por fold foi 0,02510 no KNN (M5). Em `r1_f1`, as partições e os grupos são idênticos,
e todas as 39 previsões divergentes do KNN têm empate de distância entre o 5º e o 6º vizinhos. Isso é
evidência de sensibilidade à escolha dos vizinhos em empates entre plataformas. Há também uma diferença
pequena de ROC-AUC da MLP em `r3_f4` (0,00012), com diferença máxima de apenas 1,11 × 10⁻¹⁶ nas
probabilidades: a ordenação de scores quase empatados é sensível a diferenças numéricas mínimas.

Apesar dessas diferenças, os rankings médios dos oito modelos, Friedman/Nemenyi e todos os p-valores
de Holm das comparações pareadas permaneceram iguais. Portanto, a reprodução sustenta as conclusões
estatísticas originais, sem demonstrar reprodução numérica exata entre plataformas. Os resultados novos
ficam em `.runtime/review/reproduction/`, a comparação em `.runtime/review/comparison.json` e os logs
de testes e validação em `.runtime/review/`; esses arquivos locais não são versionados.

### Melhorias aplicadas sem alterar o experimento

1. A capa agora reproduz pelo código a autoria, o mês e os ajustes de posição e fonte do arquivo entregue.
   Antes, gerar novamente a apresentação restaurava dados desatualizados e mudava o layout.
2. O gerador atualiza as contagens e os títulos dos slides nas propriedades do PowerPoint, aceita
   `--output` e só substitui a exportação anterior depois de concluir o empacotamento.
3. Os slides mostram a suavização da taxa positiva e o fallback dos pesos, distinguem as famílias de
   Holm e tratam M8 × M6 como efeito conjunto de calibração e competência. A lista de avisos da MLP
   inclui QSAR e Retinopathy (60 e 13 avisos na execução original).
4. A documentação inclui instalação com `.venv` no Linux e os limites da calibração interna descritos
   em [METODO.md](METODO.md).

### Melhorias para um próximo protocolo, não aplicadas retrospectivamente

| Prioridade | Proposta | Por que exige uma nova execução |
|---|---|---|
| Alta | Calibração das SVMs com grupos e pré-processamento por subdivisão | As cinco subdivisões de `CalibratedClassifierCV` não recebem os grupos da avaliação; mudar isso altera as probabilidades |
| Alta | Critério explícito e testado para empates dos vizinhos, com registro do ambiente numérico | A seleção de vizinhos equidistantes pode alterar previsões e pesos; mudar o desempate altera o modelo |
| Média | Ajuste cruzado adicional na seleção do combinador | Evita reutilização indireta de rótulos pela calibração das OOF que estimam competências em outras dobras |
| Média | Contrastes que separem calibração de competência e seleção interna do limiar de decisão | Permitem atribuir efeitos separadamente e investigar a classe minoritária, com novo plano e controle de multiplicidade |
