# Resultados observados

Um experimento 5×3, semente 42, sem otimização. M1: SVM linear; M2: SVM-RBF; M3: MLP; M4: GaussianNB; M5: KNN; M6: híbrido global; M7: híbrido local.

350 avaliações de teste e 70 agregados. Desvios-padrão amostrais entre cinco folds (ddof=1), não intervalos de confiança. Veja summary.csv para todas as métricas e contagens válidas.

| Base | M1 | M2 | M3 | M4 | M5 | M6 | M7 |
|---|---|---|---|---|---|---|---|
| sonar | 0.8221 | 0.9363 | 0.8889 | 0.8001 | 0.9268 | 0.9141 | 0.9151 |
| ionosphere | 0.8441 | 0.9812 | 0.9209 | 0.9387 | 0.9097 | 0.9830 | 0.9822 |
| banknote | 0.9998 | 1.0000 | 0.9998 | 0.9380 | 0.9987 | 1.0000 | 1.0000 |
| blood | 0.7172 | 0.6963 | 0.7560 | 0.7235 | 0.6740 | 0.7271 | 0.7269 |
| climate | 0.9535 | 0.9449 | 0.9442 | 0.9548 | 0.7916 | 0.9543 | 0.9543 |
| qsar | 0.9251 | 0.9295 | 0.9174 | 0.8471 | 0.9048 | 0.9333 | 0.9334 |
| raisin | 0.9313 | 0.9233 | 0.9250 | 0.9176 | 0.9157 | 0.9338 | 0.9337 |
| rice | 0.9792 | 0.9658 | 0.9789 | 0.9747 | 0.9560 | 0.9769 | 0.9769 |
| spambase | 0.9619 | 0.9721 | 0.9764 | 0.9210 | 0.9545 | 0.9799 | 0.9800 |
| retinopathy | 0.8118 | 0.7624 | 0.8044 | 0.6955 | 0.6754 | 0.7940 | 0.7944 |

Tabela: ROC-AUC média por base.

Friedman: estatística 25.762162; p=0.00024653708; alfa=0,05.

| Modelo | Ranking médio |
|---|---|
| M7 | 2.350 |
| M6 | 2.350 |
| M3 | 3.700 |
| M2 | 4.000 |
| M1 | 4.100 |
| M4 | 5.400 |
| M5 | 6.100 |

H0 rejeitada. Nemenyi: q=2.948320 (amplitude studentizada / √2); CD=2.848348. P-valores completos em nemenyi_pvalues.csv.
- M7 versus M1: ROC-AUC maior em 8/10 bases, menor em 2/10; p de Nemenyi=0.540215. Sem evidência de diferença pelo pós-teste; isso não demonstra equivalência.
- M7 versus M2: ROC-AUC maior em 8/10 bases, menor em 1/10; p de Nemenyi=0.610712. Sem evidência de diferença pelo pós-teste; isso não demonstra equivalência.
- M7 versus M3: ROC-AUC maior em 7/10 bases, menor em 3/10; p de Nemenyi=0.803522. Sem evidência de diferença pelo pós-teste; isso não demonstra equivalência.
- M7 versus M4: ROC-AUC maior em 9/10 bases, menor em 1/10; p de Nemenyi=0.0266642. Diferença significativa nos rankings.
- M7 versus M5: ROC-AUC maior em 9/10 bases, menor em 1/10; p de Nemenyi=0.00200115. Diferença significativa nos rankings.
- M7 versus M6: ROC-AUC maior em 4/10 bases, menor em 4/10; p de Nemenyi=1. Sem evidência de diferença pelo pós-teste; isso não demonstra equivalência.

Comparação descritiva M7 − M6 por base (sem testes adicionais):

| Base | Acurácia | Sensibilidade | Especificidade |
|---|---|---|---|
| sonar | +0.0048 | +0.0091 | +0.0000 |
| ionosphere | +0.0000 | +0.0000 | +0.0000 |
| banknote | +0.0000 | +0.0000 | +0.0000 |
| blood | -0.0013 | -0.0056 | +0.0000 |
| climate | +0.0000 | +0.0000 | +0.0000 |
| qsar | +0.0038 | +0.0084 | +0.0014 |
| raisin | +0.0000 | +0.0000 | +0.0000 |
| rice | +0.0000 | +0.0000 | +0.0000 |
| spambase | +0.0009 | +0.0011 | +0.0007 |
| retinopathy | +0.0000 | +0.0000 | +0.0000 |

Agrupamentos no treino externo: within é a soma dos quadrados das distâncias aos centroides; between é a soma ponderada dos quadrados das distâncias dos centroides à média geral. Não são índices de seleção de K.

| Base | Menor–maior cluster | Within médio por fold | Between médio por fold |
|---|---|---|---|
| banknote | 166–581 | 1938.98 | 2450.50 |
| blood | 7–411 | 1122.59 | 1271.01 |
| climate | 121–177 | 7173.29 | 602.71 |
| ionosphere | 33–153 | 6588.63 | 2677.77 |
| qsar | 12–604 | 28542.61 | 6061.39 |
| raisin | 110–340 | 2069.53 | 2970.47 |
| retinopathy | 9–501 | 10848.70 | 6646.19 |
| rice | 823–1215 | 9117.46 | 12217.57 |
| sonar | 26–72 | 7410.24 | 2573.76 |
| spambase | 23–2631 | 185116.19 | 24687.64 |

Avisos capturados: 425 (warnings.json). Registros com métricas indefinidas: 3; causas em metrics_by_fold.csv. Não foram alterados parâmetros após observar resultados.

Limitações: dez bases, uma semente e hiperparâmetros fixos limitam a generalização. Os folds são dependentes e não foram tratados como bases independentes. A projeção PCA perde dimensões; clusters não equivalem a classes. Probabilidades são estimativas sem garantia de calibração perfeita. Dependências não identificáveis nos dados públicos permanecem uma limitação (docs/BASES.md).
