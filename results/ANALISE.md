# Resultados observados

Protocolo: 3 repetições (sementes 11, 22, 33) de 5 folds externos × 3 internos; 1500 registros e 100 agregados. Médias e desvios usam os 15 folds externos de cada base; os folds de repetições diferentes se sobrepõem e não são amostras independentes.

M1: SVM linear; M2: SVM-RBF; M3: MLP; M4: GaussianNB; M5: KNN; M6: Global (aula); M7: Local K-Means (original); M8: Global v2; M9: Local K-Means v2; M10: Local k-NN v2.

## ROC-AUC média (15 folds)

| Base | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | M10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sonar | 0.8216 | 0.9283 | 0.8885 | 0.7966 | 0.9223 | 0.9057 | 0.9063 | 0.9147 | 0.9173 | 0.9248 |
| ionosphere | 0.8724 | 0.9815 | 0.9323 | 0.9336 | 0.9214 | 0.9799 | 0.9799 | 0.9801 | 0.9824 | 0.9794 |
| banknote | 0.9997 | 1.0000 | 0.9999 | 0.9387 | 0.9989 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| blood | 0.7017 | 0.6963 | 0.7501 | 0.7078 | 0.6823 | 0.7293 | 0.7306 | 0.7496 | 0.7467 | 0.7407 |
| climate | 0.9581 | 0.9458 | 0.9531 | 0.9524 | 0.8031 | 0.9608 | 0.9606 | 0.9624 | 0.9594 | 0.9620 |
| qsar | 0.9256 | 0.9291 | 0.9244 | 0.8521 | 0.9054 | 0.9325 | 0.9326 | 0.9339 | 0.9324 | 0.9326 |
| raisin | 0.9315 | 0.9201 | 0.9271 | 0.9187 | 0.9086 | 0.9303 | 0.9304 | 0.9307 | 0.9305 | 0.9309 |
| rice | 0.9791 | 0.9655 | 0.9788 | 0.9747 | 0.9583 | 0.9770 | 0.9770 | 0.9774 | 0.9782 | 0.9775 |
| spambase | 0.9633 | 0.9723 | 0.9774 | 0.9278 | 0.9543 | 0.9808 | 0.9809 | 0.9809 | 0.9806 | 0.9808 |
| retinopathy | 0.8116 | 0.7683 | 0.8151 | 0.6927 | 0.6809 | 0.7976 | 0.7984 | 0.8220 | 0.8250 | 0.8196 |

Médias, desvios e contagens de todas as métricas em summary.csv; mapas em figures/.

## Friedman e Nemenyi (comparação principal)

Modelos: M1, M2, M3, M4, M5, M8, M9, M10; 10 bases como blocos; α = 0.05.
Friedman χ² = 37.4627; p = 3.83287e-06.

| Modelo | Ranking médio |
|---|---|
| M8 Global v2 | 2.55 |
| M9 Local K-Means v2 | 2.75 |
| M10 Local k-NN v2 | 2.75 |
| M3 MLP | 4.40 |
| M2 SVM-RBF | 4.75 |
| M1 SVM linear | 4.90 |
| M4 GaussianNB | 6.80 |
| M5 KNN | 7.10 |

H0 rejeitada. Nemenyi: q = 3.0309; CD = 3.3202 (figures/critical_difference.png).

Pares com diferença significativa: M4 × M8 (p = 0.002648); M4 × M9 (p = 0.005359); M4 × M10 (p = 0.005359); M5 × M8 (p = 0.0008571); M5 × M9 (p = 0.001835); M5 × M10 (p = 0.001835).

## M9 contra cada modelo da comparação principal

| Modelo | Referência | Δ AUC média | V / D / E | p Wilcoxon | p Holm | p sinal |
|---|---|---|---|---|---|---|
| M9 | M1 | +0.0288 | 8 / 2 / 0 | 0.01953 | 0.09766 | 0.1094 |
| M9 | M2 | +0.0145 | 8 / 1 / 1 | 0.03125 | 0.125 | 0.03906 |
| M9 | M3 | +0.0106 | 8 / 2 / 0 | 0.03711 | 0.125 | 0.1094 |
| M9 | M4 | +0.0557 | 10 / 0 / 0 | 0.001953 | 0.01367 * | 0.001953 |
| M9 | M5 | +0.0517 | 9 / 1 / 0 | 0.005859 | 0.03516 * | 0.02148 |
| M9 | M8 | +0.0001 | 4 / 5 / 1 | 0.9609 | 1 | 1 |
| M9 | M10 | +0.0004 | 4 / 5 / 1 | 0.8828 | 1 | 1 |

## Ablação: efeito de cada mudança

| Modelo | Referência | Δ AUC média | V / D / E | p Wilcoxon | p Holm | p sinal |
|---|---|---|---|---|---|---|
| M8 | M6 | +0.0058 | 9 / 0 / 1 | 0.003906 | 0.01172 * | 0.003906 |
| M9 | M7 | +0.0056 | 6 / 3 / 1 | 0.1445 | 0.2891 | 0.5078 |
| M10 | M8 | -0.0003 | 3 / 6 / 1 | 0.3477 | 0.3477 | 0.5078 |

Wilcoxon pareado por base (diferenças nulas divididas entre os sinais), correção de Holm dentro de cada família; * indica p ajustado < α. V/D/E: vitórias, derrotas e empates do primeiro modelo.

## Escolhas feitas na validação interna

Contagem de folds externos (15 por base) por valor escolhido. K = 1 em M9 equivale à ponderação global.

| Base | M9: K | M9: λ | M10: k | M10: λ |
|---|---|---|---|---|
| sonar | 3: 3, 4: 7, 5: 2, 6: 3 | 0: 8, 10: 5, 30: 2 | 7: 4, 15: 6, 31: 5 | 0: 9, 10: 6 |
| ionosphere | 1: 1, 2: 1, 3: 6, 4: 4, 5: 2, 6: 1 | 0: 12, 10: 2, 30: 1 | 7: 2, 15: 3, 31: 10 | 0: 5, 10: 7, 30: 3 |
| banknote | 1: 11, 2: 1, 4: 3 | 0: 1, 30: 14 | 7: 1, 31: 14 | 10: 2, 30: 13 |
| blood | 1: 3, 2: 2, 3: 2, 4: 3, 5: 3, 6: 2 | 0: 8, 10: 3, 30: 4 | 7: 5, 15: 4, 31: 6 | 0: 9, 10: 1, 30: 5 |
| climate | 1: 2, 2: 2, 3: 2, 4: 6, 5: 2, 6: 1 | 0: 7, 10: 3, 30: 5 | 7: 8, 15: 3, 31: 4 | 0: 8, 10: 1, 30: 6 |
| qsar | 1: 2, 2: 2, 3: 1, 5: 7, 6: 3 | 0: 6, 10: 2, 30: 7 | 7: 5, 15: 5, 31: 5 | 0: 4, 10: 4, 30: 7 |
| raisin | 2: 4, 3: 1, 4: 3, 5: 4, 6: 3 | 0: 10, 10: 3, 30: 2 | 7: 7, 15: 3, 31: 5 | 0: 9, 10: 5, 30: 1 |
| rice | 2: 6, 3: 6, 4: 3 | 0: 9, 10: 5, 30: 1 | 7: 2, 15: 2, 31: 11 | 10: 6, 30: 9 |
| spambase | 1: 13, 3: 1, 4: 1 | 0: 1, 10: 1, 30: 13 | 7: 13, 31: 2 | 10: 8, 30: 7 |
| retinopathy | 3: 11, 4: 2, 5: 1, 6: 1 | 0: 12, 10: 2, 30: 1 | 7: 3, 15: 1, 31: 11 | 0: 1, 10: 5, 30: 9 |

## Quanto os pesos locais se afastam dos globais

Faixa dos pesos globais (máximo − mínimo); fração das instâncias de teste em que a decisão local difere da global correspondente; deslocamento médio dos pesos locais em relação aos globais (metade da soma das diferenças absolutas, de 0 a 1).

| Base | faixa M6 | faixa M8 | M7≠M6 | M9≠M8 | M10≠M8 | desloc. M9 | desloc. M10 |
|---|---|---|---|---|---|---|---|
| sonar | 0.0363 | 0.1097 | 0.0032 | 0.0287 | 0.0431 | 0.2078 | 0.2368 |
| ionosphere | 0.0261 | 0.0706 | 0.0000 | 0.0019 | 0.0086 | 0.1316 | 0.1064 |
| banknote | 0.0332 | 0.0750 | 0.0000 | 0.0000 | 0.0000 | 0.0126 | 0.0385 |
| blood | 0.0112 | 0.5041 | 0.0040 | 0.0022 | 0.0085 | 0.1212 | 0.0798 |
| climate | 0.0061 | 0.3229 | 0.0000 | 0.0062 | 0.0043 | 0.1552 | 0.0868 |
| qsar | 0.0502 | 0.2370 | 0.0019 | 0.0047 | 0.0070 | 0.0951 | 0.1041 |
| raisin | 0.0080 | 0.0182 | 0.0007 | 0.0041 | 0.0093 | 0.1425 | 0.0698 |
| rice | 0.0027 | 0.0092 | 0.0002 | 0.0040 | 0.0014 | 0.2762 | 0.0182 |
| spambase | 0.0262 | 0.0750 | 0.0010 | 0.0012 | 0.0044 | 0.0424 | 0.0558 |
| retinopathy | 0.0353 | 0.2290 | 0.0043 | 0.0171 | 0.0281 | 0.1481 | 0.1740 |

## Diversidade dos modelos base (previsões OOF)

Oráculo: fração em que algum modelo acerta. Q de Yule médio entre pares: próximo de 1 indica erros coincidentes; próximo de 0, independentes.

| Base | oráculo | melhor individual | discordância | Q médio |
|---|---|---|---|---|
| sonar | 0.9387 | 0.8057 | 0.2219 | 0.7333 |
| ionosphere | 0.9694 | 0.9447 | 0.1135 | 0.8998 |
| banknote | 1.0000 | 0.9995 | 0.0688 | 0.3860 |
| blood | 0.8688 | 0.7800 | 0.1109 | 0.9301 |
| climate | 0.9722 | 0.9512 | 0.0379 | 0.9810 |
| qsar | 0.9561 | 0.8712 | 0.1821 | 0.7521 |
| raisin | 0.9094 | 0.8706 | 0.0580 | 0.9793 |
| rice | 0.9484 | 0.9290 | 0.0276 | 0.9920 |
| spambase | 0.9842 | 0.9297 | 0.1221 | 0.8013 |
| retinopathy | 0.9060 | 0.7265 | 0.2662 | 0.7083 |

## Agrupamentos

Roteador v2 (Yeo-Johnson + padronização): fração between/(within + between) média por K e, entre parênteses, o menor cluster observado. Roteador de M7 (padronização, K = 3): menor e maior cluster.

| Base | K=2 | K=3 | K=4 | K=5 | K=6 | M7: menor–maior |
|---|---|---|---|---|---|---|
| sonar | 0.14 (50) | 0.24 (35) | 0.29 (22) | 0.32 (16) | 0.35 (5) | 27–74 |
| ionosphere | 0.23 (133) | 0.30 (34) | 0.36 (26) | 0.39 (17) | 0.42 (16) | 31–156 |
| banknote | 0.40 (500) | 0.55 (279) | 0.65 (205) | 0.70 (111) | 0.73 (84) | 186–583 |
| blood | 0.47 (238) | 0.62 (121) | 0.71 (100) | 0.78 (62) | 0.81 (62) | 8–409 |
| climate | 0.05 (182) | 0.08 (113) | 0.10 (89) | 0.13 (70) | 0.15 (47) | 113–167 |
| qsar | 0.13 (375) | 0.21 (224) | 0.26 (31) | 0.31 (28) | 0.35 (20) | 8–558 |
| raisin | 0.46 (335) | 0.59 (179) | 0.66 (132) | 0.70 (64) | 0.73 (60) | 107–340 |
| rice | 0.47 (1316) | 0.57 (751) | 0.65 (642) | 0.69 (435) | 0.72 (312) | 798–1225 |
| spambase | 0.12 (967) | 0.21 (139) | 0.26 (133) | 0.29 (135) | 0.30 (84) | 21–2674 |
| retinopathy | 0.26 (420) | 0.42 (143) | 0.48 (124) | 0.53 (1) | 0.58 (1) | 7–514 |

## Avisos e métricas indefinidas

Avisos capturados: 73 {'ConvergenceWarning': 73} (warnings.json). Registros com métricas indefinidas: 13; causas na coluna undefined de metrics_by_fold.csv.

## Limitações

Dez bases limitam o poder dos testes entre bases. As repetições reduzem a variância de cada média, mas não criam novas bases. Hiperparâmetros dos modelos base são fixos. Calibração de Platt é uma aproximação paramétrica. A PCA perde dimensões; clusters não equivalem a classes. Dependências não identificáveis nos dados públicos permanecem uma limitação (docs/BASES.md).
