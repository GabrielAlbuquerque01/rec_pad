# -*- coding: utf-8 -*-
"""Slide: por que a ponderacao de performance da aula colapsa para a media simples.

Esquerda: tres evidencias. Direita: pesos medios em Retinopathy, formula da aula (M6) contra a
competencia corrigida (M8). A fracao em que o Naive Bayes decide sozinho vem da execucao
exploratoria (docs/ESTUDO_MELHORIAS.md, secao 5, F3)."""
from .. import dados
from ..pptxkit import (DGRAY, GRAY, LGRAY, LINEG, MUTED, MX, NAVY, RED, REDTINT, TINT, TXT, WHITE,
                       boxed, chrome, num, para, rect, reset_ids, run, seta, sld, tbox)

ID = "s04_ponto_de_partida"
TITULO = "A fórmula da aula vira média simples"
NB_SOZINHO_EXPLORATORIO = 0.0938
NOTAS = [
    "O ponto de partida foi a ponderação de performance da aula 09: o peso de cada modelo é a "
    "sua acurácia dividida pela soma das acurácias.",
    "O problema é que as acurácias ficam entre 0,6 e 1,0. Dividir uma pela outra dá pesos quase "
    "iguais, e o híbrido vira a média simples dos cinco modelos.",
    "Aplicar a mesma fórmula por cluster quase não muda nada: as decisões do híbrido local e do "
    "global praticamente coincidem.",
    "E um modelo que só acerta o que a classe majoritária já acertaria continua com cerca de 20% "
    "do peso. Na execução exploratória, o Naive Bayes, com probabilidades extremas, chegou a "
    "decidir sozinho contra os outros quatro.",
    "À direita, Retinopathy: em cinza, os pesos da fórmula da aula, todos perto de 0,20. Em azul, "
    "os pesos com a competência corrigida, que explico no próximo slide: SVM linear e MLP sobem, "
    "Naive Bayes e KNN quase somem.",
]


def build(page="04"):
    reset_ids()
    s = chrome("04 · PONTO DE PARTIDA", TITULO,
               "Ponderação de performance da aula 09 com a acurácia fora da amostra: "
               "w_m = α_m / Σ α_r", page, NAVY)

    diag = dados.diagnosticos().groupby("dataset").mean(numeric_only=True)
    faixa_min, faixa_max = diag.m6_weight_range.min(), diag.m6_weight_range.max()
    muda = diag.m7_changes_m6.max()
    cards = [
        (f"{num(faixa_min, 3)} a {num(faixa_max, 2)}",
         "diferença entre o maior e o menor peso do M6, conforme a base; o uniforme seria 0,20",
         NAVY),
        (f"até {num(100 * muda, 1)}%",
         "das decisões mudam quando a mesma fórmula é aplicada por cluster (M7 contra M6)", NAVY),
        (f"{round(100 * NB_SOZINHO_EXPLORATORIO)}%",
         "das instâncias de Retinopathy em que o Naive Bayes decide sozinho contra os outros "
         "quatro (execução exploratória)", RED),
    ]
    CX, CW, CH, CG = MX, 3300000, 820000, 110000
    for k, (valor, texto, cor) in enumerate(cards):
        y = 1400000 + k * (CH + CG)
        s.append(rect(CX, y, CW, CH, LGRAY, LGRAY))
        s.append(rect(CX, y, 54864, CH, cor, cor))
        s.append(tbox(CX + 120000, y, 1500000, CH,
                      [para([run(valor, sz=1700, b=1, color=cor)])], ins=(0, 0, 0, 0)))
        s.append(tbox(CX + 1660000, y, CW - 1720000, CH,
                      [para([run(texto, sz=900, color=TXT)], lnspc=105000)], ins=(0, 0, 0, 0)))

    # ---------------- grafico de pesos em Retinopathy
    GX, GW, GY, GH = 3780000, 5043960, 1400000, 2680000
    s.append(rect(GX, GY, GW, GH, TINT, TINT))
    s.append(tbox(GX + 100000, GY + 40000, GW - 200000, 200000,
                  [para([run("PESOS EM RETINOPATHY · MÉDIA DOS 15 FOLDS", sz=850, b=1, color=NAVY,
                             spc=150)])], ins=(0, 0, 0, 0)))
    lx = GX + 100000
    for cor, texto in [(MUTED, "M6 · fórmula da aula"), (NAVY, "M8 · competência corrigida")]:
        s.append(rect(lx, GY + 330000, 140000, 140000, cor, cor))
        s.append(tbox(lx + 180000, GY + 310000, 1700000, 180000,
                      [para([run(texto, sz=800, color=DGRAY)])], ins=(0, 0, 0, 0)))
        lx += 1750000
    s.append(seta(lx, GY + 400000, lx + 220000, GY + 400000, RED, dash="dash", head=False))
    s.append(tbox(lx + 260000, GY + 310000, 1000000, 180000,
                  [para([run("uniforme (0,20)", sz=800, color=DGRAY)])], ins=(0, 0, 0, 0)))

    m6 = dados.pesos_globais("retinopathy", "M6")
    m8 = dados.pesos_globais("retinopathy", "M8")
    base, escala = GY + GH - 420000, 4800000
    PW = (GW - 200000) // 5
    BW, BG = 300000, 40000
    # a linha do peso uniforme fica atras das barras para nao cortar os rotulos
    yu = base - int(0.2 * escala)
    s.append(seta(GX + 100000, yu, GX + GW - 100000, yu, RED, dash="dash", head=False))
    for k, modelo in enumerate(dados.MODELOS[:5]):
        gx = GX + 100000 + k * PW
        x0 = gx + (PW - (2 * BW + BG)) // 2
        for j, (valor, cor) in enumerate([(m6[modelo], MUTED), (m8[modelo], NAVY)]):
            h = int(valor * escala)
            x = x0 + j * (BW + BG)
            s.append(rect(x, base - h, BW, h, cor, cor))
            s.append(tbox(x, base - h + 30000, BW, 170000,
                          [para([run(num(valor, 2), sz=750, b=1, color=WHITE)], algn="ctr")],
                          ins=(0, 0, 0, 0)))
        s.append(tbox(gx, base + 40000, PW, 200000,
                      [para([run(dados.NOME_MODELO[modelo], sz=850, color=DGRAY)], algn="ctr")],
                      ins=(0, 0, 0, 0)))
    s.append(rect(GX + 100000, base, GW - 200000, 9144, LINEG, LINEG))

    s += boxed(MX, 4200000, 8503920, 540000, [
        para([run("Leitura: ", sz=950, b=1, color=RED),
              run("com acurácias parecidas, nenhum modelo pesa muito mais que outro, e quem só "
                  "acerta o que a classe majoritária acertaria ainda recebe cerca de 20%. É o "
                  "cenário do exemplo da aula 09: a fusão perde a especialização.",
                  sz=950, color=TXT)], lnspc=105000)], REDTINT, REDTINT, ins=(91440, 0, 91440, 0))
    return sld(s)
