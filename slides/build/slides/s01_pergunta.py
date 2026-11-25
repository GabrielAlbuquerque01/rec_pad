# -*- coding: utf-8 -*-
"""Slide: a pergunta do projeto em um diagrama.

Esquerda: ponderacao global, um vetor de pesos para todo o espaco. Direita: ponderacao
local, pesos proprios em cada regiao. Faixa inferior: o desenho experimental em palavras-chave.
As barras da direita sao ilustrativas; as da esquerda usam os pesos medios do M6.
"""
from .. import dados
from ..pptxkit import (BLUETINT, GRAY, LINEG, MUTED, MX, NAVY, RED, REDTINT, TINT, WHITE, DGRAY,
                       chrome, para, rect, reset_ids, run, seta, sld, tbox)

ID = "s01_pergunta"
TITULO = "Ponderar por região supera a ponderação global?"
NOTAS = [
    "A ideia de partida vem da aula de sistemas híbridos: combinar classificadores com vieses "
    "diferentes e dar mais peso a quem é mais competente.",
    "À esquerda, a ponderação global: um único vetor de pesos, estimado no treino inteiro, vale "
    "para qualquer instância.",
    "À direita, a pergunta do projeto: se o espaço de atributos for dividido em regiões, cada "
    "região pode ter os seus pesos, porque um modelo pode ser bom em uma parte do espaço e ruim "
    "em outra.",
    "Não presumimos que o local é melhor. A hipótese é testada em dez bases, com testes de "
    "Friedman, Nemenyi e Wilcoxon.",
]

REGIOES = [(0.34, 0.28, 0.20, 0.10, 0.08), (0.10, 0.16, 0.38, 0.06, 0.30),
           (0.22, 0.06, 0.26, 0.32, 0.14)]


def barras(s, x, base, largura, pesos, cor, escala):
    bw = largura // 9
    for k, w in enumerate(pesos):
        h = int(w * escala)
        s.append(rect(x + k * (bw + bw // 2), base - h, bw, h, cor, cor))


def build(page="01"):
    reset_ids()
    s = chrome("01 · PERGUNTA", TITULO,
               "Combinar cinco classificadores e testar, em vez de presumir, se a competência "
               "local ajuda", page, NAVY)

    # ---------------- esquerda: ponderacao global
    LX, LW = MX, 2500000
    BX, BW = LX + 350000, 1800000
    s.append(tbox(LX, 1380000, LW, 230000,
                  [para([run("PONDERAÇÃO GLOBAL", sz=900, b=1, color=MUTED, spc=250)])]))
    s.append(rect(BX, 1680000, BW, 380000, WHITE, LINEG))
    s.append(tbox(BX, 1680000, BW, 380000,
                  [para([run("Treino inteiro", sz=1100, b=1, color=MUTED)], algn="ctr")]))
    s.append(seta(BX + BW // 2, 2060000, BX + BW // 2, 2230000, LINEG))
    s.append(rect(BX, 2240000, BW, 720000, WHITE, LINEG))
    m6 = dados.pesos_globais("retinopathy", "M6").to_numpy()
    barras(s, BX + 450000, 2880000, 1200000, m6, MUTED, 2400000)
    s.append(seta(BX + BW // 2, 2960000, BX + BW // 2, 3230000, LINEG))
    s.append(rect(BX, 3240000, BW, 380000, WHITE, LINEG))
    s.append(tbox(BX, 3240000, BW, 380000,
                  [para([run("Fusão das probabilidades", sz=1000, b=1, color=MUTED)], algn="ctr")]))
    s.append(tbox(LX, 3700000, LW, 420000,
                  [para([run("Um único vetor de pesos para qualquer instância.", sz=900, i=1,
                             color=GRAY)], algn="ctr", lnspc=110000)], anchor="t"))

    s.append(rect(2860000, 2560000, 230000, 340000, LINEG, LINEG, prst="rightArrow"))

    # ---------------- direita: ponderacao local
    RX, RW = 3250000, 5474000
    s.append(rect(3170000, 1350000, 5654000, 2790000, TINT, TINT))
    s.append(tbox(RX, 1380000, RW, 230000,
                  [para([run("PONDERAÇÃO LOCAL · EM TESTE", sz=900, b=1, color=NAVY, spc=250)])]))
    TX, TW = RX + (RW - 3200000) // 2, 3200000
    s.append(rect(TX, 1680000, TW, 380000, NAVY, NAVY))
    s.append(tbox(TX, 1680000, TW, 380000, [
        para([run("Região de x: ", sz=1100, b=1, color=WHITE),
              run("cluster do K-Means ou k vizinhos", sz=1100, color=WHITE)], algn="ctr")]))

    CW, GAP = 1500000, (RW - 3 * 1500000) // 2
    centros = [RX + k * (CW + GAP) + CW // 2 for k in range(3)]
    s.append(seta(RX + RW // 2, 2060000, RX + RW // 2, 2160000, NAVY, head=False))
    s.append(seta(centros[0], 2160000, centros[-1], 2160000, NAVY, head=False))
    for k, (cx, pesos) in enumerate(zip(centros, REGIOES)):
        x = cx - CW // 2
        s.append(seta(cx, 2160000, cx, 2290000, NAVY))
        s.append(rect(x, 2300000, CW, 820000, BLUETINT, NAVY))
        s.append(tbox(x, 2320000, CW, 200000,
                      [para([run("Região %d · pesos próprios" % (k + 1), sz=850, b=1, color=NAVY)],
                            algn="ctr")], ins=(0, 0, 0, 0)))
        barras(s, x + 400000, 3050000, 900000, pesos, NAVY, 1300000)
        s.append(seta(cx, 3120000, cx, 3250000, RED, head=False))
    s.append(seta(centros[0], 3250000, centros[-1], 3250000, RED, head=False))
    s.append(seta(RX + RW // 2, 3250000, RX + RW // 2, 3370000, RED))
    s.append(rect(TX, 3380000, TW, 380000, RED, RED))
    s.append(tbox(TX, 3380000, TW, 380000,
                  [para([run("Fusão com os pesos da região de x", sz=1100, b=1, color=WHITE)],
                        algn="ctr")]))
    s.append(tbox(RX, 3790000, RW, 300000,
                  [para([run("Barras ilustrativas. Com uma única região, a ponderação local volta a "
                             "ser a global.", sz=900, i=1, color=GRAY)], algn="ctr")], anchor="t"))

    # ---------------- faixa do desenho experimental
    BY, BH = 4260000, 400000
    chips = ["10 bases UCI", "5 classificadores", "5 híbridos", "Friedman · Nemenyi"]
    x = MX
    for k, c in enumerate(chips):
        s.append(rect(x, BY, 1180000, BH, WHITE, LINEG))
        s.append(tbox(x, BY, 1180000, BH,
                      [para([run(c, sz=900, color=DGRAY)], algn="ctr", lnspc=100000)]))
        if k < 3:
            s.append(tbox(x + 1180000, BY, 120000, BH,
                          [para([run("+", sz=1000, b=1, color=MUTED)], algn="ctr")]))
        x += 1300000
    s.append(rect(5440040, BY + 110000, 160000, 200000, LINEG, LINEG, prst="rightArrow"))
    s.append(rect(5640040, BY, 3183920, BH, NAVY, NAVY))
    s.append(tbox(5640040, BY, 3183920, BH,
                  [para([run("Hipótese testada, não presumida", sz=1050, b=1, color=WHITE)],
                        algn="ctr")]))
    return sld(s)
