# -*- coding: utf-8 -*-
"""Slide: arquitetura do sistema hibrido (versao 2), desenhada com formas editaveis.

Duas trilhas de inferencia: a dos modelos (pre-processamento, classificadores, calibracao)
e a do roteador (regiao, competencia). As duas se encontram na fusao. A faixa inferior
mostra o que e estimado no treinamento. Etapas delimitadas por linha tracejada."""
from ..pptxkit import (BLUETINT, DGRAY, GRAY, LGRAY, LINEG, MX, NAVY, RED, REDTINT, TXT, WHITE,
                       chrome, para, rect, reset_ids, run, seta, sld, tbox)

ID = "s03_arquitetura"
TITULO = "Arquitetura do sistema híbrido"
NOTAS = [
    "A instância entra por duas trilhas.",
    "Na trilha de cima, ela é padronizada e passa pelos cinco classificadores, cada um com um viés "
    "diferente: margem linear, kernel RBF, rede neural, Naive Bayes e vizinhança. As cinco "
    "probabilidades são calibradas.",
    "Na trilha de baixo, um roteador sem rótulos coloca a instância numa região: o cluster do "
    "K-Means ou a vizinhança dos k exemplos mais próximos.",
    "Na etapa 5 sai o peso de cada modelo naquela região, e na etapa 6 a soma ponderada das "
    "probabilidades calibradas decide a classe.",
    "A faixa de baixo é o treinamento: as três dobras internas geram as probabilidades fora da "
    "amostra, de onde saem a calibração, as competências e a escolha de K, k e λ.",
    "O mesmo desenho produz os três híbridos novos; muda só a definição de região.",
]

TOP, MID, BOT = 1380000, 2980000, 3900000          # trilhas: modelos [TOP, MID), roteador [MID, BOT)
LANE1, LANE2 = 2140000, 3440000                    # centros das trilhas
SUP = (BLUETINT, NAVY)
UNSUP = (LGRAY, DGRAY)
OWN = (REDTINT, RED)
DATA = (WHITE, LINEG)


def etapa(s, x, y, w, h, titulo):
    s.append(rect(x, y, w, h, None, NAVY, dash="dash", lw=9525))
    s.append(tbox(x + 50000, y + 30000, w - 100000, 170000,
                  [para([run(titulo, sz=700, b=1, color=NAVY, spc=50)])], ins=(0, 0, 0, 0)))


def caixa(s, x, y, w, h, linhas, estilo):
    fill, line = estilo
    s.append(rect(x, y, w, h, fill, line))
    paras = []
    for k, (texto, sz, bold, ital) in enumerate(linhas):
        cor = line if bold and line != LINEG else (GRAY if ital else TXT)
        paras.append(para([run(texto, sz=sz, b=bold, i=ital, color=cor)], algn="ctr",
                          lnspc=95000))
    s.append(tbox(x, y, w, h, paras, ins=(30000, 0, 30000, 0)))


def build(page="03"):
    reset_ids()
    s = chrome("03 · ARQUITETURA", TITULO,
               "Duas trilhas de inferência que se encontram na fusão; na faixa de baixo, o que é "
               "estimado no treinamento", page, NAVY)

    # ---------------- entrada
    s.append(rect(MX, 1900000, 480000, 1780000, NAVY, NAVY))
    s.append(tbox(MX, 1900000, 480000, 1780000, [
        para([run("Entrada", sz=850, b=1, color=WHITE)], algn="ctr"),
        para([run("x", sz=1100, i=1, color=WHITE)], algn="ctr")], ins=(0, 0, 0, 0)))

    # ---------------- 1. pre-processamento
    E1 = 920000
    etapa(s, E1, TOP, 1220000, BOT - TOP, "1 · PRÉ-PROCESSAMENTO")
    caixa(s, E1 + 60000, LANE1 - 250000, 1100000, 500000,
          [("Mediana, constantes", 800, 0, 0), ("padronização z-score", 800, 0, 0)], DATA)
    caixa(s, E1 + 60000, LANE2 - 250000, 1100000, 500000,
          [("Roteador", 850, 1, 0), ("Yeo-Johnson + z-score", 750, 0, 0)], UNSUP)
    s.append(seta(MX + 480000, LANE1, E1 + 60000, LANE1))
    s.append(seta(MX + 480000, LANE2, E1 + 60000, LANE2))

    # ---------------- 2. classificadores base
    E2 = 2260000
    etapa(s, E2, TOP, 1480000, MID - 80000 - TOP, "2 · CLASSIFICADORES BASE")
    modelos = ["SVM linear", "SVM-RBF", "MLP (32 tanh, SGD)", "Naive Bayes gaussiano",
               "KNN (k = 5, distância)"]
    ys = [1660000 + k * 245000 for k in range(5)]
    s.append(seta(E1 + 1160000, LANE1, E2 + 70000, LANE1, head=False))
    s.append(seta(E2 + 70000, ys[0] + 105000, E2 + 70000, ys[-1] + 105000, head=False))
    for y, nome in zip(ys, modelos):
        s.append(seta(E2 + 70000, y + 105000, E2 + 140000, y + 105000))
        caixa(s, E2 + 140000, y, 1240000, 210000, [(nome, 800, 0, 0)], SUP)
        s.append(seta(E2 + 1380000, y + 105000, E2 + 1430000, y + 105000, head=False))
    s.append(seta(E2 + 1430000, ys[0] + 105000, E2 + 1430000, ys[-1] + 105000, head=False))

    # ---------------- 3. calibracao
    E3 = 3860000
    etapa(s, E3, TOP, 1300000, MID - 80000 - TOP, "3 · CALIBRAÇÃO")
    s.append(seta(E2 + 1430000, LANE1, E3 + 70000, LANE1))
    caixa(s, E3 + 70000, LANE1 - 300000, 1160000, 600000,
          [("Platt por modelo", 850, 1, 0), ("p_cal = σ(α·logit p + β)", 750, 0, 0),
           ("α, β fora da amostra", 700, 0, 1)], OWN)

    # ---------------- 4. regiao
    etapa(s, E2, MID, 1480000, BOT - MID, "4 · REGIÃO (SEM RÓTULOS)")
    s.append(seta(E1 + 1160000, LANE2, E2 + 140000, LANE2))
    caixa(s, E2 + 140000, LANE2 - 250000, 1240000, 500000,
          [("Cluster do K-Means", 800, 1, 0), ("ou k vizinhos de x", 800, 1, 0)], UNSUP)

    # ---------------- 5. competencia e pesos
    etapa(s, E3, MID, 1300000, BOT - MID, "5 · COMPETÊNCIA E PESOS")
    s.append(seta(E2 + 1380000, LANE2, E3 + 70000, LANE2))
    caixa(s, E3 + 70000, LANE2 - 250000, 1160000, 500000,
          [("Acima da classe", 800, 1, 0), ("majoritária, suavizada", 800, 1, 0),
           ("→ pesos w_m(x)", 750, 0, 0)], OWN)

    # ---------------- 6. fusao
    E6 = 5280000
    etapa(s, E6, TOP, 1180000, BOT - TOP, "6 · FUSÃO E DECISÃO")
    s.append(seta(E3 + 1230000, LANE1, E6 + 70000, LANE1))
    s.append(seta(E3 + 1230000, LANE2, E6 + 70000, LANE2))
    caixa(s, E6 + 70000, 1900000, 1040000, 1780000,
          [("Soma ponderada", 850, 1, 0), ("s_j = Σ w_m · p_cal,mj", 750, 0, 0), ("", 400, 0, 0),
           ("classe =", 800, 0, 0), ("argmax s_j", 800, 1, 0)], OWN)

    # ---------------- coluna direita: variantes, saida e legenda
    CX, CW = 6580000, 2243960
    s.append(rect(CX, TOP, CW, 1020000, LGRAY, LGRAY))
    s.append(tbox(CX + 80000, TOP + 20000, CW - 160000, 1000000, [
        para([run("TRÊS HÍBRIDOS, UMA ARQUITETURA", sz=700, b=1, color=NAVY, spc=50)]),
        para([run("M8 ", sz=850, b=1, color=NAVY), run("região = treino inteiro", sz=850)],
             spcbef=300),
        para([run("M9 ", sz=850, b=1, color=NAVY), run("região = cluster do K-Means", sz=850)],
             spcbef=200),
        para([run("M10 ", sz=850, b=1, color=NAVY), run("região = k vizinhos de x", sz=850)],
             spcbef=200),
        para([run("Com K = 1, M9 coincide com M8.", sz=750, i=1, color=GRAY)], spcbef=300)],
        anchor="t", ins=(0, 0, 0, 0)))
    s.append(seta(E6 + 1110000, 2790000, CX, 2790000))
    s.append(rect(CX, 2540000, CW, 500000, NAVY, NAVY))
    s.append(tbox(CX, 2540000, CW, 500000,
                  [para([run("Classe prevista (0 ou 1)", sz=1000, b=1, color=WHITE)], algn="ctr")]))
    legenda = [(SUP, "Supervisionado"), (UNSUP, "Não supervisionado"), (OWN, "Mecanismo autoral"),
               (DATA, "Transformação de dados")]
    for k, ((fill, line), texto) in enumerate(legenda):
        y = 3180000 + k * 170000
        s.append(rect(CX, y + 20000, 200000, 120000, fill, line))
        s.append(tbox(CX + 260000, y, CW - 260000, 160000,
                      [para([run(texto, sz=750, color=DGRAY)])], ins=(0, 0, 0, 0)))
    y = 3180000 + 4 * 170000
    s.append(rect(CX, y + 20000, 200000, 120000, None, NAVY, dash="dash", lw=9525))
    s.append(tbox(CX + 260000, y, CW - 260000, 160000,
                  [para([run("Etapa", sz=750, color=DGRAY)])], ins=(0, 0, 0, 0)))

    # ---------------- faixa de treinamento
    TY, TH = 4000000, 860000
    s.append(rect(E1, TY, E6 + 1180000 - E1, TH, LGRAY, DGRAY, dash="dash", lw=9525))
    s.append(tbox(E1 + 50000, TY + 30000, 4000000, 170000,
                  [para([run("TREINAMENTO · EM CADA FOLD EXTERNO", sz=700, b=1, color=DGRAY,
                             spc=50)])], ins=(0, 0, 0, 0)))
    by, bh = TY + 260000, 460000
    caixa(s, E1 + 60000, by, 1100000, bh, [("Treino externo", 800, 1, 0),
                                            ("o teste não entra", 700, 0, 1)], DATA)
    caixa(s, E2 + 140000, by, 1240000, bh, [("5 modelos em", 800, 0, 0),
                                             ("3 dobras internas", 800, 1, 0)], DATA)
    caixa(s, E3 + 70000, by, 1160000, bh, [("Probabilidades", 800, 1, 0),
                                            ("fora da amostra", 800, 0, 0)], DATA)
    s.append(seta(E1 + 1160000, by + bh // 2, E2 + 140000, by + bh // 2))
    s.append(seta(E2 + 1380000, by + bh // 2, E3 + 70000, by + bh // 2))
    s.append(tbox(E6 + 60000, by - 40000, 1060000, bh + 80000, [
        para([run("Daqui saem α e β, os acertos por região e a escolha de K, k e λ.", sz=750,
                  i=1, color=DGRAY)], lnspc=100000)],
        anchor="ctr", ins=(0, 0, 0, 0)))
    for x, rotulo in [(E2 + 760000, "K ou k"), (E3 + 650000, "acertos · λ")]:
        s.append(seta(x, by, x, BOT, DGRAY, dash="dash"))
        s.append(tbox(x + 40000, TY + 70000, 700000, 150000,
                      [para([run(rotulo, sz=700, i=1, color=DGRAY)])], ins=(0, 0, 0, 0)))
    return sld(s)
