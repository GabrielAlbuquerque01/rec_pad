# -*- coding: utf-8 -*-
"""Slide: tabela de ROC-AUC por base e modelo, media e desvio-padrao dos folds externos.

A aula 06 pede tabelas com media e desvio; o melhor resultado de cada base fica em destaque."""
from .. import dados
from ..pptxkit import (BLUETINT, DGRAY, GRAY, LGRAY, MUTED, MX, NAVY, RED, TXT, WHITE,
                       chrome, num, para, rect, reset_ids, run, sld, tbox)

ID = "s07_auc_por_base"
TITULO = "ROC-AUC por base"
NOTAS = [
    "A tabela completa: média e desvio-padrão da ROC-AUC nos 15 folds de cada base. Em azul, o "
    "melhor de cada linha.",
    "Os híbridos novos ganham ou empatam em boa parte das bases, mas raramente com folga: em "
    "Sonar e Rice, por exemplo, uma SVM sozinha fica à frente.",
    "As bases mais difíceis, Blood e Retinopathy, são onde a correção da competência mais "
    "aparece em relação ao M6.",
    "Os desvios entre folds costumam ser maiores que as diferenças entre os híbridos, o que "
    "explica por que os testes não separam M8, M9 e M10.",
]

GRUPOS = [("INDIVIDUAIS", 0, 5, NAVY), ("FÓRMULA DA AULA", 5, 7, MUTED), ("VERSÃO 2", 7, 10, RED)]
CURTO = {"M1": "SVM linear", "M2": "SVM-RBF", "M3": "MLP", "M4": "Naive Bayes", "M5": "KNN",
         "M6": "Global", "M7": "Local", "M8": "Global", "M9": "K-Means", "M10": "k-NN"}


def build(page="07"):
    reset_ids()
    s = chrome("07 · RESULTADOS · POR BASE", TITULO,
               "Média ± desvio-padrão dos 15 folds externos; em azul, o melhor de cada base",
               page, RED)
    NW = 960000
    CW = (8503920 - NW) // len(dados.MODELOS)
    y = 1400000
    for rotulo, a, b, cor in GRUPOS:
        x = MX + NW + a * CW
        s.append(rect(x + 10000, y, (b - a) * CW - 20000, 200000, cor, cor))
        s.append(tbox(x, y, (b - a) * CW, 200000,
                      [para([run(rotulo, sz=700, b=1, color=WHITE, spc=100)], algn="ctr")],
                      ins=(0, 0, 0, 0)))
    y += 220000
    s.append(rect(MX, y, 8503920, 320000, LGRAY, LGRAY))
    s.append(tbox(MX + 60000, y, NW, 320000, [para([run("BASE", sz=750, b=1, color=NAVY,
                                                        spc=100)])], ins=(0, 0, 0, 0)))
    for k, modelo in enumerate(dados.MODELOS):
        s.append(tbox(MX + NW + k * CW, y, CW, 320000, [
            para([run(modelo, sz=850, b=1, color=NAVY)], algn="ctr", lnspc=90000),
            para([run(CURTO[modelo], sz=700, color=DGRAY)], algn="ctr", lnspc=90000)],
            ins=(0, 0, 0, 0)))
    y += 330000
    RH = 278000
    auc = dados.auc()
    for r, base in enumerate(dados.BASES):
        s.append(rect(MX, y, 8503920, RH, LGRAY if r % 2 else WHITE, None))
        s.append(tbox(MX + 60000, y, NW, RH, [para([run(dados.NOME_BASE[base], sz=900, b=1,
                                                        color=TXT)])], ins=(0, 0, 0, 0)))
        melhor = auc.loc[base].max()
        for k, modelo in enumerate(dados.MODELOS):
            m = dados.media(base, modelo, "roc_auc")
            d = dados.desvio(base, modelo, "roc_auc")
            top = num(m, 4) == num(melhor, 4)
            x = MX + NW + k * CW
            if top:
                s.append(rect(x + 15000, y + 15000, CW - 30000, RH - 30000, BLUETINT, NAVY))
            s.append(tbox(x, y, CW, RH, [
                para([run(num(m, 4), sz=900, b=1 if top else 0, color=NAVY if top else TXT)],
                     algn="ctr", lnspc=85000),
                para([run("± " + num(d, 3), sz=650, color=GRAY)], algn="ctr", lnspc=85000)],
                ins=(0, 0, 0, 0)))
        y += RH
    return sld(s)
