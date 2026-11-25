# -*- coding: utf-8 -*-
"""Slide: as dez bases (tamanho e desbalanceamento) e o protocolo de avaliacao.

Dimensoes e classes vem de results/datasets.csv."""
from .. import dados
from ..pptxkit import (BLUETINT, DGRAY, GRAY, LGRAY, LINEG, MX, NAVY, RED, TXT, WHITE,
                       boxed, chrome, para, rect, reset_ids, run, sld, tbox)

ID = "s02_bases_protocolo"
TITULO = "Dez bases, validação aninhada e repetida"
NOTAS = [
    "As dez bases são binárias e públicas, do repositório UCI, com tamanhos de 208 a 4601 "
    "exemplos e de 4 a 60 atributos.",
    "Climate e Blood são as mais desbalanceadas, por isso a métrica principal é a ROC-AUC, que "
    "não depende do limiar de decisão.",
    "Linhas com atributos idênticos ficam sempre na mesma partição, para que o teste não veja "
    "cópias do treino.",
    "O protocolo é aninhado: o teste externo só mede. Tudo o que o híbrido aprende, inclusive a "
    "calibração, as competências e a escolha das regiões, sai das dobras internas.",
    "São 3 repetições de 5 folds, então cada base tem 15 medidas por modelo. Os testes usam as "
    "bases como blocos, nunca os folds.",
]

PASSOS = [
    ("3 repetições × 5 folds externos", "teste externo reservado, nunca usado em ajuste"),
    ("3 dobras internas em cada treino externo", "probabilidades fora da amostra dos 5 modelos"),
    ("Calibração, competências e escolha de K, k, λ", "somente com as dobras internas"),
    ("Reajuste no treino externo e teste", "os 10 modelos medidos nos mesmos folds"),
]


def build(page="02"):
    reset_ids()
    s = chrome("02 · BASES E PROTOCOLO", TITULO,
               "Tudo o que é ajustado vê apenas o treino; o teste externo serve só para medir",
               page, NAVY)

    # ---------------- tabela das bases
    cols = [(MX, 1250000), (MX + 1250000, 1050000), (MX + 2300000, 1650000)]
    heads = ["BASE", "n × p", "CLASSE MINORITÁRIA"]
    y = 1400000
    s.append(rect(MX, y, 3950000, 250000, NAVY, NAVY))
    for (x, w), h in zip(cols, heads):
        s.append(tbox(x + 60000, y, w - 60000, 250000,
                      [para([run(h, sz=750, b=1, color=WHITE, spc=100)])]))
    info = dados.bases()
    y += 250000
    for k, base in enumerate(dados.BASES):
        row = info.loc[base]
        n, p = int(row["samples"]), int(row["attributes"])
        minor = min(row["class_0"], row["class_1"]) / n
        s.append(rect(MX, y, 3950000, 260000, LGRAY if k % 2 == 0 else WHITE, None))
        s.append(tbox(cols[0][0] + 60000, y, cols[0][1], 260000,
                      [para([run(dados.NOME_BASE[base], sz=900, b=1, color=TXT)])]))
        s.append(tbox(cols[1][0] + 60000, y, cols[1][1], 260000,
                      [para([run(f"{n} × {p}", sz=900, color=TXT)])]))
        cor = RED if minor < 0.25 else NAVY
        largura = int(minor / 0.5 * 1100000)
        s.append(rect(cols[2][0] + 60000, y + 80000, largura, 100000, cor, cor))
        s.append(tbox(cols[2][0] + 100000 + largura, y, 450000, 260000,
                      [para([run(f"{round(100 * minor)}%", sz=850, b=1, color=cor)])]))
        y += 260000
    s.append(tbox(MX, y + 40000, 3950000, 330000, [
        para([run("Atributos idênticos ficam na mesma partição. Em vermelho, bases com menos de "
                  "25% na classe minoritária.", sz=800, i=1, color=GRAY)], lnspc=105000)],
        anchor="t", ins=(0, 0, 0, 0)))

    # ---------------- protocolo
    PX, PW = 4420000, 4403960
    s.append(tbox(PX, 1380000, PW, 230000,
                  [para([run("PROTOCOLO", sz=900, b=1, color=NAVY, spc=250)])]))
    for k, (head, body) in enumerate(PASSOS):
        y = 1680000 + k * 550000
        s.append(rect(PX, y, PW, 470000, LGRAY, LGRAY))
        s += boxed(PX, y, 420000, 470000, [para([run(str(k + 1), sz=1400, b=1, color=WHITE)],
                                                algn="ctr")], NAVY, NAVY)
        s.append(tbox(PX + 500000, y, PW - 540000, 470000, [
            para([run(head, sz=1000, b=1, color=NAVY)], lnspc=100000),
            para([run(body, sz=850, color=DGRAY)], lnspc=100000, spcbef=200)]))
    y = 1680000 + 4 * 550000 + 20000
    s.append(rect(PX, y, PW, 640000, BLUETINT, NAVY))
    s.append(tbox(PX + 100000, y, PW - 200000, 640000, [
        para([run("ESTATÍSTICA", sz=800, b=1, color=NAVY, spc=200)]),
        para([run("ROC-AUC média dos 15 folds por base. Friedman + Nemenyi com as bases como "
                  "blocos; Wilcoxon com correção de Holm e teste do sinal.", sz=900, color=TXT)],
             lnspc=105000, spcbef=200)]))
    return sld(s)
