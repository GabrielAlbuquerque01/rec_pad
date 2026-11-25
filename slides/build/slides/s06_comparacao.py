# -*- coding: utf-8 -*-
"""Slide: comparacao principal entre bases (Friedman, Nemenyi e Wilcoxon).

Diagrama de diferenca critica desenhado com formas: eixo de ranking medio, cada modelo
ligado ao seu rotulo e barras vermelhas unindo modelos que o Nemenyi nao distingue."""
from .. import dados
from ..pptxkit import (DGRAY, GRAY, LGRAY, LINEG, MX, NAVY, RED, REDTINT, TXT, WHITE,
                       boxed, chrome, num, para, rect, reset_ids, run, seta, sld, tbox)

ID = "s06_comparacao"
TITULO = "Os híbridos lideram, sem se separar das SVMs e da MLP"
HIBRIDOS = {"M8", "M9", "M10"}
NOTAS = [
    "Esta é a comparação principal: os cinco classificadores e os três híbridos novos, com a "
    "ROC-AUC média de 15 folds em cada base.",
    "O Friedman rejeita a hipótese de que todos têm o mesmo desempenho.",
    "No diagrama, quanto mais à esquerda, melhor o ranking médio. Os três híbridos ficam nas "
    "três primeiras posições.",
    "As barras vermelhas ligam modelos que o Nemenyi não consegue separar. Os híbridos se separam "
    "do Naive Bayes e do KNN, mas não das SVMs nem da MLP.",
    "O Wilcoxon, que compara o M9 com cada modelo, mostra vitória em 8 das 10 bases contra as SVMs "
    "e a MLP, mas depois da correção de Holm para várias comparações nenhum p fica abaixo de 0,05. "
    "Com 10 bases o teste tem pouco poder.",
]


def build(page="06"):
    reset_ids()
    s = chrome("06 · RESULTADOS · COMPARAÇÃO ENTRE BASES", TITULO,
               "ROC-AUC média de 15 folds por base; Friedman com as 10 bases como blocos",
               page, RED)
    est = dados.estatistica()
    fr = est["friedman"]
    ranks = sorted(est["mean_ranks"].items(), key=lambda kv: kv[1])
    cd = fr.get("critical_difference")
    count = len(ranks)

    # ---------------- diagrama de diferenca critica
    PX, PW, PY, PH = MX, 5500000, 1400000, 2560000
    s.append(rect(PX, PY, PW, PH, WHITE, LINEG))
    x0, x1 = PX + 1800000, PX + PW - 1800000
    ax = PY + 620000

    def xr(r):
        return int(x0 + (r - 1) / (count - 1) * (x1 - x0))

    s.append(rect(x0, ax, x1 - x0, 12700, TXT, TXT))
    for r in range(1, count + 1):
        s.append(rect(xr(r) - 6350, ax - 70000, 12700, 70000, TXT, TXT))
        s.append(tbox(xr(r) - 150000, ax - 260000, 300000, 170000,
                      [para([run(str(r), sz=850, color=TXT)], algn="ctr")], ins=(0, 0, 0, 0)))
    if cd:
        s.append(rect(xr(1), PY + 200000, xr(1 + cd) - xr(1), 19050, TXT, TXT))
        s.append(tbox(xr(1), PY + 40000, xr(1 + cd) - xr(1), 160000,
                      [para([run(f"CD = {num(cd, 2)}", sz=850, b=1, color=TXT)], algn="ctr")],
                      ins=(0, 0, 0, 0)))

    half = (count + 1) // 2
    for pos, (modelo, rank) in enumerate(ranks):
        left = pos < half
        level = pos if left else count - 1 - pos
        y = ax + 480000 + level * 300000
        x = xr(rank)
        hib = modelo in HIBRIDOS
        cor = NAVY if hib else TXT
        s.append(seta(x, ax, x, y, cor, head=False, lw=9525))
        fim = PX + 80000 if left else PX + PW - 80000
        s.append(seta(x, y, fim, y, cor, head=False, lw=9525))
        rotulo = f"{modelo} · {dados.NOME_MODELO[modelo]} ({num(rank, 2)})"
        bx = PX + 80000 if left else x1 + 40000
        bw = x0 - 40000 - bx if left else PX + PW - 80000 - bx
        s.append(tbox(bx, y - 190000, bw, 180000,
                      [para([run(rotulo, sz=850, b=1 if hib else 0, color=cor)],
                            algn="l" if left else "r")], ins=(0, 0, 0, 0)))
    if cd:
        valores = [r for _, r in ranks]
        grupos = []
        for i in range(count):
            j = i
            while j + 1 < count and valores[j + 1] - valores[i] < cd:
                j += 1
            if j > i and not any(a <= i and j <= b for a, b in grupos):
                grupos.append((i, j))
        for lvl, (i, j) in enumerate(grupos):
            y = ax + 110000 + lvl * 90000
            s.append(rect(xr(valores[i]) - 40000, y, xr(valores[j]) - xr(valores[i]) + 80000,
                          38100, RED, RED))
    s.append(tbox(PX + 80000, PY + PH - 300000, PW - 160000, 250000,
                  [para([run("Ranking médio (1 = melhor). Modelos ligados por uma barra vermelha "
                             "não diferem pelo Nemenyi (α = 0,05).", sz=800, i=1, color=GRAY)])],
                  ins=(0, 0, 0, 0)))

    # ---------------- cartoes de estatistica
    nem_sig = []
    if fr["reject_null"]:
        import pandas as pd
        nem = pd.read_csv(dados.RESULTS / "nemenyi_pvalues.csv", index_col=0)
        nem_sig = [b for b in nem.columns if b not in HIBRIDOS
                   and all(nem.loc[a, b] < fr["alpha"] for a in HIBRIDOS)]
    controle = [dados.par("M9", m) for m in ("M1", "M2", "M3")]
    vitorias = sorted({c["wins"] for c in controle})
    pmin, pmax = min(c["wilcoxon_p"] for c in controle), max(c["wilcoxon_p"] for c in controle)
    holm_min = min(c["wilcoxon_p_holm"] for c in controle)
    expoente = int(f"{fr['pvalue']:e}".split("e")[1])
    mantissa = fr["pvalue"] / 10 ** expoente
    nomes = " e ".join(dados.NOME_MODELO[m] for m in nem_sig)
    cards = [
        ("FRIEDMAN", [run(f"χ² = {num(fr['statistic'], 2)} · p = {num(mantissa, 1)}·10", sz=1300,
                          b=1, color=NAVY), run(str(expoente).replace("-", "−"), sz=1300, b=1, color=NAVY, sup=True)],
         "Há diferença sistemática entre os rankings dos 8 modelos."),
        (f"NEMENYI · CD = {num(cd, 2)}", None,
         f"Os três híbridos superam {nomes}; as diferenças para SVMs e MLP ficam abaixo do CD."),
        ("WILCOXON + HOLM · M9 CONTRA CADA", None,
         f"Vence SVM linear, SVM-RBF e MLP em {'/'.join(map(str, vitorias))} de 10 bases; p bruto "
         f"de {num(pmin, 2)} a {num(pmax, 2)}, que sobe para {num(holm_min, 3)} ou mais após Holm."),
    ]
    CX, CW, CH, CG = MX + 5660000, 2843920, 780000, 110000
    for k, (cab, destaque, texto) in enumerate(cards):
        y = 1400000 + k * (CH + CG)
        s.append(rect(CX, y, CW, CH, LGRAY, LGRAY))
        s.append(rect(CX, y, 54864, CH, NAVY, NAVY))
        paras = [para([run(cab, sz=800, b=1, color=NAVY, spc=100)])]
        if destaque:
            paras.append(para(destaque, spcbef=200))
        paras.append(para([run(texto, sz=900, color=TXT)], lnspc=105000, spcbef=200))
        s.append(tbox(CX + 110000, y + 40000, CW - 170000, CH - 80000, paras, anchor="t",
                      ins=(0, 0, 0, 0)))

    s += boxed(MX, 4120000, 8503920, 560000, [
        para([run("Leitura: ", sz=950, b=1, color=RED),
              run("combinar os cinco modelos dá o melhor ranking médio, mas, com 10 bases, a "
                  "vantagem sobre os melhores modelos individuais não atinge significância.",
                  sz=950, color=TXT)], lnspc=105000)], REDTINT, REDTINT, ins=(91440, 0, 91440, 0))
    return sld(s)
