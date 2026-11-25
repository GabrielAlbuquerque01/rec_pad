# -*- coding: utf-8 -*-
"""Slide: conclusoes do experimento e limitacoes."""
from .. import dados
from ..pptxkit import (DGRAY, GRAY, LGRAY, MX, NAVY, RED, TXT, WHITE,
                       boxed, chrome, num, para, rect, reset_ids, run, sld, tbox)

ID = "s09_conclusoes"
TITULO = "O que o experimento responde"
NOTAS = [
    "Três conclusões, todas restritas a estas dez bases e ao protocolo fixado antes de rodar.",
    "Primeiro: medir a competência acima da classe majoritária, com probabilidades calibradas, "
    "supera a fórmula da aula, com significância.",
    "Segundo: os híbridos ficam no topo do ranking e superam Naive Bayes e KNN, mas, com dez bases, "
    "não dá para afirmar que superam as SVMs e a MLP.",
    "Terceiro, a resposta à pergunta do projeto: não há evidência de que ponderar por região "
    "supere a ponderação global corrigida. Isso não prova que sejam equivalentes, e desta vez o "
    "resultado vem do experimento, não de um defeito do desenho.",
    "A principal limitação é o limiar de 0,5, que deixa a classe minoritária mal atendida em Blood "
    "e Climate. O passo natural seria escolher o limiar na validação interna.",
]


def build(page="09"):
    reset_ids()
    s = chrome("09 · CONCLUSÕES", TITULO,
               "Conclusões restritas às 10 bases e ao protocolo fixado antes da execução", page, NAVY)
    correcao = dados.par("M8", "M6")
    local = [dados.par("M9", "M8"), dados.par("M10", "M8")]
    cards = [
        ("A competência corrigida supera a fórmula da aula",
         f"M8 × M6: {correcao['wins']} vitórias, {correcao['losses']} derrotas, {correcao['ties']} "
         f"empate; p de Holm = {num(correcao['wilcoxon_p_holm'], 3)}."),
        ("Os híbridos lideram o ranking",
         "Superam Naive Bayes e KNN. Diante das SVMs e da MLP, a vantagem não é significativa "
         "com 10 bases."),
        ("Sem evidência de vantagem local",
         "Contra o global corrigido, " + " e ".join(
             f"{r['model']} tem {r['wins']} vitórias e {r['losses']} derrotas" for r in local)
         + ". O mecanismo atua, mas não ganha; isso não demonstra equivalência."),
    ]
    CW, CH, CG = 5200000, 760000, 100000
    for k, (titulo, texto) in enumerate(cards):
        y = 1400000 + k * (CH + CG)
        s.append(rect(MX, y, CW, CH, LGRAY, LGRAY))
        s += boxed(MX, y, 560000, CH, [para([run(str(k + 1), sz=2400, b=1, color=WHITE)],
                                            algn="ctr")], NAVY, NAVY)
        s.append(tbox(MX + 680000, y, CW - 760000, CH, [
            para([run(titulo, sz=1150, b=1, color=NAVY)], lnspc=100000),
            para([run(texto, sz=900, color=TXT)], lnspc=105000, spcbef=300)], ins=(0, 0, 0, 0)))

    sens = dados.media("blood", "M8", "sensitivity")
    espec = dados.media("climate", "M8", "specificity")
    limites = [
        f"Limiar de 0,5: classe minoritária mal atendida (sensibilidade {num(sens, 2)} em Blood, "
        f"especificidade {num(espec, 2)} em Climate).",
        "Dez bases limitam o poder dos testes entre bases.",
        "Hiperparâmetros dos modelos base fixos.",
        "MLP sem convergência em QSAR.",
    ]
    LX = MX + CW + 160000
    LW = MX + 8503920 - LX
    LH = 3 * CH + 2 * CG
    s.append(rect(LX, 1400000, LW, LH, LGRAY, LGRAY))
    s.append(rect(LX, 1400000, 54864, LH, RED, RED))
    paras = [para([run("LIMITAÇÕES", sz=850, b=1, color=RED, spc=150)])]
    paras += [para([run(t, sz=1000, color=TXT)], lnspc=105000, spcbef=700, bullet=True,
                   marL=150000, indent=-150000) for t in limites]
    paras.append(para([run("Próximo passo: escolher o limiar de decisão na validação interna.",
                           sz=950, i=1, color=DGRAY)], lnspc=105000, spcbef=1200))
    s.append(tbox(LX + 130000, 1460000, LW - 200000, LH - 120000, paras, anchor="t",
                  ins=(0, 0, 0, 0)))

    s += boxed(MX, 4140000, 8503920, 480000, [
        para([run("Em uma frase: ", sz=1100, b=1, color=WHITE),
              run("nestas bases, como a competência é medida importa mais do que onde ela é "
                  "medida.", sz=1100, color=WHITE)], algn="ctr")], NAVY, NAVY)
    return sld(s)
