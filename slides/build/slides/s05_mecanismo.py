# -*- coding: utf-8 -*-
"""Slide: as tres mudancas do combinador autoral (versao 2), no formato de etapas do template."""
from .. import dados
from ..pptxkit import (DGRAY, GRAY, LGRAY, LINEG, MX, NAVY, RED, TXT, WHITE,
                       boxed, chrome, num, para, rect, reset_ids, run, sld, tbox)

ID = "s05_mecanismo"
TITULO = "Três mudanças no combinador"
NOTAS = [
    "O mecanismo autoral muda três coisas na fórmula da aula, e todas são estimadas só com as "
    "dobras internas.",
    "Primeiro, a calibração. Cada modelo passa por uma regressão logística sobre o logit da sua "
    "probabilidade. Isso busca reduzir a influência de probabilidades superconfiantes, sem garantir "
    "calibração perfeita. A própria aula diz "
    "que a confiança só vale se as saídas forem calibradas.",
    "Segundo, e é o centro do trabalho: a competência passa a ser a acurácia acima da classe "
    "majoritária da região. É o critério da aula de pré-processamento: em dados desbalanceados, "
    "uma acurácia só é aceitável se superar a da classe majoritária. Quem não supera recebe peso "
    "zero, salvo o fallback: se nenhum supera, usam-se os pesos globais; se nenhum supera "
    "globalmente, pesos iguais. Para regiões pequenas, λ suaviza tanto a acurácia quanto a taxa positiva.",
    "Terceiro, a região não é mais fixa. O M9 testa K de 1 a 6 no K-Means, o M10 testa "
    "vizinhanças de 7, 15 ou 31 exemplos, e λ entre 0, 10 e 30. Vence a configuração com maior "
    "ROC-AUC fora da amostra. Como K = 1 é o global, o local só atua quando há sinal.",
]

CARDS = [
    ("01", "Calibração", NAVY, [
        ("Platt por modelo, sobre o logit da probabilidade.", 0),
        ("p_cal = σ(α · logit p + β)", 1),
        ("α e β ajustados nas probabilidades fora da amostra, em ajuste cruzado.", 0),
        ("Busca reduzir a influência de probabilidades superconfiantes.", 2),
    ]),
    ("02", "Competência acima da classe majoritária", RED, [
        ("â = (acertos + λ · a_global) / (n_R + λ)", 1),
        ("π̂_R = (positivos + λ · π_global) / (n_R + λ)", 1),
        ("base_R = max(π̂_R, 1 − π̂_R)", 1),
        ("w_m = max(â_m − base_R, 0), normalizado", 1),
        ("Sem excesso positivo: pesos globais; se também não houver excesso global, pesos iguais.", 2),
    ]),
    ("03", "Região escolhida na validação interna", NAVY, [
        ("M8: treino inteiro", 0),
        ("M9: cluster do K-Means, K de 1 a 6", 0),
        ("M10: k vizinhos, k = 7, 15 ou 31", 0),
        ("λ ∈ {0, 10, 30}. Vence a maior ROC-AUC fora da amostra. K = 1 é o global: o local só "
         "atua se houver sinal.", 2),
    ]),
]


def efeitos():
    """Uma evidencia dos resultados para cada mudanca, na ordem dos cartoes."""
    antes, depois = dados.probabilidade_extrema("qsar", dados.MODELOS.index("M4"))
    m6, m8 = dados.pesos_globais("blood", "M6"), dados.pesos_globais("blood", "M8")
    frac = dados.fracao_k_global()
    globais = [b for b in frac.index if frac[b] > 0.5]
    nomes = " e ".join(dados.NOME_BASE[b] for b in globais)
    return [
        [run("QSAR: o Naive Bayes dava probabilidade acima de 0,99 em ", sz=900),
         run(f"{round(100 * antes)}%", sz=900, b=1, color=NAVY),
         run(" das instâncias; depois da calibração, em ", sz=900),
         run(f"{round(100 * depois)}%", sz=900, b=1, color=NAVY), run(".", sz=900)],
        [run("Blood: SVM linear e KNN caem para ", sz=900),
         run(f"{num(m8['M1'], 2)} e {num(m8['M5'], 2)}", sz=900, b=1, color=RED),
         run(f" de peso (eram {num(m6['M1'], 2)}); a MLP sobe para ", sz=900),
         run(num(m8["M3"], 2), sz=900, b=1, color=RED), run(".", sz=900)],
        [run("K = 1 (global) ", sz=900, b=1, color=NAVY),
         run(f"na maioria dos folds de {nomes}; nas outras {len(frac) - len(globais)} bases, ",
             sz=900),
         run("K ≥ 2", sz=900, b=1, color=NAVY), run(" na maioria.", sz=900)],
    ]


def build(page="05"):
    reset_ids()
    s = chrome("05 · MECANISMO AUTORAL", TITULO,
               "Código próprio, sem combinadores prontos; tudo estimado nas dobras internas de "
               "cada fold", page, NAVY)

    GAP = 160000
    CW = (8503920 - 2 * GAP) // 3
    HY, HH, BY, BH = 1400000, 460000, 1860000, 2280000
    for k, (numero, titulo, cor, itens) in enumerate(CARDS):
        x = MX + k * (CW + GAP)
        s.append(rect(x, HY, CW, HH, cor, cor))
        s.append(tbox(x + 100000, HY, CW - 700000, HH,
                      [para([run(titulo, sz=1050, b=1, color=WHITE)], lnspc=95000)],
                      ins=(0, 0, 0, 0)))
        s.append(tbox(x + CW - 600000, HY, 520000, HH,
                      [para([run(numero, sz=2000, b=1, color=WHITE)], algn="r")], ins=(0, 0, 0, 0)))
        s.append(rect(x, BY, CW, BH, LGRAY, LGRAY))
        paras = []
        for texto, estilo in itens:
            if estilo == 1:
                paras.append(para([run(texto, sz=950, b=1, color=cor)], lnspc=100000, spcbef=500))
            elif estilo == 2:
                paras.append(para([run(texto, sz=900, i=1, color=DGRAY)], lnspc=105000,
                                  spcbef=900))
            else:
                paras.append(para([run(texto, sz=950, color=TXT)], lnspc=105000, spcbef=500,
                                  bullet=True, marL=150000, indent=-150000))
        s.append(tbox(x + 120000, BY + 60000, CW - 240000, BH - 120000, paras, anchor="t",
                      ins=(0, 0, 0, 0)))
        EY, EH = BY + BH - 700000, 620000
        s.append(rect(x + 80000, EY, CW - 160000, EH, WHITE, cor))
        s.append(tbox(x + 160000, EY, CW - 320000, EH, [
            para([run("EFEITO OBSERVADO", sz=750, b=1, color=cor, spc=100)]),
            para(efeitos()[k], lnspc=100000, spcbef=200)], ins=(0, 0, 0, 0)))

    s += boxed(MX, 4280000, 8503920, 440000, [
        para([run("Em uma frase: ", sz=1050, b=1, color=WHITE),
              run("o peso passa a medir o quanto o modelo acerta além do óbvio, na região de "
                  "cada instância.", sz=1050, color=WHITE)], algn="ctr")], NAVY, NAVY)
    return sld(s)
