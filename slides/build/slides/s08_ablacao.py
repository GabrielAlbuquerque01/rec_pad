# -*- coding: utf-8 -*-
"""Slide: efeito de cada mudanca (ablacao), com o resultado base a base.

Cada comparacao mostra dez quadrados, um por base: vitoria, derrota ou empate do primeiro
modelo na ROC-AUC media. Ao lado, a diferenca media e o p de Wilcoxon corrigido por Holm."""
from .. import dados
from ..pptxkit import (ARROW, BLUETINT, DGRAY, GRAY, LGRAY, LINEG, MX, NAVY, RED, REDTINT, TXT,
                       WHITE, chrome, num, para, rect, reset_ids, run, sld, tbox)

ID = "s08_ablacao"
TITULO = "A correção funciona; a região não acrescenta"
NOTAS = [
    "Aqui está o efeito de cada mudança. Cada quadrado é uma base: azul quando o primeiro modelo "
    "ganha, vermelho quando perde, cinza quando empata.",
    "Primeira linha: a competência corrigida contra a fórmula da aula. Ganha em nove bases, empata "
    "em uma e não perde nenhuma. É o único resultado significativo depois da correção de Holm.",
    "Segunda linha: o local novo contra o local original. Melhora, mas sem significância.",
    "As duas últimas linhas respondem à pergunta do projeto: o local, por cluster ou por "
    "vizinhança, contra o global corrigido. O placar fica dividido, sem vantagem para o local.",
    "E não é por falta de atuação. Os pesos locais se afastam bastante dos globais e algumas "
    "decisões mudam; só que isso não vira ganho. Em Banknote e Spambase, a própria validação "
    "interna escolhe K = 1, ou seja, o global.",
]

LINHAS = [
    ("M8", "M6", "competência corrigida × fórmula da aula"),
    ("M9", "M7", "local v2 × local original"),
    ("M9", "M8", "local por cluster × global corrigido"),
    ("M10", "M8", "local por vizinhança × global corrigido"),
]
SIGLA = {"sonar": "SON", "ionosphere": "ION", "banknote": "BAN", "blood": "BLO", "climate": "CLI",
         "qsar": "QSA", "raisin": "RAI", "rice": "RIC", "spambase": "SPA", "retinopathy": "RET"}


def build(page="08"):
    reset_ids()
    s = chrome("08 · RESULTADOS · O QUE CADA MUDANÇA FEZ", TITULO,
               "Cada quadrado é uma base; Wilcoxon pareado entre bases, com correção de Holm",
               page, RED)

    LW, SQ, SG = 2350000, 220000, 45000
    QX = MX + LW + 100000
    QW = 10 * SQ + 9 * SG
    DX = QX + QW + 150000
    DW, PWD = 850000, 850000
    PX = DX + DW + 50000
    VX = PX + PWD + 100000
    VW = MX + 8503920 - VX

    y = 1400000
    for k, base in enumerate(dados.BASES):
        s.append(tbox(QX + k * (SQ + SG) - 30000, y + 120000, SQ + 60000, 180000,
                      [para([run(SIGLA[base], sz=650, b=1, color=DGRAY)], algn="ctr")],
                      ins=(0, 0, 0, 0)))
    for x, w, texto in [(DX, DW, "Δ ROC-AUC"), (PX, PWD, "p (Holm)"), (VX, VW, "LEITURA")]:
        s.append(tbox(x, y + 120000, w, 180000,
                      [para([run(texto, sz=750, b=1, color=NAVY, spc=50)], algn="ctr")],
                      ins=(0, 0, 0, 0)))

    RH, RG = 480000, 70000
    for r, (a, b, descricao) in enumerate(LINHAS):
        y = 1760000 + r * (RH + RG)
        row = dados.par(a, b)
        s.append(rect(MX, y, 8503920, RH, LGRAY, LGRAY))
        s.append(tbox(MX + 100000, y, LW, RH, [
            para([run(f"{a} × {b}", sz=1200, b=1, color=NAVY)], lnspc=95000),
            para([run(descricao, sz=850, color=GRAY)], lnspc=95000, spcbef=100)],
            ins=(0, 0, 0, 0)))
        for k, res in enumerate(dados.resultado_por_base(a, b)):
            cor = NAVY if res > 0 else (RED if res < 0 else ARROW)
            s.append(rect(QX + k * (SQ + SG), y + (RH - SQ) // 2, SQ, SQ, cor, cor))
        delta = row["mean_difference"]
        s.append(tbox(DX, y, DW, RH, [para([run(("+" if delta >= 0 else "−") + num(abs(delta), 4),
                                                sz=1200, b=1, color=TXT)], algn="ctr")],
                      ins=(0, 0, 0, 0)))
        p = row["wilcoxon_p_holm"]
        s.append(tbox(PX, y, PWD, RH, [para([run(num(p, 3), sz=1200, b=1,
                                                 color=NAVY if row["significant"] else TXT)],
                                            algn="ctr")], ins=(0, 0, 0, 0)))
        placar = f"{row['wins']}V · {row['losses']}D · {row['ties']}E"
        if row["significant"]:
            s.append(rect(VX, y + 90000, VW, RH - 180000, NAVY, NAVY))
            texto, cor = "significativo", WHITE
        else:
            s.append(rect(VX, y + 90000, VW, RH - 180000, WHITE, LINEG))
            texto, cor = "sem evidência", GRAY
        s.append(tbox(VX, y + 90000, VW, RH - 180000, [
            para([run(texto, sz=950, b=1, color=cor)], algn="ctr", lnspc=90000),
            para([run(placar, sz=750, color=cor)], algn="ctr", lnspc=90000)], ins=(0, 0, 0, 0)))

    y = 1760000 + 4 * (RH + RG)
    lx = QX
    for cor, texto in [(NAVY, "vitória do primeiro modelo"), (RED, "derrota"), (ARROW, "empate")]:
        s.append(rect(lx, y + 20000, 120000, 120000, cor, cor))
        s.append(tbox(lx + 160000, y, 1300000, 160000, [para([run(texto, sz=750, color=DGRAY)])],
                      ins=(0, 0, 0, 0)))
        lx += 1500000

    # ---------------- onde ganhou e por que o local nao acrescenta
    auc = dados.auc()
    ind = dados.MODELOS[:5]
    melhor_ind = {b: auc.loc[b, ind].max() for b in ("retinopathy", "blood")}
    diag = dados.diagnosticos()
    porbase = diag.groupby("dataset").mean(numeric_only=True)
    shift = max(porbase.m9_weight_shift.max(), porbase.m10_weight_shift.max())
    muda = max(porbase.m9_changes_m8.max(), porbase.m10_changes_m8.max())
    frac = dados.fracao_k_global()
    k1_txt = " e ".join(dados.NOME_BASE[b] for b in frac.index if frac[b] > 0.5)
    BY, BH, BW = 4200000, 560000, (8503920 - 160000) // 2
    s.append(rect(MX, BY, BW, BH, BLUETINT, BLUETINT))
    s.append(tbox(MX + 100000, BY, BW - 200000, BH, [
        para([run("Onde a correção ganhou: ", sz=900, b=1, color=NAVY),
              run(f"Retinopathy {num(auc.loc['retinopathy', 'M6'], 3)} → "
                  f"{num(auc.loc['retinopathy', 'M8'], 3)} (melhor individual "
                  f"{num(melhor_ind['retinopathy'], 3)}); Blood {num(auc.loc['blood', 'M6'], 3)} → "
                  f"{num(auc.loc['blood', 'M8'], 3)} (melhor individual "
                  f"{num(melhor_ind['blood'], 3)}).", sz=900, color=TXT)], lnspc=105000)],
        ins=(0, 0, 0, 0)))
    s.append(rect(MX + BW + 160000, BY, BW, BH, REDTINT, REDTINT))
    s.append(tbox(MX + BW + 260000, BY, BW - 200000, BH, [
        para([run("O local atua, mas não ganha: ", sz=900, b=1, color=RED),
              run(f"pesos deslocados até {round(100 * shift)}% e decisões alteradas até "
                  f"{round(100 * muda)}%; a validação interna volta a K = 1 em {k1_txt}.",
                  sz=900, color=TXT)], lnspc=105000)], ins=(0, 0, 0, 0)))
    return sld(s)
