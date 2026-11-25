# -*- coding: utf-8 -*-
"""Primitivas de desenho no idioma visual do Template_PPGEC_2026_1.pptx.

Coordenadas em EMU. Slide 9144000 x 5143500 (16:9, 10 x 5.625 pol).
Todas as constantes foram extraidas do template oficial do PPGEC. Nao invente cores
novas: se precisar de um tom, derive dos existentes e registre aqui.
"""
from xml.sax.saxutils import escape

NAVY = "003F7F"
RED = "C0152B"
GRAY = "888888"
DGRAY = "555555"
TXT = "3A3A3A"
LGRAY = "F4F4F4"
BLUETINT = "E8EFF8"
REDTINT = "FAEAEC"
WHITE = "FFFFFF"
ARROW = "DDDDDD"

W = 9144000
H = 5143500
MX = 320040                 # margem esquerda
MW = 8503920                # largura util

FONT = ('<a:latin typeface="Calibri" pitchFamily="34" charset="0"/>'
        '<a:ea typeface="Calibri" pitchFamily="34" charset="-122"/>'
        '<a:cs typeface="Calibri" pitchFamily="34" charset="-120"/>')

_id = [1]


def nid():
    _id[0] += 1
    return _id[0]


def rect(x, y, cx, cy, fill=None, line=None, prst="rect", dash="solid", lw=12700):
    i = nid()
    f = '<a:solidFill><a:srgbClr val="%s"/></a:solidFill>' % fill if fill else '<a:noFill/>'
    if line:
        l = ('<a:ln w="%d"><a:solidFill><a:srgbClr val="%s"/></a:solidFill>'
             '<a:prstDash val="%s"/></a:ln>' % (lw, line, dash))
    else:
        l = '<a:ln><a:noFill/></a:ln>'
    return (
        '<p:sp><p:nvSpPr><p:cNvPr id="%d" name="Shape %d"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
        '<p:spPr><a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm>'
        '<a:prstGeom prst="%s"><a:avLst/></a:prstGeom>%s%s</p:spPr>'
        '<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang="pt-BR"/></a:p></p:txBody>'
        '</p:sp>' % (i, i, x, y, cx, cy, prst, f, l))


def run(t, sz=1000, b=0, i=0, color=TXT, spc=None, sup=False):
    sp = ' spc="%d"' % spc if spc else ''
    if sup:
        sp += ' baseline="30000"'
    return ('<a:r><a:rPr lang="pt-BR" sz="%d"%s%s%s kern="0" dirty="0">'
            '<a:solidFill><a:srgbClr val="%s"/></a:solidFill>%s</a:rPr>'
            '<a:t xml:space="preserve">%s</a:t></a:r>' % (sz, ' b="1"' if b else '', ' i="1"' if i else '',
                                                          sp, color, FONT, escape(t)))


def para(runs, algn=None, lnspc=None, spcbef=None, marL=0, indent=0, bullet=False):
    pPr = '<a:pPr marL="%d" indent="%d"%s>' % (marL, indent, ' algn="%s"' % algn if algn else '')
    if lnspc:
        pPr += '<a:lnSpc><a:spcPct val="%d"/></a:lnSpc>' % lnspc
    if spcbef:
        pPr += '<a:spcBef><a:spcPts val="%d"/></a:spcBef>' % spcbef
    if bullet:
        pPr += ('<a:buClr><a:srgbClr val="%s"/></a:buClr><a:buSzPct val="80000"/>'
                '<a:buFont typeface="Arial" pitchFamily="34" charset="0"/>'
                '<a:buChar char="\u25aa"/>' % NAVY)
    else:
        pPr += '<a:buNone/>'
    pPr += '</a:pPr>'
    return '<a:p>%s%s</a:p>' % (pPr, ''.join(runs))


def tbox(x, y, cx, cy, paras, anchor="ctr", ins=None):
    i = nid()
    b = '<a:bodyPr wrap="square" rtlCol="0" anchor="%s"' % anchor
    if ins:
        b += ' lIns="%d" tIns="%d" rIns="%d" bIns="%d"' % ins
    b += '/>'
    return (
        '<p:sp><p:nvSpPr><p:cNvPr id="%d" name="Text %d"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
        '<p:spPr><a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln/></p:spPr>'
        '<p:txBody>%s<a:lstStyle/>%s</p:txBody></p:sp>'
        % (i, i, x, y, cx, cy, b, ''.join(paras)))


def boxed(x, y, cx, cy, paras, fill=None, line=None, anchor="ctr", ins=None, prst="rect"):
    """Retangulo com texto por cima, o par mais comum nos diagramas."""
    return [rect(x, y, cx, cy, fill, line, prst=prst), tbox(x, y, cx, cy, paras, anchor, ins)]


def chrome(eyebrow, title, subtitle, page, topbar=NAVY):
    s = []
    s.append(rect(0, 0, W, 164592, topbar, topbar))
    s.append(tbox(MX, 228600, MW, 256032,
                  [para([run(eyebrow, sz=900, b=1, color=NAVY, spc=300)])]))
    s.append(rect(0, 4956048, W, 109728, NAVY, NAVY))
    s.append(rect(0, 5065776, W, 68580, RED, RED))
    s.append(tbox(8503920, 4956048, 548640, 173736,
                  [para([run(page, sz=900, b=1, color=WHITE)], algn="ctr")]))
    s.append(rect(MX, 594360, 64008, 457200, RED, RED))
    s.append(tbox(475488, 566928, 8321040, 502920,
                  [para([run(title, sz=2200, b=1, color=NAVY)])]))
    if subtitle:
        s.append(tbox(475488, 1078992, 8321040, 256032,
                      [para([run(subtitle, sz=1100, i=1, color=GRAY)])]))
    return s


def sld(shapes):
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
        '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>'
        '<a:effectLst/></p:bgPr></p:bg><p:spTree>'
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        + ''.join(shapes) +
        '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')



# tons derivados, usados nos diagramas
MUTED = "9AA3AB"     # cinza de texto secundario (estado "hoje")
LINEG = "C9CFD6"     # linha e contorno neutro
TINT = "F6F8FB"      # fundo de destaque muito leve


def notes_xml(src: str, lines) -> str:
    """Injeta paragrafos no placeholder de corpo de um notesSlide do template."""
    body = "".join(
        '<a:p><a:r><a:rPr lang="pt-BR" dirty="0"/><a:t>%s</a:t></a:r></a:p>' % escape(t)
        for t in lines)
    return src.replace(
        '<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang="en-US" dirty="0"/></a:p></p:txBody>',
        '<p:txBody><a:bodyPr/><a:lstStyle/>%s</p:txBody>' % body, 1)


def reset_ids() -> None:
    """Zera o contador de shape ids. Chame no inicio de cada slide."""
    _id[0] = 1


def seta(x1, y1, x2, y2, color=DGRAY, dash="solid", lw=12700, head=True):
    """Linha reta de (x1, y1) a (x2, y2), com ponta de seta no destino. Usa conector,
    que continua editavel como linha no PowerPoint."""
    i = nid()
    flip = (' flipH="1"' if x2 < x1 else '') + (' flipV="1"' if y2 < y1 else '')
    tail = '<a:tailEnd type="triangle" w="med" len="med"/>' if head else ''
    return (
        '<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="%d" name="Connector %d"/><p:cNvCxnSpPr/><p:nvPr/>'
        '</p:nvCxnSpPr><p:spPr><a:xfrm%s><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm>'
        '<a:prstGeom prst="line"><a:avLst/></a:prstGeom>'
        '<a:ln w="%d"><a:solidFill><a:srgbClr val="%s"/></a:solidFill><a:prstDash val="%s"/>%s</a:ln>'
        '</p:spPr></p:cxnSp>'
        % (i, i, flip, min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1), lw, color, dash, tail))


def num(value, digits=4):
    """Numero com virgula decimal, no padrao do texto em portugues."""
    return ("%.*f" % (digits, value)).replace(".", ",")
