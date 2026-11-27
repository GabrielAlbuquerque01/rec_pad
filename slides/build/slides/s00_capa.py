# -*- coding: utf-8 -*-
"""Capa: o slide 1 do template com os textos da disciplina e do projeto."""
import re
from xml.sax.saxutils import escape

ID = "s00_capa"
TITULO = "Capa"
CAPA = True
NOTAS = [
    "Projeto da disciplina de Reconhecimento de Padrões: um sistema híbrido que combina cinco "
    "classificadores e pondera cada um pela competência na região do espaço de atributos.",
    "A pergunta é se essa ponderação local supera a ponderação global, e a resposta vem de "
    "um experimento em dez bases com testes de hipótese.",
]

# Texto original do template -> texto deste projeto (cada par e um <a:t> inteiro).
TEXTOS = [
    ("Processamento", "Reconhecimento"),
    ("Linguagem", "Padrões"),
    (" Natural", ""),
    ("2026.1", "2026.2"),
    ("Nome da Equipe", "Projeto da disciplina"),
    ("Título do Projeto de Pesquisa", "Classificação híbrida por competência local"),
    ("Subtítulo ou linha complementar do projeto (opcional)",
     "Fusão de cinco classificadores ponderada pela competência em regiões do espaço de atributos"),
    ("15 de ", ""),
    ("maio", "Dezembro"),
    ("Integrantes: Nome 1 · Nome 2 · Nome 3",
     "Grupo: Daniel Dias, Gabriel Albuquerque e Mário Guerra · Professor: Prof. Dr. João Fausto Lorenzato"),
]

# Ajustes da capa entregue, trazidos de volta ao código para não se perderem ao gerar.
LAYOUT = {
    "Classificação híbrida por competência local": (3182112, 1172736, 2400),
    "Fusão de cinco classificadores ponderada pela competência em regiões do espaço de atributos":
        (3244073, 2361131, 1100),
}


def build(cover_xml):
    out = cover_xml
    for antigo, novo in TEXTOS:
        alvo = "<a:t>%s</a:t>" % escape(antigo)
        if alvo not in out:
            raise ValueError("texto da capa nao encontrado no template: %r" % antigo)
        out = out.replace(alvo, "<a:t>%s</a:t>" % escape(novo), 1)

    def ajustar(match):
        shape = match.group()
        for texto, (x, y, tamanho) in LAYOUT.items():
            if f"<a:t>{escape(texto)}</a:t>" in shape:
                shape = re.sub(r'<a:off x="\d+" y="\d+"', f'<a:off x="{x}" y="{y}"', shape)
                shape = re.sub(r'\bsz="\d+"', f'sz="{tamanho}"', shape)
        return shape

    out = re.sub(r"<p:sp>.*?</p:sp>", ajustar, out, flags=re.S)
    return out
