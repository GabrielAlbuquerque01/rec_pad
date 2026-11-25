# -*- coding: utf-8 -*-
"""Capa: o slide 1 do template com os textos da disciplina e do projeto."""
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
    ("maio", "Novembro"),
    ("Integrantes: Nome 1 · Nome 2 · Nome 3",
     "Aluno: Mário · Professor: Prof. Dr. João Fausto Lorenzato"),
]


def build(cover_xml):
    out = cover_xml
    for antigo, novo in TEXTOS:
        alvo = "<a:t>%s</a:t>" % escape(antigo)
        if alvo not in out:
            raise ValueError("texto da capa nao encontrado no template: %r" % antigo)
        out = out.replace(alvo, "<a:t>%s</a:t>" % escape(novo), 1)
    return out
