"""Um modulo por slide.

Contrato de cada modulo:
    ID      str, igual ao nome do arquivo
    TITULO  str, para referencia humana
    NOTAS   list[str], vira notas do apresentador
    build(page: str) -> str, devolve o XML completo de <p:sld>

A capa e a excecao: CAPA = True e build(cover_xml) recebe o XML do slide 1 do template.
Numeros de resultados vem de build/dados.py, lidos da pasta results/ do repositorio.
"""
