# -*- coding: utf-8 -*-
"""Monta um .pptx a partir de um manifesto JSON e do template oficial do PPGEC.

Cada slide e um modulo em build/slides/; o manifesto em decks/ lista quais entram e em
que ordem. Um modulo com CAPA = True recebe o XML do slide 1 do template e substitui os
textos, preservando a arte da capa (painel lateral e logotipo).

Uso, a partir da pasta slides/ e com o ambiente do projeto:
    python -m build.deck decks/apresentacao.json

O .pptx vai para exports/. A fonte de verdade e o codigo: edicoes feitas a mao no
PowerPoint se perdem se o arquivo for gerado de novo por cima.
"""
import argparse
import importlib
import json
import pathlib
import re
import shutil
import tempfile
import zipfile
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

HERE = pathlib.Path(__file__).resolve().parent
SLIDES_DIR = HERE.parent
EXPORTS = SLIDES_DIR / "exports"

TEMPLATE_SLIDES = 7  # slides que o template traz; acima disso as partes sao clonadas
MAX_SLIDES = 40

SLIDE_CT = "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
NOTES_CT = "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"


def slide_rid(i):
    """Id da relacao presentation -> slide i. Os slides do template usam rId2 a rId8; os
    clonados recebem ids fora da faixa ja ocupada pelo template."""
    return f"rId{i + 1}" if i <= TEMPLATE_SLIDES else f"rId{1000 + i}"


def clone_slide_parts(tmp, i):
    """Cria as partes de um slide alem dos que o template traz, copiando as do ultimo."""
    last = TEMPLATE_SLIDES
    sl = tmp / "ppt" / "slides" / "_rels"
    nt = tmp / "ppt" / "notesSlides"
    (sl / f"slide{i}.xml.rels").write_text(
        (sl / f"slide{last}.xml.rels").read_text(encoding="utf-8")
        .replace(f"notesSlide{last}.xml", f"notesSlide{i}.xml"), encoding="utf-8")
    (nt / "_rels" / f"notesSlide{i}.xml.rels").write_text(
        (nt / "_rels" / f"notesSlide{last}.xml.rels").read_text(encoding="utf-8")
        .replace(f"slides/slide{last}.xml", f"slides/slide{i}.xml"), encoding="utf-8")

    prels = tmp / "ppt" / "_rels" / "presentation.xml.rels"
    prels.write_text(prels.read_text(encoding="utf-8").replace(
        "</Relationships>",
        f'<Relationship Id="{slide_rid(i)}" Type="http://schemas.openxmlformats.org/'
        f'officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        "</Relationships>"), encoding="utf-8")

    ct = tmp / "[Content_Types].xml"
    ct.write_text(ct.read_text(encoding="utf-8").replace(
        "</Types>",
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="{SLIDE_CT}"/>'
        f'<Override PartName="/ppt/notesSlides/notesSlide{i}.xml" ContentType="{NOTES_CT}"/>'
        "</Types>"), encoding="utf-8")


def clone_notes(src, i):
    """Notas de um slide clonado: sem o creationId do original e com o numero certo."""
    out = re.sub(r"<p:extLst>.*?</p:extLst>", "", src, flags=re.S)
    return re.sub(r'(type="slidenum">.*?<a:t>)\d+(</a:t>)', r"\g<1>%d\g<2>" % i, out,
                  flags=re.S)


def load_module(name):
    return importlib.import_module("build.slides." + name)


def set_title(tmp, title):
    core = tmp / "docProps" / "core.xml"
    core.write_text(re.sub(r"<dc:title>.*?</dc:title>", lambda _: f"<dc:title>{escape(title)}</dc:title>",
                           core.read_text(encoding="utf-8")), encoding="utf-8")


def update_properties(tmp, mods):
    """Mantém a contagem e os títulos coerentes com os slides realmente gerados."""
    path = tmp / "docProps" / "app.xml"
    ns = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
    vt = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"
    ET.register_namespace("", ns)
    ET.register_namespace("vt", vt)
    root = ET.fromstring(path.read_bytes())
    for name in ("Slides", "Notes"):
        root.find(f"{{{ns}}}{name}").text = str(len(mods))
    headings = root.find(f"{{{ns}}}HeadingPairs/{{{vt}}}vector")
    counts = headings.findall(f"{{{vt}}}variant/{{{vt}}}i4")
    titles = root.find(f"{{{ns}}}TitlesOfParts/{{{vt}}}vector")
    # O último grupo de propriedades do template contém os títulos dos slides.
    old_count = int(counts[-1].text)
    for child in list(titles)[len(titles) - old_count:]:
        titles.remove(child)
    counts[-1].text = str(len(mods))
    for mod in mods:
        ET.SubElement(titles, f"{{{vt}}}lpstr").text = mod.TITULO
    titles.set("size", str(len(titles)))
    path.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))


def build(manifest_path: pathlib.Path, output: pathlib.Path | None = None) -> pathlib.Path:
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    template = (SLIDES_DIR / man["template"]).resolve()
    mods = [load_module(n) for n in man["slides"]]
    n = len(mods)
    if not 1 <= n <= MAX_SLIDES:
        raise SystemExit(f"manifesto pede {n} slides; suportado: 1 a {MAX_SLIDES}")
    if any(getattr(m, "CAPA", False) for m in mods[1:]):
        raise SystemExit("a capa precisa ser o primeiro slide do manifesto")
    has_cover = getattr(mods[0], "CAPA", False)

    tmp = pathlib.Path(tempfile.mkdtemp())
    try:
        with zipfile.ZipFile(template) as z:
            z.extractall(tmp)
            notes_src = {
                i: z.read(f"ppt/notesSlides/notesSlide{i}.xml").decode("utf-8")
                for i in range(1, TEMPLATE_SLIDES + 1)
            }
            cover_src = z.read("ppt/slides/slide1.xml").decode("utf-8")
        for i in range(TEMPLATE_SLIDES + 1, n + 1):
            clone_slide_parts(tmp, i)
            notes_src[i] = clone_notes(notes_src[TEMPLATE_SLIDES], i)
            (tmp / "ppt" / "notesSlides" / f"notesSlide{i}.xml").write_text(
                notes_src[i], encoding="utf-8")

        # 1. escreve os slides pedidos
        for i, mod in enumerate(mods, start=1):
            page = f"{i - 1 if has_cover else i:02d}"
            xml = mod.build(cover_src) if getattr(mod, "CAPA", False) else mod.build(page=page)
            (tmp / "ppt" / "slides" / f"slide{i}.xml").write_text(xml, encoding="utf-8")
            notas = getattr(mod, "NOTAS", None)
            if notas:
                from .pptxkit import notes_xml
                out = notes_xml(notes_src[i], notas)
                (tmp / "ppt" / "notesSlides" / f"notesSlide{i}.xml").write_text(
                    out, encoding="utf-8")

        # 2. sem capa, o slide 1 do template deixa de usar a imagem do logotipo
        drop = []
        if not has_cover:
            rels1 = tmp / "ppt" / "slides" / "_rels" / "slide1.xml.rels"
            rels1.write_text(re.sub(
                r'<Relationship Id="[^"]+" Type="[^"]*relationships/image"[^>]*?/>', "",
                rels1.read_text(encoding="utf-8")), encoding="utf-8")
            drop += ["ppt/media/" + m.name for m in (tmp / "ppt" / "media").glob("*")]

        # 3. apara a lista de slides da apresentacao
        pres = tmp / "ppt" / "presentation.xml"
        ids = "".join(f'<p:sldId id="{255 + i}" r:id="{slide_rid(i)}"/>'
                      for i in range(1, n + 1))
        pres.write_text(re.sub(r"<p:sldIdLst>.*?</p:sldIdLst>",
                               f"<p:sldIdLst>{ids}</p:sldIdLst>",
                               pres.read_text(encoding="utf-8"), flags=re.S),
                        encoding="utf-8")

        # 4. remove partes nao usadas e suas referencias
        for i in range(n + 1, TEMPLATE_SLIDES + 1):
            drop += [f"ppt/slides/slide{i}.xml",
                     f"ppt/slides/_rels/slide{i}.xml.rels",
                     f"ppt/notesSlides/notesSlide{i}.xml",
                     f"ppt/notesSlides/_rels/notesSlide{i}.xml.rels"]
        for rel in drop:
            p = tmp / rel
            if p.exists():
                p.unlink()

        prels = tmp / "ppt" / "_rels" / "presentation.xml.rels"
        txt = prels.read_text(encoding="utf-8")
        for i in range(n + 1, TEMPLATE_SLIDES + 1):
            txt = re.sub(r'<Relationship Id="[^"]+"[^>]*Target="slides/slide%d\.xml"/>' % i,
                         "", txt)
        prels.write_text(txt, encoding="utf-8")

        ct = tmp / "[Content_Types].xml"
        txt = ct.read_text(encoding="utf-8")
        for rel in drop:
            txt = re.sub(r'<Override PartName="/%s"[^/]*/>' % re.escape(rel), "", txt)
        ct.write_text(txt, encoding="utf-8")

        set_title(tmp, man["nome"])
        update_properties(tmp, mods)

        # 5. empacota antes de substituir a exportação anterior
        out = pathlib.Path(output) if output is not None else EXPORTS / (manifest_path.stem + ".pptx")
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".pptx", delete=False) as staging:
            staged = pathlib.Path(staging.name)
        try:
            with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED) as z:
                for p in sorted(tmp.rglob("*")):
                    if p.is_file():
                        z.write(p, p.relative_to(tmp).as_posix())
            staged.replace(out)
        finally:
            staged.unlink(missing_ok=True)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--output", type=pathlib.Path, help="Arquivo de saída alternativo para revisão")
    args = ap.parse_args()
    out = build(pathlib.Path(args.manifest).resolve(), args.output)
    print(f"gerado: {out}")


if __name__ == "__main__":
    main()
