"""Build real PPTX packages in temporary directories, without touching exports/."""
import importlib.util
import json
import posixpath
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
# `build` is the existing slide package, not the third-party packaging tool.
SPEC = importlib.util.spec_from_file_location(
    "recpad_slides", ROOT / "slides/build/__init__.py",
    submodule_search_locations=[str(ROOT / "slides/build")])
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
from recpad_slides import deck

NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "ep": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties",
      "vt": "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"}


class SlideBuildTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        # Keep the production CLI's existing import path intact.
        self.loader = patch.object(deck, "load_module", lambda name: __import__(
            f"recpad_slides.slides.{name}", fromlist=[name]))
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def manifest(self, modules, title="Teste"):
        path = self.root / "deck.json"
        path.write_text(json.dumps({"nome": title,
            "template": "template/Template_PPGEC_2026_1.pptx", "slides": modules}), encoding="utf-8")
        return path

    def check_package(self, path, count):
        with zipfile.ZipFile(path) as archive:
            self.assertIsNone(archive.testzip())
            names = set(archive.namelist())
            for name in sorted(names):
                if not name.endswith((".xml", ".rels")):
                    continue
                root = ET.fromstring(archive.read(name))
                if name.endswith(".rels"):
                    base = posixpath.dirname(posixpath.dirname(name)) if name != "_rels/.rels" else ""
                    for rel in root:
                        if rel.get("TargetMode") == "External":
                            continue
                        target = posixpath.normpath(posixpath.join(base, rel.attrib["Target"])).lstrip("/")
                        self.assertIn(target, names, (name, target))
            presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
            self.assertEqual(len(presentation.find("p:sldIdLst", NS)), count)
            props = ET.fromstring(archive.read("docProps/app.xml"))
            self.assertEqual(int(props.find("ep:Slides", NS).text), count)
            self.assertEqual(int(props.find("ep:Notes", NS).text), count)
            titles = props.find("ep:TitlesOfParts/vt:vector", NS)
            self.assertEqual(int(titles.get("size")), len(titles))
            counts = props.findall("ep:HeadingPairs/vt:vector/vt:variant/vt:i4", NS)
            self.assertEqual(int(counts[-1].text), count)
            for i in range(1, count + 1):
                slide = ET.fromstring(archive.read(f"ppt/slides/slide{i}.xml"))
                ids = [node.get("id") for node in slide.findall(".//p:cNvPr", NS)]
                self.assertEqual(len(ids), len(set(ids)))
                notes = ET.fromstring(archive.read(f"ppt/notesSlides/notesSlide{i}.xml"))
                self.assertTrue(notes.findall(".//a:t", NS))

    def test_short_template_and_cloned_decks_with_and_without_cover(self):
        for count in (1, 2, 7, 10, 11, 40):
            for cover in (False, True):
                with self.subTest(count=count, cover=cover):
                    modules = (["s00_capa"] if cover else []) + ["s01_pergunta"] * (count - int(cover))
                    path = deck.build(self.manifest(modules), self.root / "output.pptx")
                    self.check_package(path, count)

    def test_project_deck_preserves_delivered_authorship_and_notes(self):
        path = deck.build(ROOT / "slides/decks/apresentacao.json", self.root / "project.pptx")
        self.check_package(path, 10)
        with zipfile.ZipFile(path) as archive:
            cover = ET.fromstring(archive.read("ppt/slides/slide1.xml"))
            text = " ".join(node.text or "" for node in cover.findall(".//a:t", NS))
            for author in ("Daniel Dias", "Gabriel Albuquerque", "Mário Guerra", "Dezembro"):
                self.assertIn(author, text)
            title = next(sp for sp in cover.findall(".//p:sp", NS)
                         if any("Classificação híbrida" in (t.text or "") for t in sp.findall(".//a:t", NS)))
            self.assertEqual(title.find(".//a:rPr", NS).get("sz"), "2400")
            self.assertEqual(title.find("p:spPr/a:xfrm/a:off", NS).attrib,
                             {"x": "3182112", "y": "1172736"})
            notes = ET.fromstring(archive.read("ppt/notesSlides/notesSlide1.xml"))
            self.assertIn("Projeto da disciplina", " ".join(n.text or "" for n in notes.findall(".//a:t", NS)))

    def test_packaging_failure_preserves_existing_export(self):
        output = self.root / "existing.pptx"
        output.write_bytes(b"previous export")
        with patch.object(deck.zipfile.ZipFile, "write", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                deck.build(self.manifest(["s01_pergunta"]), output)
        self.assertEqual(output.read_bytes(), b"previous export")
        self.assertEqual(list(self.root.glob("*.pptx")), [output])

    def test_title_with_xml_characters_and_backslashes(self):
        title = r"Pesquisa & análise <v2> \ resultados"
        path = deck.build(self.manifest(["s01_pergunta"], title), self.root / "title.pptx")
        with zipfile.ZipFile(path) as archive:
            core = ET.fromstring(archive.read("docProps/core.xml"))
            self.assertEqual(core.find("{http://purl.org/dc/elements/1.1/}title").text, title)

    def test_invalid_slide_counts_and_misplaced_cover(self):
        for modules in ([], ["s01_pergunta"] * 41, ["s01_pergunta", "s00_capa"]):
            with self.subTest(modules=len(modules)), self.assertRaises(SystemExit):
                deck.build(self.manifest(modules), self.root / "invalid.pptx")
        self.assertFalse((self.root / "invalid.pptx").exists())
