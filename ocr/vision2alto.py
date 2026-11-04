"""Convert a Google Cloud Vision DOCUMENT_TEXT_DETECTION response to ALTO XML.

Vision returns pages > blocks > paragraphs > words > symbols, each with a
bounding polygon. There is no line level: a line ends at a word whose last
symbol has a LINE_BREAK, EOL_SURE_SPACE or HYPHEN break. The mapping is

    Vision block  -> ComposedBlock (TextBlock if it has one paragraph)
    paragraph     -> TextBlock
    (derived)     -> TextLine
    word          -> String (WC = word confidence)
    HYPHEN break  -> HYP

Boxes are the axis-aligned bounds of each polygon, in the pixels of the image
sent to Vision (pass --scale if the jp2 is a different size).

Usage: python vision2alto.py ocr.json [--name kr104zb7305_0004] [--out-dir DIR]
Writes one ALTO file per page in the response (DIR/NAME.xml); with no
--out-dir, a single page is printed to stdout.
"""
import argparse, json, sys
from pathlib import Path
from xml.etree import ElementTree as ET

NS = "http://www.loc.gov/standards/alto/ns-v4#"
ET.register_namespace("", NS)
q = lambda t: f"{{{NS}}}{t}"

LINE_END = {"LINE_BREAK", "EOL_SURE_SPACE", "HYPHEN"}
BLOCK_TYPES = {"TABLE": "Table", "PICTURE": "Illustration", "RULER": "GraphicalElement",
               "BARCODE": "Illustration"}


def bounds(poly, scale=1.0):
    """(x1, y1, x2, y2) of a boundingBox. Vision omits x or y when it is 0."""
    vs = poly.get("vertices") or []
    xs = [v.get("x", 0) for v in vs] or [0]
    ys = [v.get("y", 0) for v in vs] or [0]
    return min(xs) * scale, min(ys) * scale, max(xs) * scale, max(ys) * scale


def union(boxes):
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def geo(b):
    h, v = round(b[0]), round(b[1])
    return dict(HPOS=str(h), VPOS=str(v),
                WIDTH=str(max(1, round(b[2]) - h)), HEIGHT=str(max(1, round(b[3]) - v)))


def word_break(word):
    syms = word.get("symbols") or []
    return ((syms[-1].get("property") or {}).get("detectedBreak") or {}).get("type") if syms else None


def split_lines(words):
    """Group a paragraph's words into lines at line-ending breaks."""
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if word_break(w) in LINE_END:
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    return lines


def lang(el):
    langs = (el.get("property") or {}).get("detectedLanguages") or []
    return langs[0]["languageCode"] if langs else None


def convert_page(page, file_name="", scale=1.0):
    W, H = round(page["width"] * scale), round(page["height"] * scale)
    alto = ET.Element(q("alto"))
    desc = ET.SubElement(alto, q("Description"))
    ET.SubElement(desc, q("MeasurementUnit")).text = "pixel"
    src = ET.SubElement(desc, q("sourceImageInformation"))
    ET.SubElement(src, q("fileName")).text = file_name
    ocr = ET.SubElement(desc, q("OCRProcessing"), ID="OCR1")
    step = ET.SubElement(ocr, q("ocrProcessingStep"))
    sw = ET.SubElement(step, q("processingSoftware"))
    ET.SubElement(sw, q("softwareCreator")).text = "Google"
    ET.SubElement(sw, q("softwareName")).text = "Cloud Vision API DOCUMENT_TEXT_DETECTION"
    pg = ET.SubElement(ET.SubElement(alto, q("Layout")), q("Page"), ID="P1",
                       WIDTH=str(W), HEIGHT=str(H), PHYSICAL_IMG_NR="1")
    if "confidence" in page:
        pg.set("PC", f"{page['confidence']:.4f}")
    ps = ET.SubElement(pg, q("PrintSpace"), HPOS="0", VPOS="0", WIDTH=str(W), HEIGHT=str(H))

    for n, block in enumerate(page.get("blocks", []), 1):
        bid = f"B{n}"
        btype = block.get("blockType", "TEXT")
        bb = bounds(block["boundingBox"], scale)
        if btype in BLOCK_TYPES and not block.get("paragraphs"):
            ET.SubElement(ps, q(BLOCK_TYPES[btype]), ID=bid, TYPE=btype.lower(), **geo(bb))
            continue
        paras = block.get("paragraphs", [])
        parent = ps
        if len(paras) > 1:
            parent = ET.SubElement(ps, q("ComposedBlock"), ID=bid, TYPE=btype.lower(), **geo(bb))
        for m, para in enumerate(paras, 1):
            pid = f"{bid}_P{m}" if len(paras) > 1 else bid
            tb = ET.SubElement(parent, q("TextBlock"), ID=pid,
                               **geo(bounds(para["boundingBox"], scale)))
            if lang(para):
                tb.set("LANG", lang(para))
            for i, line in enumerate(split_lines(para.get("words", [])), 1):
                boxes = [bounds(w["boundingBox"], scale) for w in line]
                tl = ET.SubElement(tb, q("TextLine"), ID=f"{pid}_L{i}", **geo(union(boxes)))
                for j, (w, wb) in enumerate(zip(line, boxes), 1):
                    text = "".join(s.get("text", "") for s in w.get("symbols", []))
                    attrs = {"ID": f"{pid}_L{i}_W{j}", "CONTENT": text, **geo(wb)}
                    if "confidence" in w:
                        attrs["WC"] = f"{w['confidence']:.4f}"
                    ET.SubElement(tl, q("String"), **attrs)
                    brk = word_break(w)
                    if brk == "HYPHEN":
                        ET.SubElement(tl, q("HYP"), CONTENT="-")
                    elif j < len(line):
                        ET.SubElement(tl, q("SP"))
    ET.indent(alto)
    return ET.tostring(alto, encoding="unicode", xml_declaration=True)


def pages(response):
    """Pages from a Vision response: {"responses": [...]} or a single response."""
    for r in response.get("responses", [response]):
        yield from (r.get("fullTextAnnotation") or {}).get("pages", [])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert Google Vision OCR JSON to ALTO XML.")
    ap.add_argument("json")
    ap.add_argument("--name", help="page name for the output file and fileName (default: JSON stem)")
    ap.add_argument("--out-dir", help="write NAME.xml (NAME_n.xml for multiple pages) here")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply coordinates, e.g. when Vision saw a downsized image")
    a = ap.parse_args()
    name = a.name or Path(a.json).stem
    ps = list(pages(json.load(open(a.json, encoding="utf-8"))))
    if not ps:
        sys.exit("no fullTextAnnotation pages in response")
    if not a.out_dir:
        print(convert_page(ps[0], name, a.scale))
    else:
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for k, p in enumerate(ps, 1):
            stem = name if len(ps) == 1 else f"{name}_{k}"
            (out / f"{stem}.xml").write_text(convert_page(p, stem, a.scale), encoding="utf-8")
            print(f"wrote {out / stem}.xml", file=sys.stderr)
