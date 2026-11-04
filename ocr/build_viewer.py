"""Build a standalone HTML viewer that overlays ALTO boxes on page scans.

Usage: python build_viewer.py kr104zb7305 [--alto DIR] [--label NAME] [--out FILE]

Shows the ALTO in DIR (default: DRUID/) next to the ABBYY ALTO that
SDR already has for each page. Images and data are embedded, so the output
file works offline.
"""
import argparse, base64, json, os, sys, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET

from ocr import PURL, fetch, jp2_files

IMG = "https://stacks.stanford.edu/image/iiif/{druid}/{name}/full/!1400,1400/0/default.jpg"
ABBYY = "https://stacks.stanford.edu/file/druid:{druid}/{name}.xml"
TEMPLATE = Path(__file__).parent / "viewer_template.html"


def box(el):
    """[HPOS, VPOS, WIDTH, HEIGHT], or None when the element has no position."""
    if el.get("HPOS") is None or el.get("VPOS") is None:
        return None
    return [round(float(el.get(k, 0))) for k in ("HPOS", "VPOS", "WIDTH", "HEIGHT")]


def parse_alto(xml):
    root = ET.fromstring(xml)
    page = next(e for e in root.iter() if e.tag.endswith("}Page"))
    blocks = []
    for el in page.iter():
        tag = el.tag.split("}")[1]
        if tag in ("Illustration", "GraphicalElement"):
            blocks.append({"k": "ill", "type": el.get("TYPE", tag), "b": box(el)})
        elif tag == "TextBlock":
            lines = []
            for tl in el:
                if not tl.tag.endswith("TextLine"):
                    continue
                words = [[s.get("CONTENT"), *(box(s) or [None] * 4),
                          round(float(s.get("WC", -1)), 2)]
                         for s in tl if s.tag.endswith("String")]
                lines.append({"b": box(tl), "w": words})
            blocks.append({"k": "text", "b": box(el), "l": lines})
    blocks = [b for b in blocks if b["b"]]
    return {"W": round(float(page.get("WIDTH"))), "H": round(float(page.get("HEIGHT"))),
            "blocks": blocks}


def load_page(druid, alto_dir, n, f):
    name = os.path.splitext(f["filename"])[0]
    img = fetch(IMG.format(druid=druid, name=name))
    local = alto_dir / f"{name}.xml"
    chandra = parse_alto(local.read_bytes()) if local.exists() else None
    try:
        abbyy = parse_alto(fetch(ABBYY.format(druid=druid, name=name)))
    except urllib.error.HTTPError:
        abbyy = None
    pres = f.get("presentation") or {}
    print(f"  {name}: chandra={'yes' if chandra else 'no'} abbyy={'yes' if abbyy else 'no'}",
          file=sys.stderr)
    return {"n": n, "name": name, "W": pres.get("width"), "H": pres.get("height"),
            "img": "data:image/jpeg;base64," + base64.b64encode(img).decode(),
            "chandra": chandra, "abbyy": abbyy}


def build(druid, alto_dir, out, label="Chandra", limit=None):
    cocina = json.loads(fetch(PURL.format(druid=druid)))
    files = list(jp2_files(druid))[:limit]
    print(f"{druid}: {len(files)} pages", file=sys.stderr)
    with ThreadPoolExecutor(6) as pool:
        pages = list(pool.map(lambda a: load_page(druid, alto_dir, *a),
                              enumerate(files, 1)))
    meta = {"druid": druid, "title": cocina.get("label", druid),
            "chandra": f"ALTO from {alto_dir}/", "label": label}
    js = lambda v: json.dumps(v, separators=(",", ":")).replace("</", "<\\/")
    html = (TEMPLATE.read_text(encoding="utf-8")
            .replace("__DRUID__", druid)
            .replace("__META__", js(meta))
            .replace("__DATA__", js(pages)))
    Path(out).write_text(html, encoding="utf-8")
    print(f"wrote {out} ({Path(out).stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build an offline ALTO viewer for an SDR object.")
    ap.add_argument("druid")
    ap.add_argument("--alto", help="directory of ALTO files to compare with ABBYY (default: DRUID)")
    ap.add_argument("--label", default="Chandra", help="name shown for those files (default: Chandra)")
    ap.add_argument("--out", help="output HTML file (default: DRUID-viewer.html)")
    ap.add_argument("--pages", type=int, help="only include the first N pages")
    a = ap.parse_args()
    build(a.druid, Path(a.alto or a.druid), a.out or f"{a.druid}-viewer.html", a.label, a.pages)
