"""Convert line-level OCR JSON (Qwen3-VL style) to ALTO XML.

Input is a JSON array of {"text": "...", "bbox_2d": [x1, y1, x2, y2]} with
bboxes normalized 0-1000, one entry per text line in reading order.

Usage: python lines2alto.py page.json page.jpg [--no-word-boxes] > page.xml
"""
import argparse, json, re
from statistics import median
from xml.etree import ElementTree as ET

from PIL import Image

NS = "http://www.loc.gov/standards/alto/ns-v4#"
ET.register_namespace("", NS)
q = lambda t: f"{{{NS}}}{t}"


# One {"text": "...", "bbox_2d": [x1, y1, x2, y2]} entry; the box may end in ], ) or ".
ENTRY = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"bbox_2d"\s*:\s*\[\s*'
                   r'(-?[\d.]+\s*,\s*-?[\d.]+\s*,\s*-?[\d.]+\s*,\s*-?[\d.]+)\s*[\])"]')


def parse_lines(text):
    """Pull the JSON array of lines out of a model response.

    Handles <think> blocks, ```json fences, output cut off mid-array, and
    small syntax slips (e.g. a box closed with ")" or '"' instead of "]") by falling
    back to reading each {"text", "bbox_2d"} entry on its own.
    Returns a list of valid line dicts.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start = text.find("[")
    if start < 0:
        return []
    body = text[start:]
    try:
        data = json.loads(body[:body.rfind("]") + 1])
    except ValueError:
        data = [{"text": json.loads(f'"{t}"'), "bbox_2d": [float(v) for v in b.split(",")]}
                for t, b in ENTRY.findall(body) if len(b.split(",")) == 4]
    lines = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        t, b = item.get("text"), item.get("bbox_2d")
        if (isinstance(t, str) and t.strip() and isinstance(b, list) and len(b) == 4
                and all(isinstance(v, (int, float)) for v in b)):
            x1, y1, x2, y2 = (min(1000, max(0, v)) for v in b)
            lines.append({"text": " ".join(t.split()),
                          "bbox_2d": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]})
    return lines


def group_blocks(lines):
    """Split lines into blocks at large vertical gaps or upward jumps (new column)."""
    if not lines:
        return []
    heights = [l["bbox_2d"][3] - l["bbox_2d"][1] for l in lines]
    mh = median(heights) or 1
    blocks, cur = [], [lines[0]]
    for prev, line in zip(lines, lines[1:]):
        gap = line["bbox_2d"][1] - prev["bbox_2d"][3]
        if gap > 2.5 * mh or line["bbox_2d"][1] < prev["bbox_2d"][1]:
            blocks.append(cur)
            cur = []
        cur.append(line)
    blocks.append(cur)
    return blocks


def convert(lines, image_path, scale=1000, size=None, word_boxes=True):
    """lines: parsed line dicts. size=(width, height) overrides reading image_path.

    Line boxes come from the model. With word_boxes=True, word positions are
    estimated from letter counts; with False, Strings carry no position.
    """
    W, H = size or Image.open(image_path).size
    sx, sy = W / scale, H / scale
    alto = ET.Element(q("alto"))
    desc = ET.SubElement(alto, q("Description"))
    ET.SubElement(desc, q("MeasurementUnit")).text = "pixel"
    src = ET.SubElement(desc, q("sourceImageInformation"))
    ET.SubElement(src, q("fileName")).text = image_path
    page = ET.SubElement(ET.SubElement(alto, q("Layout")), q("Page"),
                         ID="P1", WIDTH=str(W), HEIGHT=str(H), PHYSICAL_IMG_NR="1")
    ps = ET.SubElement(page, q("PrintSpace"), HPOS="0", VPOS="0",
                       WIDTH=str(W), HEIGHT=str(H))

    def geo(x1, y1, x2, y2):
        h, v = int(x1 * sx), int(y1 * sy)
        return dict(HPOS=str(h), VPOS=str(v),
                    WIDTH=str(max(1, int(x2 * sx) - h)), HEIGHT=str(max(1, int(y2 * sy) - v)))

    for n, block in enumerate(group_blocks(lines), 1):
        bb = [min(l["bbox_2d"][0] for l in block), min(l["bbox_2d"][1] for l in block),
              max(l["bbox_2d"][2] for l in block), max(l["bbox_2d"][3] for l in block)]
        tb = ET.SubElement(ps, q("TextBlock"), ID=f"B{n}", **geo(*bb))
        for i, line in enumerate(block, 1):
            g = geo(*line["bbox_2d"])
            tl = ET.SubElement(tb, q("TextLine"), ID=f"B{n}_L{i}", **g)
            words = line["text"].split()
            x, y, w, h = (int(g[k]) for k in ("HPOS", "VPOS", "WIDTH", "HEIGHT"))
            total = sum(len(t) for t in words) or 1
            for j, word in enumerate(words, 1):
                attrs = {"ID": f"B{n}_L{i}_W{j}", "CONTENT": word}
                if word_boxes:
                    ww = max(1, int(w * len(word) / total))  # estimated width
                    attrs.update(HPOS=str(x), VPOS=str(y), WIDTH=str(ww), HEIGHT=str(h))
                    x += ww
                ET.SubElement(tl, q("String"), **attrs)
                if j < len(words):
                    ET.SubElement(tl, q("SP"))
    ET.indent(alto)
    return ET.tostring(alto, encoding="unicode", xml_declaration=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert line OCR JSON to ALTO XML.")
    ap.add_argument("json", help="model response text or JSON array of lines")
    ap.add_argument("image", help="page image (used for its pixel dimensions)")
    ap.add_argument("--no-word-boxes", action="store_true",
                    help="keep line boxes but give words no estimated position")
    a = ap.parse_args()
    lines = parse_lines(open(a.json, encoding="utf-8").read())
    print(convert(lines, a.image, word_boxes=not a.no_word_boxes))
