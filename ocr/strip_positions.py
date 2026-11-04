"""Turn existing ALTO into blocks-only ALTO by removing line and word positions.

Usage: python strip_positions.py SRC_DIR DEST_DIR

TextBlock and Illustration keep their coordinates; TextLine, String and SP
lose HPOS/VPOS/WIDTH/HEIGHT (all optional in ALTO v4). Use this to convert
files made before ocr.py cached the model's HTML; with cached HTML,
`ocr.py --blocks-only` also splits lines at <br>.
"""
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

NS = "http://www.loc.gov/standards/alto/ns-v4#"
ET.register_namespace("", NS)
POS = ("HPOS", "VPOS", "WIDTH", "HEIGHT")


def strip(xml):
    root = ET.fromstring(xml)
    for el in root.iter():
        if el.tag in (f"{{{NS}}}TextLine", f"{{{NS}}}String", f"{{{NS}}}SP"):
            for k in POS:
                el.attrib.pop(k, None)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} SRC_DIR DEST_DIR")
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.mkdir(exist_ok=True)
    files = sorted(src.glob("*.xml"))
    for f in files:
        (dest / f.name).write_text(strip(f.read_bytes()), encoding="utf-8")
    print(f"wrote {len(files)} files to {dest}/", file=sys.stderr)
