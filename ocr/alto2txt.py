"""Write plain-text derivatives of ALTO XML files.

Usage: python alto2txt.py kr104zb7305-paddle/kr104zb7305_*.xml

For each FILE.xml writes FILE.txt next to it: one line of text per ALTO
TextLine (its String CONTENTs joined by spaces), with a blank line between
TextBlocks. No coordinates or markup.
"""
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def local(tag):
    return tag.rsplit("}", 1)[-1]


def alto_text(xml):
    blocks = []
    for block in ET.fromstring(xml).iter():
        if local(block.tag) != "TextBlock":
            continue
        lines = []
        for line in block:
            if local(line.tag) == "TextLine":
                words = [s.get("CONTENT", "") for s in line if local(s.tag) == "String"]
                if any(words):
                    lines.append(" ".join(w for w in words if w))
        if lines:
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n" if blocks else ""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} FILE.xml ...")
    for path in map(Path, sys.argv[1:]):
        path.with_suffix(".txt").write_text(alto_text(path.read_bytes()), encoding="utf-8")
    print(f"wrote {len(sys.argv) - 1} .txt files", file=sys.stderr)
