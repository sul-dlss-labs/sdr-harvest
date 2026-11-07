"""Write plain-text derivatives of ALTO XML files.

Usage: python alto2txt.py kr104zb7305-paddle/kr104zb7305_*.xml

For each FILE.xml writes FILE.txt next to it: one line of text per ALTO
TextLine, with a blank line between TextBlocks. No coordinates or markup.
Strings are joined with a space where the line has an SP between them (or
always, if the line has no SPs at all); a HYP becomes "-".
"""
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def local(tag):
    return tag.rsplit("}", 1)[-1]


def line_text(line):
    kids = [(local(k.tag), k.get("CONTENT", "")) for k in line]
    spaced = not any(t == "SP" for t, _ in kids)
    out = []
    for t, content in kids:
        if t == "String" and content:
            if out and out[-1] != " " and spaced:
                out.append(" ")
            out.append(content)
        elif t == "SP" and out:
            out.append(" ")
        elif t == "HYP":
            out.append(content or "-")
    return "".join(out).strip()


def alto_text(xml):
    blocks = []
    for block in ET.fromstring(xml).iter():
        if local(block.tag) != "TextBlock":
            continue
        lines = []
        for line in block:
            if local(line.tag) == "TextLine":
                text = line_text(line)
                if text:
                    lines.append(text)
        if lines:
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n" if blocks else ""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} FILE.xml ...")
    for path in map(Path, sys.argv[1:]):
        path.with_suffix(".txt").write_text(alto_text(path.read_bytes()), encoding="utf-8")
    print(f"wrote {len(sys.argv) - 1} .txt files", file=sys.stderr)
