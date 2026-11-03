"""Score line-level OCR boxes against the ABBYY ALTO in SDR.

Usage: python score_lines.py kr104zb7305 kr104zb7305-qwen kr104zb7305-qwen-1536 [--pages 5,7]

Each DIR holds line JSON files (kr104zb7305_0005.json, ... as cached by
`ocr.py --engine qwen`). Each model line is matched to the ABBYY line with
the most similar text; the vertical error is the distance between the two
line centres in full-size jp2 pixels. Indents are first lines that start
more than 100px right of the page's usual left margin.
"""
import argparse, difflib, json, os, re, sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from lines2alto import parse_lines
from ocr import fetch, jp2_files

ABBYY = "https://stacks.stanford.edu/file/druid:{druid}/{name}.xml"


def norm(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def abbyy_lines(druid, name):
    root = ET.fromstring(fetch(ABBYY.format(druid=druid, name=name)))
    out = []
    for tl in root.iter():
        if tl.tag.endswith("TextLine"):
            text = "".join(s.get("CONTENT") for s in tl if s.tag.endswith("String"))
            x, y, w, h = (float(tl.get(k)) for k in ("HPOS", "VPOS", "WIDTH", "HEIGHT"))
            out.append((norm(text), x, y + h / 2))
    return out


def margin(xs):
    return Counter(round(x / 25) for x in xs).most_common(1)[0][0] * 25


def score_page(lines, ab, W, H):
    """Return per-line (abbyy_y_err, is_paragraph_start, abbyy_indent, model_indent)."""
    pairs = []
    for l in lines:
        t = norm(l["text"])
        if len(t) <= 15:
            continue
        best = max(ab, key=lambda a: difflib.SequenceMatcher(None, t, a[0]).ratio())
        if difflib.SequenceMatcher(None, t, best[0]).ratio() > .8:
            x1, y1, _, y2 = l["bbox_2d"]
            pairs.append((x1 * W / 1000, (y1 + y2) / 2 * H / 1000, best[1], best[2]))
    if len(pairs) < 5:
        return []
    am, qm = margin(p[2] for p in pairs), margin(p[0] for p in pairs)
    rows = []
    for k, (qx, qy, ax, ay) in enumerate(pairs):
        start = k == 0 or ay - pairs[k - 1][3] > 1.4 * (pairs[1][3] - pairs[0][3] or 1) \
            or ax - am > 100
        rows.append((ay - qy, start, ax - am > 100, qx - qm > 100))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Score line OCR boxes against SDR's ABBYY ALTO.")
    ap.add_argument("druid")
    ap.add_argument("dirs", nargs="+", help="directories of line JSON files")
    ap.add_argument("--pages", type=lambda v: {int(x) for x in v.split(",")},
                    help="only these page numbers, e.g. 5,7")
    a = ap.parse_args()
    files = list(jp2_files(a.druid))
    ab_cache = {}
    print(f"{'dir':28} {'lines':>5} {'mean |y err|':>12} {'para starts':>11} {'other':>7} "
          f"{'max':>6} {'indents found':>13}")
    for d in a.dirs:
        rows = []
        for i, f in enumerate(files, 1):
            name = os.path.splitext(f["filename"])[0]
            path = Path(d) / f"{name}.json"
            if (a.pages and i not in a.pages) or not path.exists():
                continue
            if name not in ab_cache:
                ab_cache[name] = abbyy_lines(a.druid, name)
            pres = f.get("presentation") or {}
            rows += score_page(parse_lines(path.read_text(encoding="utf-8")),
                               ab_cache[name], pres["width"], pres["height"])
        if not rows:
            print(f"{d:28} no matching pages")
            continue
        mean = lambda v: sum(v) / len(v) if v else float("nan")
        err = [abs(r[0]) for r in rows]
        starts = [abs(r[0]) for r in rows if r[1]]
        other = [abs(r[0]) for r in rows if not r[1]]
        ind = [r for r in rows if r[2]]
        found = sum(1 for r in ind if r[3])
        print(f"{d:28} {len(rows):5d} {mean(err):10.1f}px {mean(starts):9.1f}px "
              f"{mean(other):5.1f}px {max(err):4.0f}px {found:5d} of {len(ind):<3d}")


if __name__ == "__main__":
    main()
