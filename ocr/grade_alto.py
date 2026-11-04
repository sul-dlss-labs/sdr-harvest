"""Estimate the quality of ALTO OCR without a reference transcription.

Usage:
    uv run python ocr/grade_alto.py by836jp6882 [more druids or ALTO dirs] [--out pages.csv] [--ink]

Each argument is either a druid, whose ALTO files (``use: transcription``) are
fetched from Stacks, or a local directory of ALTO ``.xml`` files (for example,
output from ocr.py). One CSV row per page is written to --out (default
stdout), and a per-item summary goes to stderr.

Signals for each page:

dict_rate      Share of words found in the wordfreq word list for the item's
               language, or the language ABBYY tagged on the word. Words with
               digits or under 2 letters are skipped, words split by
               hyphenation are joined, and capitalised words that are not in
               the list are left out, since they are usually names.
garbage_rate   Share of tokens that look like OCR noise (rules after Taghva et
               al. 2001): 4+ repeated letters, 6+ consonants in a row, no
               vowels in a 4+ letter word, letters mixed with digits, odd
               capitalisation, punctuation inside the word, 25+ characters.
lang_mismatch  Share of words whose LANG differs from the item's language.
coverage       Area of the text lines as a share of the page.
mean_wc        Mean word confidence, if the ALTO has WC attributes.
ink            (--ink only) Share of dark pixels in a 1000px IIIF image, to find
               pages with content but little or no OCR text.
score          dict_rate * (1 - garbage_rate). It has not been calibrated
               against measured error rates, so use it to rank pages, not as a
               character error rate.
flags          Reasons a page looks wrong.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree as ET

import langcodes
import requests
from urllib3.util import Retry
from tqdm import tqdm
from wordfreq import available_languages, zipf_frequency

PURL = "https://purl.stanford.edu/{druid}.json"
FILE = "https://stacks.stanford.edu/file/{druid}/{name}"
IIIF = "https://stacks.stanford.edu/image/iiif/{druid}/{stem}/full/!1000,1000/0/default.jpg"
DRUID = re.compile(r"^(?:druid:)?([a-z]{2}\d{3}[a-z]{2}\d{4})$")

WORDFREQ_LANGS = set(available_languages())
VOWELS = set("aeiouyàáâãäåæèéêëìíîïòóôõöøùúûüýÿœ")
EDGE_PUNCT = "\"'“”‘’«»()[]{}<>.,;:!?¡¿-–—…*/\\|"

# Thresholds for the flags column. They are first guesses, not tuned values.
LOW_DICT = 0.85
HIGH_GARBAGE = 0.05
HIGH_LANG_MISMATCH = 0.2
FEW_WORDS = 10
INKED = 0.01

session = requests.Session()
session.headers["User-Agent"] = "sdr-harvest grade_alto.py"
session.mount("https://", requests.adapters.HTTPAdapter(
    pool_maxsize=32,
    max_retries=Retry(total=5, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504)),
))


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def base_lang(code: str | None) -> str | None:
    """Turn 'en-US' or 'eng' into a wordfreq language code, or None."""
    if not code:
        return None
    try:
        lang = langcodes.Language.get(code).language
    except (ValueError, LookupError):
        return None
    return lang if lang in WORDFREQ_LANGS else None


@lru_cache(maxsize=200_000)
def known(word: str, lang: str) -> bool:
    return zipf_frequency(word, lang) > 0


@dataclass
class Word:
    text: str
    lang: str | None
    wc: float | None


@dataclass
class Page:
    item: str
    page: str
    words: list[Word] = field(default_factory=list)
    line_area: float = 0.0
    page_area: float = 0.0
    ink: float | None = None


def parse_alto(item: str, name: str, data: bytes) -> Page:
    """Read the words on a page, joining words hyphenated across lines."""
    page = Page(item, name)
    root = ET.fromstring(data)
    for el in root.iter():
        if local(el.tag) == "Page":
            page.page_area = float(el.get("WIDTH", 0)) * float(el.get("HEIGHT", 0))
            break
    pending: Word | None = None  # first half of a hyphenated word
    block_lang = None
    for el in root.iter():
        tag = local(el.tag)
        if tag == "TextBlock":
            block_lang = el.get("LANG")
        elif tag == "TextLine":
            try:
                page.line_area += float(el.get("WIDTH")) * float(el.get("HEIGHT"))
            except (TypeError, ValueError):
                pass  # lines without positions, e.g. ocr.py --blocks-only
            strings = [s for s in el if local(s.tag) in ("String", "HYP")]
            for i, s in enumerate(strings):
                if local(s.tag) == "HYP":
                    continue
                wc = s.get("WC")
                word = Word(
                    s.get("CONTENT", ""),
                    s.get("LANG") or block_lang,
                    float(wc) if wc else None,
                )
                if pending is not None:
                    word.text = pending.text.rstrip("-") + word.text
                    pending = None
                last = i == len(strings) - 1 or (
                    i == len(strings) - 2 and local(strings[-1].tag) == "HYP"
                )
                hyphen = local(strings[-1].tag) == "HYP" or word.text.endswith("-")
                if last and hyphen:
                    pending = word
                else:
                    page.words.append(word)
    if pending is not None:
        page.words.append(pending)
    return page


def is_garbage(token: str) -> bool:
    if re.match(r"(10\.\d{4,}/|https?:|www\.|doi:)", token, re.I):
        return False  # DOIs and URLs
    if len(token) >= 25:
        return True
    alnum = [c for c in token if c.isalnum()]
    if not alnum:
        return False
    if len(set(token) - set(alnum)) > 2 or sum(not c.isalnum() for c in token) > len(alnum):
        return True  # more than a little punctuation inside the word
    if re.search(r"([^\W\d_])\1{3,}", token):
        return True
    # A digit after a lowercase letter (fo11owed, b1ack) or letters between
    # digits (19S3, 1OO.00) is usually a misread l, I, O or S. Numbers with
    # units or labels (30cm, 1970s, n=17, H2O) are not flagged.
    if re.search(r"[a-z]\d|\d[A-Za-z]+[.,]?\d", token):
        return True
    # Check each part of a compound like Hernandez-Gonzalez or cm/hr separately
    for part in re.split(r"[-/]", token):
        letters = "".join(c for c in part if c.isalpha())
        if not letters or letters.isupper():
            continue
        low = letters.lower()
        if len(low) >= 4 and not set(low) & VOWELS:
            return True
        if re.search(r"[^aeiouy]{6,}", low):
            return True
        upper = sum(c.isupper() for c in letters)
        if upper > len(letters) - upper:
            return True  # e.g. "tHE", but not "THE"
        if re.search(r"[a-z][A-Z][a-z]", letters) and not re.match(r"(Mc|Mac|O'|De|La|Le|Van|Von)", part):
            return True  # missing space (perchThey) or a misread capital
    return False


def grade(page: Page, item_lang: str) -> dict:
    tokens = [w for w in page.words if w.text.strip()]
    dict_checked = dict_known = garbage = mismatch = tagged = 0
    for w in tokens:
        token = w.text.strip(EDGE_PUNCT)
        if not token:
            continue
        if is_garbage(token):
            garbage += 1
        lang = base_lang(w.lang)
        if lang:
            tagged += 1
            if lang != item_lang:
                mismatch += 1
        if any(c.isdigit() for c in token) or sum(c.isalpha() for c in token) < 2:
            continue
        word = token.lower()
        hit = known(word, item_lang) or bool(lang and lang != item_lang and known(word, lang))
        if not hit and "-" in word:
            hit = all(known(p, item_lang) for p in word.split("-") if p)
        if not hit and token[0].isupper():
            continue  # probably a name
        dict_checked += 1
        dict_known += hit
    n = len(tokens)
    wcs = [w.wc for w in tokens if w.wc is not None]
    row = {
        "item": page.item,
        "page": page.page,
        "words": n,
        "dict_rate": round(dict_known / dict_checked, 4) if dict_checked else None,
        "garbage_rate": round(garbage / n, 4) if n else None,
        "lang_mismatch": round(mismatch / tagged, 4) if tagged else None,
        "coverage": round(page.line_area / page.page_area, 4) if page.page_area and page.line_area else None,
        "mean_wc": round(statistics.mean(wcs), 4) if wcs else None,
        "ink": round(page.ink, 4) if page.ink is not None else None,
    }
    if row["dict_rate"] is not None:
        row["score"] = round(row["dict_rate"] * (1 - row["garbage_rate"]), 4)
    else:
        row["score"] = None
    flags = []
    if n == 0:
        flags.append("empty")
    elif n < FEW_WORDS:
        flags.append("few_words")
    if page.ink is not None and page.ink > INKED and n < FEW_WORDS:
        flags.append("ink_without_text")
    if row["dict_rate"] is not None and dict_checked >= FEW_WORDS and row["dict_rate"] < LOW_DICT:
        flags.append("low_dict")
    if row["garbage_rate"] is not None and n >= FEW_WORDS and row["garbage_rate"] > HIGH_GARBAGE:
        flags.append("high_garbage")
    if row["lang_mismatch"] is not None and tagged >= FEW_WORDS and row["lang_mismatch"] > HIGH_LANG_MISMATCH:
        flags.append("lang_mismatch")
    row["flags"] = " ".join(flags)
    row["language"] = item_lang
    return row


def get(url: str) -> requests.Response:
    headers = {}
    if url.startswith("https://stacks.stanford.edu/") and os.environ.get("STACKS_TOKEN"):
        # the same service token the harvester uses for Stanford-only files
        headers["Authorization"] = f"Bearer {os.environ['STACKS_TOKEN']}"
    r = session.get(url, headers=headers, timeout=60, allow_redirects=False)
    if 300 <= r.status_code < 400:
        hint = "" if headers else "; set STACKS_TOKEN for Stanford-only files"
        raise RuntimeError(f"needs login for {url.rsplit('/', 1)[-1]} (HTTP {r.status_code}{hint})")
    r.raise_for_status()
    return r


def ink_fraction(druid: str, stem: str) -> float | None:
    from PIL import Image, ImageOps

    try:
        img = Image.open(io.BytesIO(get(IIIF.format(druid=druid, stem=quote(stem))).content))
    except (requests.RequestException, RuntimeError, OSError):
        return None
    gray = ImageOps.grayscale(img)
    w, h = gray.size
    gray = gray.crop((w // 20, h // 20, w - w // 20, h - h // 20))  # skip scanner edges
    hist = gray.histogram()
    total = sum(hist)
    # dark means well below the page's typical (median) brightness
    acc, median = 0, 255
    for value, count in enumerate(hist):
        acc += count
        if acc >= total / 2:
            median = value
            break
    return sum(hist[: max(median - 30, 0)]) / total


def druid_pages(druid: str, ink: bool) -> tuple[list[Page], str | None]:
    cocina = get(PURL.format(druid=druid)).json()
    langs = [l.get("code") or l.get("value") for l in cocina.get("description", {}).get("language", [])]
    alto = []
    for fileset in cocina.get("structural", {}).get("contains", []):
        for f in fileset.get("structural", {}).get("contains", []):
            if f.get("use") == "transcription" and f.get("hasMimeType") in ("application/xml", "text/xml"):
                alto.append(f["filename"])

    def load(name: str) -> Page:
        page = parse_alto(druid, name, get(FILE.format(druid=druid, name=quote(name))).content)
        if ink:
            page.ink = ink_fraction(druid, Path(name).stem)
        return page

    with ThreadPoolExecutor(8) as pool:
        pages = list(pool.map(load, alto))
    lang = next((base_lang(c) for c in langs if base_lang(c)), None)
    return pages, lang


def dir_pages(path: Path) -> list[Page]:
    return [parse_alto(path.name, p.name, p.read_bytes()) for p in sorted(path.glob("*.xml"))]


def item_language(pages: list[Page], cocina_lang: str | None) -> str:
    if cocina_lang:
        return cocina_lang
    counts = Counter(base_lang(w.lang) for p in pages for w in p.words)
    counts.pop(None, None)
    return counts.most_common(1)[0][0] if counts else "en"


def summarise(item: str, rows: list[dict], error: str = "") -> dict:
    scores = sorted(r["score"] for r in rows if r["score"] is not None)
    summary = {"item": item, "pages": len(rows), "graded_pages": len(scores),
               "words": sum(r["words"] for r in rows),
               "flagged_pages": sum(bool(r["flags"]) for r in rows),
               "median_score": None, "p10_score": None, "min_score": None,
               "language": rows[0]["language"] if rows else None, "error": error}
    if scores:
        summary["median_score"] = round(statistics.median(scores), 4)
        summary["p10_score"] = scores[max(0, int(len(scores) * 0.1) - 1)] if len(scores) >= 10 else scores[0]
        summary["min_score"] = scores[0]
    return summary


def describe(s: dict) -> str:
    if s["error"]:
        return f"{s['item']}: {s['error']}"
    if s["median_score"] is None:
        return f"{s['item']}: {s['pages']} pages, no gradable text"
    return (f"{s['item']}: {s['pages']} pages, median score {s['median_score']:.3f}, "
            f"worst-10% {s['p10_score']:.3f}, min {s['min_score']:.3f}, {s['flagged_pages']} flagged")


def grade_item(arg: str, ink: bool) -> tuple[str, list[dict], str]:
    item = arg
    try:
        if Path(arg).is_dir():
            item = Path(arg).name
            pages, lang = dir_pages(Path(arg)), None
        elif m := DRUID.match(arg):
            item = m.group(1)
            pages, lang = druid_pages(item, ink)
        else:
            raise ValueError("not a druid or directory")
        lang = item_language(pages, lang)
        return item, [grade(p, lang) for p in pages], ""
    except Exception as exc:  # keep going through a long batch
        return item, [], f"{type(exc).__name__}: {exc}"


def read_druids(path: Path) -> list[str]:
    """Druids from the first column of a CSV (a header row is skipped)."""
    with path.open(newline="") as f:
        values = [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]
    return [v for v in values if DRUID.match(v)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("items", nargs="*", help="druids or directories of ALTO files")
    ap.add_argument("--from", dest="from_csv", type=Path, help="CSV whose first column lists druids")
    ap.add_argument("--out", type=Path, help="CSV file for per-page rows (default: stdout)")
    ap.add_argument("--summary", type=Path, help="CSV file for one row per item")
    ap.add_argument("--ink", action="store_true", help="fetch IIIF images to find pages with ink but no text")
    ap.add_argument("--jobs", type=int, default=3, help="items graded at once (each fetches 8 pages at once)")
    args = ap.parse_args()
    items = args.items + (read_druids(args.from_csv) if args.from_csv else [])
    if not items:
        ap.error("give druids, directories or --from")

    columns = ["item", "page", "words", "score", "dict_rate", "garbage_rate",
               "lang_mismatch", "coverage", "mean_wc", "ink", "language", "flags"]
    summary_columns = ["item", "pages", "graded_pages", "words", "median_score", "p10_score",
                       "min_score", "flagged_pages", "language", "error"]
    out = args.out.open("w", newline="") if args.out else sys.stdout
    writer = csv.DictWriter(out, fieldnames=columns)
    writer.writeheader()
    summary_file = args.summary.open("w", newline="") if args.summary else None
    summary_writer = csv.DictWriter(summary_file, fieldnames=summary_columns) if summary_file else None
    if summary_writer:
        summary_writer.writeheader()
    failed = 0
    pages = 0
    # The bar only draws on a terminal; redirected stderr gets the per-item lines alone.
    bar = tqdm(total=len(items), unit="item", file=sys.stderr, dynamic_ncols=True, disable=None)
    with bar, ThreadPoolExecutor(args.jobs) as pool:
        for n, (item, rows, error) in enumerate(pool.map(lambda a: grade_item(a, args.ink), items), 1):
            failed += bool(error)
            pages += len(rows)
            writer.writerows(rows)
            out.flush()
            summary = summarise(item, rows, error)
            if summary_writer:
                summary_writer.writerow(summary)
                summary_file.flush()
            tqdm.write(f"[{n}/{len(items)}] {describe(summary)}", file=sys.stderr)
            bar.set_postfix(pages=pages, failed=failed, refresh=False)
            bar.update()
    for f in (out if args.out else None, summary_file):
        if f:
            f.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
