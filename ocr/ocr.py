"""OCR every jp2 image in an SDR object and write ALTO XML per page.

Usage: python ocr.py kr104zb7305
Writes kr104zb7305/kr104zb7305_0001.xml, ...
"""
import base64, json, os, re, sys, urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

from html2alto import convert

# LM Studio server
URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "chandra-ocr-2"   # replace with the id from /v1/models

PURL = "https://purl.stanford.edu/{druid}.json"
IIIF = "https://stacks.stanford.edu/image/iiif/{druid}/{name}/full/!1024,1024/0/default.jpg"

PROMPT = (Path(__file__).parent / "prompt.txt").read_text(encoding="utf-8").strip()


def fetch(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def jp2_files(druid):
    """Yield the cocina file dicts with mimetype image/jp2, in structural order."""
    cocina = json.loads(fetch(PURL.format(druid=druid)))
    for fileset in cocina.get("structural", {}).get("contains", []):
        for f in fileset.get("structural", {}).get("contains", []):
            if f.get("hasMimeType") == "image/jp2":
                yield f


def ocr(image_bytes, mime="image/jpeg"):
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 8192,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    }
    req = urllib.request.Request(URL, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def describe(response):
    """Summarize a chat completion response for debugging a bad result."""
    choice = (response.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    lines = [f"finish_reason={choice.get('finish_reason')!r} usage={response.get('usage')}"]
    for key, val in msg.items():
        if key in ("role",) or not val:
            continue
        text = val if isinstance(val, str) else json.dumps(val)
        more = f" ... ({len(text)} chars)" if len(text) > 500 else ""
        lines.append(f"{key}: {text[:500]!r}{more}")
    if len(lines) == 1:
        lines.append(f"message has no content; keys={list(msg)}")
    return "\n    ".join(lines)


def clean_html(text):
    """Strip a surrounding ```html fence if the model added one."""
    m = re.match(r"^\s*```(?:html)?\s*\n(.*?)\n?```\s*$", text, re.S)
    return m.group(1) if m else text


def extract_html(response):
    """Return the layout HTML from a chat completion response.

    LM Studio's reasoning parser sometimes files the whole answer under
    reasoning_content and leaves content empty, so fall back to that.
    """
    msg = response["choices"][0]["message"]
    for key in ("content", "reasoning_content"):
        html = clean_html(msg.get(key) or "")
        if is_layout_html(html):
            return html
    return clean_html(msg.get("content") or "")


def is_layout_html(html):
    """True if the output has at least one top-level layout <div data-bbox>."""
    soup = BeautifulSoup(html, "html.parser")
    return any(div.has_attr("data-bbox")
               for div in soup.find_all("div", recursive=False))


def process(druid):
    outdir = Path(druid)
    outdir.mkdir(exist_ok=True)
    files = list(jp2_files(druid))
    print(f"{druid}: {len(files)} jp2 files", file=sys.stderr)
    failures = []
    for i, f in enumerate(files, 1):
        filename = f["filename"]
        name = os.path.splitext(filename)[0]
        out = outdir / f"{name}.xml"
        if out.exists():
            print(f"[{i}/{len(files)}] {name}: exists, skipping", file=sys.stderr)
            continue
        print(f"[{i}/{len(files)}] {name}", file=sys.stderr)
        try:
            image = fetch(IIIF.format(druid=druid, name=name))
            response = ocr(image)
            html = extract_html(response)
        except Exception as e:
            print(f"  error: {e}", file=sys.stderr)
            failures.append(name)
            continue
        if not is_layout_html(html):
            raw = outdir / f"{name}.response.json"
            raw.write_text(json.dumps(response, indent=2), encoding="utf-8")
            print(f"  not HTML layout output, skipping (full response in {raw}):\n"
                  f"    {describe(response)}", file=sys.stderr)
            failures.append(name)
            continue
        # Bboxes are normalized 0-1000, so scale them to the full-size jp2
        # dimensions rather than the 1024px derivative we sent to the model.
        pres = f.get("presentation") or {}
        size = (pres["width"], pres["height"]) if "width" in pres and "height" in pres else None
        if size is None:
            from io import BytesIO
            from PIL import Image
            size = Image.open(BytesIO(image)).size
        out.write_text(convert(html, filename, size=size), encoding="utf-8")
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}", file=sys.stderr)
    return not failures


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} DRUID")
    sys.exit(0 if process(sys.argv[1]) else 1)
