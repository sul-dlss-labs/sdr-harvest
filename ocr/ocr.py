"""OCR every jp2 image in an SDR object and write ALTO XML per page.

Usage: python ocr.py kr104zb7305 [--engine chandra|qwen] [--blocks-only] [--out DIR]
Chandra writes kr104zb7305/kr104zb7305_0001.xml, ... and caches the model's
HTML as kr104zb7305/kr104zb7305_0001.html so later runs skip the OCR step.
Qwen writes to kr104zb7305-qwen/ and caches its line JSON as .json files.
"""
import argparse, base64, json, os, re, sys, urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

import lines2alto
from html2alto import convert

# LM Studio server
URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "chandra-ocr-2"   # replace with the id from /v1/models

PURL = "https://purl.stanford.edu/{druid}.json"
IIIF = "https://stacks.stanford.edu/image/iiif/{druid}/{name}/full/!{size},{size}/0/default.jpg"
IMAGE_SIZE = 1024  # long edge in pixels of the image sent to the model

PROMPT = (Path(__file__).parent / "prompt.txt").read_text(encoding="utf-8").strip()
QWEN_PROMPT = (Path(__file__).parent / "prompt_qwen.txt").read_text(encoding="utf-8").strip()


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


def ocr(image_bytes, mime="image/jpeg", model=MODEL, prompt=PROMPT, max_tokens=8192,
        **sampling):
    """sampling: extra request fields such as repeat_penalty=1.1."""
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        **sampling,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
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


def extract_lines(response):
    """Return the parsed line list from a Qwen-style JSON response."""
    msg = response["choices"][0]["message"]
    for key in ("content", "reasoning_content"):
        lines = lines2alto.parse_lines(msg.get(key) or "")
        if lines:
            return lines
    return []


# Each engine: model id, prompt, how to pull its output from a response,
# how to cache it, and how to turn it into ALTO.
ENGINES = {
    "chandra": dict(
        model=MODEL, prompt=PROMPT, ext="html", suffix="",
        extract=extract_html, ok=is_layout_html,
        dump=lambda html: html, load=lambda text: text,
        convert=lambda html, fn, size, blocks_only:
            convert(html, fn, size=size, blocks_only=blocks_only)),
    "qwen": dict(
        model="qwen/qwen3-vl-4b", prompt=QWEN_PROMPT, ext="json", suffix="-qwen",
        extract=extract_lines, ok=bool,
        dump=lambda lines: json.dumps(lines, indent=1, ensure_ascii=False),
        load=json.loads,
        convert=lambda lines, fn, size, blocks_only:
            lines2alto.convert(lines, fn, size=size, word_boxes=not blocks_only)),
    # PaddleOCR-VL's text spotting mode: a fixed task prompt, and boxes as
    # <|LOC_n|> corner tokens. ~1.6 MP is the size its own pipeline uses.
    # Dense tables and charts can send it into a loop at that size; retry
    # at other sizes, nearest first. Which size works varies by page.
    "paddle": dict(
        model="paddleocr-vl-1.6-mlx", prompt="Spotting:", ext="json", suffix="-paddle",
        image_size=1440, retry_sizes=(1280, 1024, 896, 768),
        # LM Studio's default repeat penalty (1.1) drops repeated punctuation,
        # letters and page numbers and slightly blurs the boxes, whose corner
        # tokens repeat by design; it doesn't prevent the loops either.
        sampling={"repeat_penalty": 1.0},
        extract=lambda r: lines2alto.parse_loc_lines(
            r["choices"][0]["message"].get("content") or ""),
        ok=bool,
        dump=lambda lines: json.dumps(lines, indent=1, ensure_ascii=False),
        load=json.loads,
        convert=lambda lines, fn, size, blocks_only:
            lines2alto.convert(lines, fn, size=size, word_boxes=not blocks_only)),
}


def process(druid, out_dir=None, blocks_only=False, engine="chandra", model=None,
            image_size=None, pages=None, prompt_file=None, max_tokens=8192,
            retry_sizes=None):
    eng = ENGINES[engine]
    retry_sizes = retry_sizes if retry_sizes is not None else eng.get("retry_sizes", ())
    model = model or eng["model"]
    default_size = eng.get("image_size", IMAGE_SIZE)
    image_size = image_size or default_size
    prompt = (Path(prompt_file).read_text(encoding="utf-8").strip()
              if prompt_file else eng["prompt"])
    # A non-default model or image size gets its own cache, e.g.
    # kr104zb7305-qwen-1536/ or kr104zb7305-qwen-qwen3-vl-8b-1536/
    model_suffix = "" if model == eng["model"] else "-" + re.sub(r"[^\w.-]+", "-", model.split("/")[-1])
    # A prompt file other than the default adds its name, e.g. prompt_qwen_tight.txt -> -tight
    prompt_suffix = ("-" + re.sub(r"^prompt_(qwen_)?", "", Path(prompt_file).stem)
                     if prompt_file and prompt != eng["prompt"] else "")
    size_suffix = "" if image_size == default_size else f"-{image_size}"
    cachedir = Path(druid + eng["suffix"] + model_suffix + prompt_suffix + size_suffix)
    outdir = Path(out_dir) if out_dir else cachedir
    cachedir.mkdir(exist_ok=True)
    outdir.mkdir(exist_ok=True)
    files = list(jp2_files(druid))
    print(f"{druid}: {len(files)} jp2 files, engine={engine} model={model} -> {outdir}/",
          file=sys.stderr)
    failures = []
    for i, f in enumerate(files, 1):
        if pages and i not in pages:
            continue
        filename = f["filename"]
        name = os.path.splitext(filename)[0]
        out = outdir / f"{name}.xml"
        if out.exists():
            print(f"[{i}/{len(files)}] {name}: exists, skipping", file=sys.stderr)
            continue
        cached = cachedir / f"{name}.{eng['ext']}"
        image = None
        if cached.exists():
            print(f"[{i}/{len(files)}] {name} (cached {eng['ext']})", file=sys.stderr)
            result = eng["load"](cached.read_text(encoding="utf-8"))
        else:
            print(f"[{i}/{len(files)}] {name}", file=sys.stderr)
            try:
                image = fetch(IIIF.format(druid=druid, name=name, size=image_size))
                response = ocr(image, model=model, prompt=prompt, max_tokens=max_tokens,
                               **eng.get("sampling", {}))
                result = eng["extract"](response)
            except Exception as e:
                print(f"  error: {e}", file=sys.stderr)
                failures.append(name)
                continue
            if not eng["ok"](result):
                raw = cachedir / f"{name}.response.json"
                raw.write_text(json.dumps(response, indent=2), encoding="utf-8")
                print(f"  no usable {eng['ext']} output, skipping (full response in {raw}):\n"
                      f"    {describe(response)}", file=sys.stderr)
                failures.append(name)
                continue
            # Hitting max_tokens is usually a repetition loop; a different
            # image size often avoids it. Try each retry size until one finishes.
            for retry_size in retry_sizes:
                if response["choices"][0].get("finish_reason") != "length":
                    break
                if retry_size == image_size:
                    continue
                print(f"  output hit max_tokens; retrying at {retry_size}px", file=sys.stderr)
                try:
                    small = fetch(IIIF.format(druid=druid, name=name, size=retry_size))
                    retry = ocr(small, model=model, prompt=prompt, max_tokens=max_tokens,
                                **eng.get("sampling", {}))
                    retry_result = eng["extract"](retry)
                    if retry["choices"][0].get("finish_reason") != "length" and eng["ok"](retry_result):
                        print(f"  retry finished: {len(retry_result)} entries", file=sys.stderr)
                        response, result, image = retry, retry_result, small
                except Exception as e:
                    print(f"  retry error: {e}", file=sys.stderr)
            if response["choices"][0].get("finish_reason") == "length":
                raw = cachedir / f"{name}.response.json"
                raw.write_text(json.dumps(response, indent=2), encoding="utf-8")
                print(f"  warning: output hit max_tokens; later lines may be missing "
                      f"(full response in {raw})", file=sys.stderr)
            cached.write_text(eng["dump"](result), encoding="utf-8")
        # Bboxes are normalized 0-1000, so scale them to the full-size jp2
        # dimensions rather than the 1024px derivative we sent to the model.
        pres = f.get("presentation") or {}
        size = (pres["width"], pres["height"]) if "width" in pres and "height" in pres else None
        if size is None:
            from io import BytesIO
            from PIL import Image
            image = image or fetch(IIIF.format(druid=druid, name=name, size=image_size))
            size = Image.open(BytesIO(image)).size
        out.write_text(eng["convert"](result, filename, size, blocks_only), encoding="utf-8")
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}", file=sys.stderr)
    return not failures


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OCR an SDR object's jp2 images to ALTO XML.")
    ap.add_argument("druid", help="object id, e.g. kr104zb7305")
    ap.add_argument("--engine", choices=ENGINES, default="chandra",
                    help="chandra: layout blocks as HTML (default); "
                         "qwen: text lines as JSON (qwen/qwen3-vl-4b); "
                         "paddle: text lines from PaddleOCR-VL spotting")
    ap.add_argument("--model", help="override the LM Studio model id for the engine")
    ap.add_argument("--blocks-only", action="store_true",
                    help="drop estimated positions: chandra keeps only block boxes, "
                         "qwen keeps block and line boxes but not word boxes")
    ap.add_argument("--prompt", help="prompt file to use instead of the engine's default")
    ap.add_argument("--image-size", type=int,
                    help=f"long edge in pixels of the image sent to the model "
                         f"(default: {IMAGE_SIZE}, or 1440 for paddle)")
    ap.add_argument("--pages", type=lambda v: {int(x) for x in v.split(",") if x.strip()},
                    help="only these page numbers, e.g. 5,7")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="limit on the model's output per page (default: 8192)")
    ap.add_argument("--retry-sizes", type=lambda v: tuple(int(x) for x in v.split(",") if int(x)),
                    help="if a page hits --max-tokens, retry it at these image sizes in order, "
                         "e.g. 1024,768 (default: 1280,1024,896,768 for paddle, none for other engines; "
                         "0 to disable)")
    ap.add_argument("--out", help="directory for the XML files (default: DRUID, "
                                  "DRUID-qwen for qwen, plus -MODEL and -SIZE for a non-default --model or --image-size)")
    a = ap.parse_args()
    sys.exit(0 if process(a.druid, out_dir=a.out, blocks_only=a.blocks_only,
                          engine=a.engine, model=a.model, image_size=a.image_size, pages=a.pages,
                          prompt_file=a.prompt, max_tokens=a.max_tokens,
                          retry_sizes=a.retry_sizes) else 1)
