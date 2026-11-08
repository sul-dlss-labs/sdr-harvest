"""OCR every jp2 page of an SDR object with Google Cloud Vision and write ALTO.

Usage: python google_ocr.py kr104zb7305 [--out DIR] [--project sul-ai-sandbox] [--viewer]

For each page, the raw Vision response is cached as DIR/NAME.json (default
DIR: DRUID-google/) so later runs skip pages already done, and converted to
DIR/NAME.xml with vision2alto. --viewer then builds DRUID-google-viewer.html.

Authenticates with `gcloud auth print-access-token`. Vision fetches the
full-size IIIF image from stacks itself; if it can't reach the URL, the image
is downloaded and sent as content instead.
"""
import argparse, base64, json, subprocess, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from build_viewer import PURL, build, fetch, jp2_files
from vision2alto import convert_page, pages

VISION = "https://vision.googleapis.com/v1/images:annotate"
FULL = "https://stacks.stanford.edu/image/iiif/{druid}/{name}/full/full/0/default.jpg"
SIZED = "https://stacks.stanford.edu/image/iiif/{druid}/{name}/full/{w},/0/default.jpg"
# Vision rejects images over 75 megapixels (and files over 20 MB); larger jp2s
# are sent downsized and their coordinates scaled back up.
MAX_PIXELS = 40_000_000


def annotate(image_url, token, project, retries=3):
    """Have Vision fetch image_url itself; if it can't reach the URL, download
    the image and send its bytes instead."""
    try:
        return _annotate({"source": {"imageUri": image_url}}, token, project, retries)
    except RuntimeError as e:
        if "access the URL" not in str(e):
            raise
    content = base64.b64encode(fetch(image_url, timeout=300)).decode()
    return _annotate({"content": content}, token, project, retries)


def _annotate(image, token, project, retries):
    body = json.dumps({"requests": [{
        "image": image,
        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
    }]}).encode()
    req = urllib.request.Request(VISION, data=body, method="POST", headers={
        "Authorization": f"Bearer {token}",
        "x-goog-user-project": project,
        "Content-Type": "application/json; charset=utf-8",
    })
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt * 5)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        # Per-image errors (e.g. Vision couldn't fetch the URL) come back with HTTP 200.
        err = data["responses"][0].get("error")
        if not err:
            return data
        if attempt == retries - 1:
            raise RuntimeError(f"Vision error: {err.get('message')}")
        time.sleep(2 ** attempt * 5)


def image_url(druid, name, size):
    """Full-size IIIF URL, or a downsized one for images over MAX_PIXELS."""
    if size and size[0] * size[1] > MAX_PIXELS:
        w = int(size[0] * (MAX_PIXELS / (size[0] * size[1])) ** 0.5)
        return SIZED.format(druid=druid, name=name, w=w)
    return FULL.format(druid=druid, name=name)


def process_page(druid, f, out, token, project):
    """OCR one cocina jp2 file into OUT/NAME.json and OUT/NAME.xml.

    Coordinates are scaled to the jp2's size from cocina, so they're right
    even when Vision saw a downsized image. A page with no text still gets
    an (empty) ALTO file.
    """
    name = Path(f["filename"]).stem
    pres = f.get("presentation") or {}
    size = (pres["width"], pres["height"]) if pres.get("width") and pres.get("height") else None
    raw = out / f"{name}.json"
    if raw.exists():
        status = "cached"
    else:
        raw.write_text(json.dumps(annotate(image_url(druid, name, size), token, project),
                                  indent=2), encoding="utf-8")
        status = "ocr"
    ps = list(pages(json.loads(raw.read_text(encoding="utf-8"))))
    if ps:
        page = ps[0]
        scale = size[0] / page["width"] if size and page.get("width") else 1.0
    else:
        status += ", no text"
        if not size:
            print(f"  {name}: {status}, no size in cocina, no ALTO written", file=sys.stderr)
            return
        page, scale = {"width": size[0], "height": size[1], "blocks": []}, 1.0
    (out / f"{name}.xml").write_text(convert_page(page, name, scale), encoding="utf-8")
    if scale != 1.0:
        status += f", scaled x{scale:.3f}"
    print(f"  {name}: {status}", file=sys.stderr)


def ocr_object(druid, files, out, token, project, workers=4):
    """OCR every jp2 file of one object; returns the names that failed."""
    failed = []
    def run(f):
        try:
            process_page(druid, f, out, token, project)
        except Exception as e:
            print(f"  {Path(f['filename']).stem}: FAILED {e}", file=sys.stderr)
            failed.append(Path(f["filename"]).stem)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, files))
    return failed


def gcloud_token():
    return subprocess.run(["gcloud", "auth", "print-access-token"], check=True,
                          capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description="OCR an SDR object with Google Cloud Vision.")
    ap.add_argument("druid")
    ap.add_argument("--out", help="output directory (default: DRUID-google)")
    ap.add_argument("--project", default="sul-ai-sandbox", help="GCP project billed for Vision")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--viewer", action="store_true", help="also build DRUID-google-viewer.html")
    a = ap.parse_args()

    out = Path(a.out or f"{a.druid}-google")
    out.mkdir(parents=True, exist_ok=True)
    token = gcloud_token()
    cocina = json.loads(fetch(PURL.format(druid=a.druid)))
    files = list(jp2_files(cocina))
    print(f"{a.druid}: {len(files)} pages", file=sys.stderr)
    failed = ocr_object(a.druid, files, out, token, a.project, a.workers)

    if a.viewer:
        build(a.druid, out, f"{a.druid}-google-viewer.html", "Google Vision", only_local=True)
    if failed:
        sys.exit(f"{len(failed)} pages failed: {', '.join(sorted(failed))}")


if __name__ == "__main__":
    main()
