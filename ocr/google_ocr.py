"""OCR every jp2 page of an SDR object with Google Cloud Vision and write ALTO.

Usage: python google_ocr.py kr104zb7305 [--out DIR] [--project sul-ai-sandbox] [--viewer]

For each page, the raw Vision response is cached as DIR/NAME.json (default
DIR: DRUID-google/) so later runs skip pages already done, and converted to
DIR/NAME.xml with vision2alto. --viewer then builds DRUID-google-viewer.html.

Authenticates with `gcloud auth print-access-token`; Vision fetches the
full-size IIIF image from stacks itself.
"""
import argparse, json, subprocess, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from build_viewer import PURL, build, fetch, jp2_files
from vision2alto import convert_page, pages

VISION = "https://vision.googleapis.com/v1/images:annotate"
FULL = "https://stacks.stanford.edu/image/iiif/{druid}/{name}/full/full/0/default.jpg"


def annotate(image_url, token, project, retries=3):
    body = json.dumps({"requests": [{
        "image": {"source": {"imageUri": image_url}},
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


def process_page(druid, name, out, token, project):
    raw = out / f"{name}.json"
    if raw.exists():
        status = "cached"
    else:
        raw.write_text(json.dumps(annotate(FULL.format(druid=druid, name=name), token, project),
                                  indent=2), encoding="utf-8")
        status = "ocr"
    ps = list(pages(json.loads(raw.read_text(encoding="utf-8"))))
    if ps:
        (out / f"{name}.xml").write_text(convert_page(ps[0], name), encoding="utf-8")
    else:
        status += ", no text"
    print(f"  {name}: {status}", file=sys.stderr)


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
    token = subprocess.run(["gcloud", "auth", "print-access-token"], check=True,
                           capture_output=True, text=True).stdout.strip()
    cocina = json.loads(fetch(PURL.format(druid=a.druid)))
    names = [Path(f["filename"]).stem for f in jp2_files(cocina)]
    print(f"{a.druid}: {len(names)} pages", file=sys.stderr)

    failed = []
    def run(name):
        try:
            process_page(a.druid, name, out, token, a.project)
        except Exception as e:
            print(f"  {name}: FAILED {e}", file=sys.stderr)
            failed.append(name)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(run, names))

    if a.viewer:
        build(a.druid, out, f"{a.druid}-google-viewer.html", "Google Vision", only_local=True)
    if failed:
        sys.exit(f"{len(failed)} pages failed: {', '.join(sorted(failed))}")


if __name__ == "__main__":
    main()
