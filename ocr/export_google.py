"""OCR every object in a CSV of druids with Google Cloud Vision into export/.

Usage: python export_google.py hopkins_student_papers_world_access.csv [--out export]

For each druid in the CSV's "Druid" column, reads its cocina from PURL and
writes, for every jp2 file, export/{druid}/{name}.json (the raw Vision
response) and export/{druid}/{name}.xml (ALTO). Pages with no text get an
empty ALTO. Pages whose .json already exists aren't sent to Vision again, so
the command can be rerun to resume or to retry failures. Failures are listed
in export/failures.tsv.
"""
import argparse, csv, json, sys, time
from pathlib import Path

from build_viewer import PURL, fetch, jp2_files
from google_ocr import gcloud_token, ocr_object


def main():
    ap = argparse.ArgumentParser(description="OCR a CSV of SDR objects with Google Vision.")
    ap.add_argument("csv", help='CSV with a "Druid" column')
    ap.add_argument("--out", default="export", help="output directory (default: export)")
    ap.add_argument("--project", default="sul-ai-sandbox", help="GCP project billed for Vision")
    ap.add_argument("--workers", type=int, default=6, help="parallel Vision requests per object")
    a = ap.parse_args()

    druids = [r["Druid"].strip() for r in csv.DictReader(open(a.csv, encoding="utf-8"))
              if r.get("Druid", "").strip()]
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    failures = []
    start = time.time()
    for n, druid in enumerate(druids, 1):
        objdir = out / druid
        objdir.mkdir(exist_ok=True)
        try:
            files = list(jp2_files(json.loads(fetch(PURL.format(druid=druid)))))
        except Exception as e:
            print(f"[{n}/{len(druids)}] {druid}: cocina FAILED {e}", file=sys.stderr)
            failures.append((druid, "", f"cocina: {e}"))
            continue
        done = sum(1 for f in files if (objdir / f"{Path(f['filename']).stem}.xml").exists())
        elapsed = (time.time() - start) / 60
        print(f"[{n}/{len(druids)}] {druid}: {len(files)} pages ({done} already done), "
              f"{elapsed:.0f} min elapsed", file=sys.stderr, flush=True)
        if done == len(files):
            continue
        # Access tokens last an hour; a fresh one per object keeps a long run going.
        failed = ocr_object(druid, files, objdir, gcloud_token(), a.project, a.workers)
        failures += [(druid, name, "vision") for name in failed]

    with open(out / "failures.tsv", "w", encoding="utf-8") as fh:
        fh.write("druid\tpage\treason\n")
        fh.writelines(f"{d}\t{p}\t{r}\n" for d, p, r in failures)
    print(f"done: {len(druids)} objects, {len(failures)} failures "
          f"(see {out / 'failures.tsv'})", file=sys.stderr)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
