# SDR Harvest

SDR Harvest builds searchable Solr documents from Stanford Digital Repository
objects.

The pipeline has four broad phases:

1. **Acquire** — Read the manifest, fetch Cocina metadata, and download
   validated source files.
2. **Prepare** — Extract text, transform metadata, and divide the content into
   overlapping chunks.
3. **Enrich** — Generate a semantic vector embedding for every chunk.
4. **Index** — Create nested parent/child Solr documents and publish them to the
   search collection.

Every build stage is checkpointed and resumable; publishing is a separate,
explicit operation.

![SDR Harvest pipeline: Acquire, Prepare, Enrich, and Index](docs/pipeline-overview.svg)

## Pipeline overview

Each of the four phases is made of one or more stages. Every stage reads from
and writes to the state directory (`.sdr-harvest/` by default), and records its
outcome in the SQLite database `state.sqlite3`.

### Where data lives

| Location | Contents |
| --- | --- |
| `state.sqlite3` → `objects` | One row per DRUID: manifest membership, source fingerprint, PURL validators (ETag, Last-Modified, last check time), and the current version directory. |
| `state.sqlite3` → `source_files` | The source-file inventory selected from COCINA, with size, SHA-1, and MD5. |
| `state.sqlite3` → `stage_state` | One row per DRUID and stage: status, input and output fingerprints, stage signature, and artifact path. This is the build cache index. |
| `state.sqlite3` → `attempts`, `runs` | A history of every stage attempt and every `run`. |
| `state.sqlite3` → `publications` | One row per DRUID and Solr target URL: the published document fingerprint and status. |
| `sources/<druid>/cocina.json` | The most recently fetched COCINA for an object. |
| `versions/<druid>/<source fingerprint>/` | The build artifacts for one version of an object. |
| `logs/<run id>/<druid>/<stage>.jsonl` | Per-attempt event logs. |

The **source fingerprint** is a SHA-256 of the complete COCINA record together
with the selected source-file inventory. Any change to an object in SDR,
including a metadata-only change, produces a new source fingerprint and
therefore a new version directory. Earlier version directories are left in
place until they are pruned.

### Cache keys

A build stage is skipped when its `stage_state` row shows that it:

1. succeeded;
2. was given the same **input fingerprint**, which is the output fingerprint of
   the preceding stage (the first stage after COCINA receives the source
   fingerprint);
3. ran with the same **stage signature**, a version string in
   `sdr_harvest/core.py` (`SIGNATURES`) that is changed whenever a stage's
   behavior changes; and
4. wrote its artifact inside the current version directory.

An output fingerprint is the SHA-256 of the stage's artifact file, or of the
names and contents of every file in its artifact directory. Because the
fingerprints form a chain, a stage whose output changes causes every later
stage to run again, while a stage whose output is identical lets later stages
remain current. Changing a signature reruns that stage and, if its output
changes, everything downstream.

A new version directory starts with no current stages. To avoid repeating
expensive work after a metadata-only change, the download, extract, and embed
stages also look in the object's earlier version directories for reusable
results, as described below.

### Acquire

**Manifest reconciliation.** `run` reads the manifest, removes DRUIDs listed in
`dark.csv`, and marks each object in `objects` as present or absent. Before
queuing work it compares every object's `stage_state` rows against the cache
keys above, so fully current objects are counted as complete without being
processed.

**COCINA (`cocina`).** Fetches `https://purl.stanford.edu/<druid>.json` into
`sources/<druid>/cocina.json`, then selects the source files to process (page
ALTO XML for complete books, otherwise PDFs) and stores them in `source_files`.

- Cache key: the DRUID. The cached record is reused without a network request
  if its SHA-256 still matches `objects.source_cache_sha256`, it was checked
  within the last seven days, and the file-selection rules
  (`SOURCE_INVENTORY_SIGNATURE`) are unchanged. Otherwise PURL is asked with a
  conditional GET using the stored ETag or Last-Modified value; a `304` keeps
  the cached record.
- Output: the source fingerprint. When it changes, all downstream stages are
  reset to pending and the object is built into a new version directory, which
  receives a copy of `cocina.json`.

**Download (`download`).** Streams each selected file from
`https://stacks.stanford.edu/file/<druid>/<filename>` into
`versions/<druid>/<fp>/pdfs/` and validates its size, SHA-1, and MD5 against
COCINA. ALTO XML files are stored in this directory too.

- Cache key: the source fingerprint plus the `download` signature.
- Reuse: a file already present with matching checksums is kept, and a
  matching file in an earlier version directory is hard-linked (or copied)
  instead of downloaded again.
- Output fingerprint: the contents of `pdfs/`.

### Prepare

**Metadata (`metadata`).** Runs Traject with `sdr_config.rb` over the version's
`cocina.json` to produce the searchable Solr fields in `metadata.json`.

- Cache key: the download output fingerprint plus the `metadata` signature.
- Output fingerprint: the SHA-256 of `metadata.json`.

**Extract (`extract`).** Converts the source files to Markdown in `markdown/`,
using the ALTO strategy for complete books and PyMuPDF4LLM text extraction
otherwise. `markdown/pages.json` maps each Markdown file to its source file and
page ranges.

- Cache key: the metadata output fingerprint plus the signature of the
  extraction strategy selected for the object, so ALTO and PDF objects are
  versioned independently.
- Reuse: `markdown/extract.json` records the strategy signature and a
  fingerprint of the downloaded files and COCINA page numbers. If an earlier
  version directory has an identical marker, its Markdown is copied rather than
  extracted again. Descriptive metadata is not part of this key, so a
  metadata-only COCINA change does not repeat extraction.
- Output fingerprint: the contents of `markdown/`.

**Chunk (`chunk`).** Splits each Markdown file into overlapping chunks (1,500
characters with 200 characters of overlap), adds one chunk rendered from
`metadata.json`, and writes `chunks.parquet` with the source file, page, and
chunk index for each row.

- Cache key: the extract output fingerprint plus the `chunk` signature.
- Output fingerprint: the SHA-256 of `chunks.parquet`.

### Enrich

**Embed (`embed`).** Formats every chunk as `title: <title> | text: <chunk>`,
requests Gemini vectors through LiteLLM, and writes `embeddings.parquet`.

- Cache key: the chunk output fingerprint plus the `embed` signature, which
  names the model and dimensions.
- Checkpoint: vectors are saved as they arrive in
  `embeddings.checkpoint.sqlite3`, keyed by a hash of the model, dimensions,
  and the ordered input texts. A failed or interrupted stage resumes with only
  the missing chunks.
- Reuse: before calling the API, missing vectors are looked up by exact input
  text in `embeddings.parquet` files of earlier version directories that were
  made with the same `embed` signature (recorded in the Parquet metadata).
  Only chunks whose text or title changed are embedded again.
- Output fingerprint: the SHA-256 of `embeddings.parquet`.

### Index

**Document (`document`).** Combines `metadata.json` and `embeddings.parquet`
into `solr.json`: one parent document carrying the metadata and the source
fingerprint (`pipeline_fingerprint_ss`), with one child document per chunk
carrying its text, vector, filename, and page. After this stage, the object's
`objects.current_artifact_dir` points at the version directory.

- Cache key: the embed output fingerprint plus the `document` signature.
- Output fingerprint: the SHA-256 of `solr.json`, which is the **document
  fingerprint** used by publication.

**Publish (`publish` command).** Not a build stage, and never run by `run`.
It first deletes dark objects from the target, then sends each ready
`solr.json` to Solr as a delete-then-add of the whole nested document, in
batches.

- Cache key: the DRUID and target URL in `publications`. An object is skipped
  when its last publication to that target succeeded with the same document
  fingerprint; `--force` ignores this.
- Output: a receipt, `published-<target hash>.json`, in the version directory,
  and an updated `publications` row.

## Quick start

Install dependencies:

```shell
uv sync
```

Build the corpus, inspect failures, and retry them:

```shell
LITELLM_API_KEY=<key> uv run sdr-harvest run \
  --manifest manifest.csv --workers 7

uv run sdr-harvest status --failed
LITELLM_API_KEY=<key> uv run sdr-harvest retry --failed
```

Building creates local Solr JSON documents. It does not write to Solr. Use the
separate `publish` command after reviewing the build results.

## Prepare the manifest

The pipeline accepts one authoritative manifest. If the desired corpus comes
from multiple CSV exports, merge them into a sorted, deduplicated manifest:

```shell
uv run sdr-harvest merge-manifests \
  world-readable-document-type-with-pdf.csv \
  oral-history-ts561xq4138-druids.csv \
  --output manifest.csv
```

Use the same merged manifest for later build and publish commands. Switching to
a different manifest marks objects omitted from it as absent, but does not
delete their local artifacts or their documents in Solr.

To preview additions, omissions, and known failures:

```shell
uv run sdr-harvest plan --manifest manifest.csv
```

## Build Solr documents

The main build command is:

```shell
LITELLM_API_KEY=<key> uv run sdr-harvest run \
  --manifest manifest.csv --workers 7
```

The progress display begins with fully current objects already counted as
complete and queues only objects that may need work. The command safely reuses
completed downloads, extracted text, chunks, embeddings, and Solr documents.
When an SDR object changes, a new version is built without overwriting its
previous artifacts.

For Cocina books with one transcription XML file for every PDF page, the build
downloads and extracts the page-level ALTO OCR instead of the image-heavy PDFs,
and its chunks name the ALTO file they came from in `filename_ss`.
If the ALTO inventory is incomplete, the build falls back to the selected PDF.
Other object types continue using PDF extraction.

Press Ctrl-C once to cancel queued work and exit. Running the command again
resumes from completed stages. Use `--no-progress` for redirected logs or a
scheduler.

Document vectors are requested in batches of up to 50, with at most two batch
requests in flight, from the Stanford LiteLLM Gemini API passthrough using
`gemini-embedding-2` at 768 dimensions. Inputs are formatted as
`title: <title> | text: <chunk>`. A question-answering client must use the same
model and dimensions and format query text as
`task: question answering | query: <question>`.

Each completed vector is checkpointed in the object's version directory. If an
embedding request fails, a retry resumes with only the missing chunks. A change
to the ordered embedding inputs, model, or dimensions invalidates that
checkpoint automatically.

## Inspect and retry failures

Show every failure or inspect one object:

```shell
uv run sdr-harvest status --failed
uv run sdr-harvest status --druid zd240tq9137
```

Transient service and network failures are retried automatically. Remaining
failures can be retried together:

```shell
LITELLM_API_KEY=<key> uv run sdr-harvest retry --failed
```

PDF downloads are streamed to disk and checked against the size, SHA-1, and
MD5 declared by COCINA. Integrity failures report every expected and calculated
value. The invalid download is deleted by default; retain it for diagnosis by
adding `--keep-failed-downloads` to `run`, `retry`, or `rebuild`. Retained files
are stored beside the intended PDF with an `.invalid` suffix and can be very
large.

To deliberately rebuild one object from a particular stage:

```shell
LITELLM_API_KEY=<key> uv run sdr-harvest rebuild \
  --druid zd240tq9137 --from extract
```

## Publish to Solr

Publish completed documents to staging only after the build has been reviewed:

```shell
uv run sdr-harvest publish \
  --manifest manifest.csv \
  --target https://solr-stage.example.edu/solr/sdr-search \
  --workers 7
```

Publication state is tracked by both DRUID and target URL. Repeating this
command skips unchanged documents already published successfully to that
target, retries failures, and publishes changed documents. Currency is based on
the generated `solr.json` fingerprint, so changes to metadata, chunking,
embeddings, or document construction require publication even when the SDR
source itself is unchanged. Use `--force` to republish documents that are
already current.

Documents are sent in batches of up to 25 objects or 25 MB, whichever limit is
reached first. Each successful update uses `commitWithin=60000`, so the client
does not issue hard commits and documents may take up to one minute to become
searchable. Solr's successful HTTP response is treated as acceptance; the
publisher does not query each document afterward. Use `--batch-size` and
`--max-batch-mb` to tune the limits for a particular Solr installation.

To publish from another machine, copy the application, `manifest.csv`, and the
complete `.sdr-harvest/` directory to it. Then publish to production:

```shell
uv run sdr-harvest publish \
  --manifest manifest.csv \
  --target https://solr-prod.example.edu/solr/sdr-search \
  --workers 7
```

Staging and production have independent publication records, so success on
staging does not cause production to be skipped. Publishing uses the documents
already built and does not require a LiteLLM key.

When publishing through a trusted local SSH tunnel, a `localhost` URL will not
match the certificate issued to the remote Solr hostname. TLS verification can
be disabled explicitly for that tunnel:

```shell
uv run sdr-harvest publish \
  --manifest manifest.csv \
  --target https://localhost:8983/solr/semantic-search-demo \
  --workers 4 \
  --insecure
```

`--insecure` accepts any certificate presented through the connection. Use it
only with a tunnel you created and trust; normal publication keeps certificate
verification enabled.

We often use:
```shell
ssh -N -L 8983:<remote_host>:443 semantic-search-demo.stanford.edu
```

## Evaluate retrieval quality

To inspect one query manually, use the same semantic retrieval path as the
evaluator:

```shell
LITELLM_API_KEY=<key> uv run sdr-harvest query \
  "What was Stanford's role in the development of Silicon Valley?"
```

The command displays the ranked DRUIDs with their similarity scores, source
filenames, chunk indexes, and matching text. Results are collapsed to the best
chunk per DRUID, just as they are during evaluation. Use `--limit 20` to show
more objects, `--candidates 200` to expand the chunk candidate pool, or
`--json` for machine-readable output. To query another collection, pass its
full URL with `--target`.

Keep a reviewed set of natural-language queries and relevant SDR objects in a
JSON array of test cases (see
[`evaluations/test-cases-schema.md`](evaluations/test-cases-schema.md) for the
full field reference). Each case has a stable test ID, a query, and a
`judgments` array of `{document_id, score, explanation}` graded `0`-`3`, with
`3` for highly relevant, `2` for relevant, and `1` for marginally relevant:

```json
[
  {"test_id":"water-policy-1","subject_id":"water-policy","query":"How did water policy affect California agriculture?","judgments":[{"document_id":"cc333dd4444","score":3,"explanation":"..."},{"document_id":"ee555ff6666","score":1,"explanation":"..."}]}
]
```

Test cases live under [`evaluations/judgments/`](evaluations/judgments/),
one file per subject named after its `subject_id` (e.g. `shs-oral-histories.json`).
Create a new file for a new subject, or add miscellaneous judgments to `judgments.json`.

Run an evaluation against the local collection and save it as the baseline for
the current pipeline. Pointing `--judgments` at the `judgments` directory
evaluates every subject together; pointing it at one file evaluates just that
subject.

```shell
LITELLM_API_KEY=<key> uv run sdr-harvest evaluate \
  --judgments evaluations/judgments \
  --output evaluations/baseline.json
```
Optionally add `--test-type <test_type>` to further restrict the run to test cases of that
type (e.g., `--test-type known_item` or any value from the
[`test_type` vocabulary](./evaluations/test-cases-schema.md#test_type-vocabulary)).

The evaluator embeds queries with the same Gemini model as the pipeline,
retrieves chunk candidates from Solr, and collapses them by DRUID before
scoring. This prevents documents with more chunks from occupying multiple
ranking positions. It reports these metrics at ranks 1, 5, and 10 by default:

- `success`: fraction of queries with at least one relevant result;
- `recall`: fraction of all judged-relevant objects retrieved;
- `mrr`: how early the first relevant result appears;
- `ndcg`: ranking quality that gives more credit to highly graded results.

After changing extraction, chunking, metadata, embeddings, or Solr behavior,
rebuild and publish the affected corpus, then compare with the saved baseline:

```shell
LITELLM_API_KEY=<key> uv run sdr-harvest evaluate \
  --judgments evaluations/judgments \
  --output evaluations/candidate.json \
  --baseline evaluations/baseline.json \
  --fail-on-regression ndcg@10 \
  --regression-tolerance 0.01
```

The command exits with status 2 when the selected metric decreases by more than
the tolerance, which makes it suitable for a release check. The full JSON
report includes aggregate deltas and each query's ranked DRUID, best chunk,
score, filename, and snippet for diagnosis. Use `--cutoffs` to change the
reported ranks and `--candidates` to change how many chunks are retrieved
before collapsing.

Treat the judgment files as a maintained test fixture: include representative
query types, difficult terminology, known near-misses, and more than one
relevant object where appropriate. Keep the files fixed while comparing two
pipeline versions; the evaluator rejects a baseline made with different
judgments or metrics.

## State directory

All managed state is stored in `.sdr-harvest/` (see
[Pipeline overview](#pipeline-overview) for its layout). To store it elsewhere,
place the global option before the command:

```shell
uv run sdr-harvest --state-dir /data/sdr-harvest run \
  --manifest manifest.csv --workers 7
```

Only one build or publish process may use a state directory at a time.

## Maintenance and scheduling

The `run` command exits nonzero when an object fails, making it suitable for
cron or a systemd timer. Do not overlap scheduled runs; use `--workers` for
concurrency within one run.

Removing an object from a manifest does not remove it from Solr. To disavow
objects permanently, see [Dark objects](#dark-objects).

Old non-current artifact versions can be pruned after a retention period:

```shell
uv run sdr-harvest prune --failed-before 2026-07-01
```

## Dark objects

An object is made dark when it must be disavowed: removed from every Solr
environment and never harvested again. The committed `dark.csv` file is the
only record kept of it:

```csv
druid,darkened_on
dp867wt7458,2026-10-20
```

To darken objects, add them to the list and delete them from the Solr
environments you can reach:

```shell
uv run sdr-harvest darken \
  --druid dp867wt7458 --druid wq403wh0847 \
  --target http://localhost:8983/solr/sdr-search
```

`darken` adds new DRUIDs to `dark.csv` and then makes everything agree with the
whole list. It removes dark rows from `manifest.csv` (`--manifest` selects
another file), deletes their database rows, cached COCINA, artifacts, and logs
from the state directory, and deletes each object's parent and chunk documents
from every `--target`. Every step is idempotent. Then commit `dark.csv` and
`manifest.csv`.

Other machines and Solr environments catch up from the committed list. Running
`darken` without `--druid` purges the local state directory and the given
targets, for example production through a tunnel:

```shell
uv run sdr-harvest darken \
  --target https://localhost:8983/solr/semantic-search-demo --insecure
```

Dark objects are also handled automatically: `publish` deletes every dark
object from its target before publishing and never publishes one, and `run`,
`plan`, and `merge-manifests` exclude them even when an input CSV still lists
them. Use the global `--dark-list` option to read a different file.

## Developer documentation

Implementation details and module responsibilities are documented in
[`sdr_harvest/README.md`](sdr_harvest/README.md).
