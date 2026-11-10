from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from .core import StageError
from .manifests import cocina_caption_media_files, cocina_public_files


class SolrDocumentBuilder:
    """Create one nested parent/child Solr JSON document from embeddings."""

    def run(self, druid: str, source_fp: str, version_dir: Path) -> Path:
        metadata = json.loads((version_dir / "metadata.json").read_text())
        metadata["id"] = druid
        metadata["doc_type_ssi"] = "parent"
        metadata["pipeline_fingerprint_ss"] = source_fp
        cocina = json.loads((version_dir / "cocina.json").read_text())
        public_files = cocina_public_files(cocina)
        caption_media = cocina_caption_media_files(cocina)
        table = pq.read_table(version_dir / "embeddings.parquet").to_pylist()
        children = []
        for row in table:
            filename = (
                "_metadata_"
                if row["file"] == "_metadata_"
                else row.get("source_file")
                or Path(row["file"]).with_suffix(".pdf").name
            )
            base = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in filename
            )
            child = {
                "id": f"{druid}_{base}_c{row['chunk_index']}",
                "chunk_text_tesi": row["text"],
                "vector": row["embedding"],
                "chunk_index_i": row["chunk_index"],
                # For VTTs, use the filename of the media file they
                # go with. Otherwise use the file's own filename
                "filename_ss": caption_media.get(filename, filename),
                "doc_type_ssi": "child",
                # Object metadata is always public; file text follows access.view.
                "public_bsi": filename == "_metadata_" or filename in public_files,
            }
            if row.get("page") is not None:
                child["page_ss"] = str(row["page"])
            if row.get("start_seconds") is not None:
                child["caption_file_ss"] = filename
                child["start_seconds_i"] = math.floor(row["start_seconds"])
                child["end_seconds_i"] = math.ceil(row["end_seconds"])
            children.append(child)
        ids = [child["id"] for child in children]
        if len(ids) != len(set(ids)):
            raise StageError("Generated duplicate child IDs")
        metadata["_childDocuments_"] = children
        metadata["child_count_i"] = len(children)
        output = version_dir / "solr.json"
        output.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
        return output
