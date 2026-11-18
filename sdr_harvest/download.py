from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from urllib.parse import quote

import requests

from .core import StageError, TransientStageError, file_digest, previous_versions
from .manifests import safe_name

# Records which JP2 and IIIF size produced each derivative in ``pdfs/``.
DERIVATIVES_FILENAME = "derivatives.json"
JPEG_MAGIC = b"\xff\xd8\xff"


def derivative_name(filename: str) -> str:
    return f"{Path(safe_name(filename)).stem}.jpg"


def derivative_record(info: dict) -> dict:
    return {
        "source_file": info["filename"],
        "sha1": info.get("sha1"),
        "md5": info.get("md5"),
        "iiif_size": info["iiif_size"],
    }


def read_derivatives(directory: Path) -> dict[str, dict]:
    try:
        return json.loads((directory / DERIVATIVES_FILENAME).read_text())
    except (OSError, ValueError):
        return {}


class FileDownloader:
    """Download and validate the source-file inventory declared by COCINA."""

    def __init__(
        self,
        http: requests.Session,
        *,
        keep_failed_downloads: bool = False,
        stacks_token: str | None = None,
    ) -> None:
        self.http = http
        self.keep_failed_downloads = keep_failed_downloads
        # Sent per request so the token never reaches other hosts on the session.
        self.headers = (
            {"Authorization": f"Bearer {stacks_token}"} if stacks_token else {}
        )

    @staticmethod
    def _matches(path: Path, info: dict) -> bool:
        valid = path.exists() and (
            not info.get("size") or path.stat().st_size == info["size"]
        )
        if valid and info.get("sha1"):
            valid = file_digest(path, "sha1") == info["sha1"]
        if valid and info.get("md5"):
            valid = file_digest(path, "md5") == info["md5"]
        return valid

    def _reuse_previous(
        self, version_dir: Path, filename: str, info: dict, temporary: Path
    ) -> bool:
        """Link a verified copy from an earlier version instead of downloading."""
        if not (info.get("sha1") or info.get("md5")):
            return False
        for previous in previous_versions(version_dir):
            candidate = previous / "pdfs" / filename
            if not self._matches(candidate, info):
                continue
            temporary.unlink(missing_ok=True)
            try:
                # Downloads always replace files by rename, so sharing an inode is safe.
                os.link(candidate, temporary)
            except OSError:
                shutil.copy2(candidate, temporary)
            return True
        return False

    @staticmethod
    def _check_response(response: requests.Response, filename: str) -> None:
        if 300 <= response.status_code < 400:
            raise StageError(
                f"Stacks requires authentication for {filename} "
                f"(HTTP {response.status_code} redirect to "
                f"{response.headers.get('Location')})"
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise TransientStageError(
                f"Stacks HTTP {response.status_code} for {filename}"
            )
        if response.status_code != 200:
            raise StageError(f"Stacks HTTP {response.status_code} for {filename}")

    def _download_derivative(
        self,
        druid: str,
        info: dict,
        output: Path,
        version_dir: Path,
        current: dict[str, dict],
    ) -> str:
        """Fetch a IIIF JPEG derivative of a JP2; COCINA digests describe the JP2."""
        name = derivative_name(info["filename"])
        record = derivative_record(info)
        target = output / name
        if target.exists() and current.get(name) == record:
            return name
        temporary = target.with_suffix(".jpg.tmp")
        for previous in previous_versions(version_dir):
            candidate = previous / "pdfs" / name
            if candidate.exists() and read_derivatives(previous / "pdfs").get(
                name
            ) == record:
                temporary.unlink(missing_ok=True)
                try:
                    os.link(candidate, temporary)
                except OSError:
                    shutil.copy2(candidate, temporary)
                temporary.replace(target)
                return name
        identifier = quote(f"{druid}/{Path(info['filename']).stem}", safe="")
        try:
            with self.http.get(
                f"https://stacks.stanford.edu/image/iiif/{identifier}/full/"
                f"{info['iiif_size']}/0/default.jpg",
                headers=self.headers,
                timeout=120,
                stream=True,
                allow_redirects=False,
            ) as response:
                self._check_response(response, name)
                with temporary.open("wb") as stream:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            stream.write(block)
            with temporary.open("rb") as stream:
                if stream.read(len(JPEG_MAGIC)) != JPEG_MAGIC:
                    raise StageError(f"IIIF response for {name} is not a JPEG")
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return name

    def run(self, druid: str, files: list[dict], version_dir: Path) -> Path:
        output = version_dir / "pdfs"
        output.mkdir(parents=True, exist_ok=True)
        expected: set[str] = set()
        names = [
            derivative_name(info["filename"])
            if info.get("iiif_size")
            else safe_name(info["filename"])
            for info in files
        ]
        if len(names) != len(set(names)):
            raise StageError("COCINA contains duplicate source filenames")
        current_derivatives = read_derivatives(output)
        derivatives: dict[str, dict] = {}
        for info in files:
            if info.get("iiif_size"):
                name = self._download_derivative(
                    druid, info, output, version_dir, current_derivatives
                )
                expected.add(name)
                derivatives[name] = derivative_record(info)
                continue
            filename = safe_name(info["filename"])
            expected.add(filename)
            target = output / filename
            if self._matches(target, info):
                continue
            temporary = target.with_suffix(target.suffix + ".tmp")
            if self._reuse_previous(version_dir, filename, info, temporary):
                temporary.replace(target)
                continue
            try:
                with self.http.get(
                    f"https://stacks.stanford.edu/file/{druid}/"
                    f"{quote(info['filename'], safe='')}",
                    headers=self.headers,
                    timeout=120,
                    stream=True,
                    allow_redirects=False,
                ) as response:
                    self._check_response(response, filename)
                    sha1 = hashlib.sha1()
                    md5 = hashlib.md5()
                    actual_size = 0
                    with temporary.open("wb") as stream:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if not block:
                                continue
                            stream.write(block)
                            sha1.update(block)
                            md5.update(block)
                            actual_size += len(block)
                actual_sha1 = sha1.hexdigest()
                actual_md5 = md5.hexdigest()
                mismatched = (
                    (info.get("size") is not None and actual_size != info["size"])
                    or (info.get("sha1") and actual_sha1 != info["sha1"])
                    or (info.get("md5") and actual_md5 != info["md5"])
                )
                if mismatched:
                    retained = "deleted"
                    if self.keep_failed_downloads:
                        invalid = target.with_suffix(target.suffix + ".invalid")
                        temporary.replace(invalid)
                        retained = str(invalid)
                    else:
                        temporary.unlink(missing_ok=True)
                    raise StageError(
                        f"Download integrity mismatch for {filename}; "
                        f"size expected={info.get('size')} actual={actual_size}; "
                        f"SHA-1 expected={info.get('sha1')} actual={actual_sha1}; "
                        f"MD5 expected={info.get('md5')} actual={actual_md5}; "
                        f"invalid_file={retained}"
                    )
                temporary.replace(target)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        if derivatives:
            # Written last so it only describes derivatives that finished.
            manifest = output / DERIVATIVES_FILENAME
            temporary = manifest.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(derivatives, sort_keys=True))
            temporary.replace(manifest)
            expected.add(DERIVATIVES_FILENAME)
        for stale in output.iterdir():
            if stale.is_file() and stale.name not in expected:
                stale.unlink()
        return output
