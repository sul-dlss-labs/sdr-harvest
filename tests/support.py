import hashlib

DRUID = "ab123cd4567"


def cocina(version=1, digest="abc"):
    return {
        "externalIdentifier": f"druid:{DRUID}",
        "version": version,
        "description": {"title": [{"value": "Example"}]},
        "structural": {
            "contains": [
                {
                    "structural": {
                        "contains": [
                            {
                                "externalIdentifier": "file:1",
                                "filename": "example.pdf",
                                "hasMimeType": "application/pdf",
                                "access": {"view": "world"},
                                "size": 3,
                                "version": 1,
                                "hasMessageDigests": [
                                    {"type": "sha1", "digest": digest}
                                ],
                            }
                        ]
                    }
                }
            ]
        },
    }


def file_info(content: bytes) -> dict:
    return {
        "filename": "example.pdf",
        "size": len(content),
        "sha1": hashlib.sha1(content).hexdigest(),
        "md5": hashlib.md5(content).hexdigest(),
    }


class StreamingResponse:
    def __init__(
        self, content: bytes, status_code: int = 200, headers: dict | None = None
    ):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]
