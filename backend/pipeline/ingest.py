"""Stage 0 — resolve an input into a local file (accepts a path or an HTTP(S)/S3 URL)."""
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..config import DOWNLOAD_DIR


def resolve_input(src: str) -> Path:
    """Return a local Path for `src`, downloading it first if it's a URL.

    Supports local file paths and http(s) URLs, including presigned/public S3
    links (e.g. https://bucket.s3.amazonaws.com/key?X-Amz-...). Private s3://
    URIs (needing AWS creds) are intentionally out of scope for now.
    """
    parsed = urlparse(src)

    if parsed.scheme in ("http", "https"):
        return _download(src, parsed)

    if parsed.scheme == "s3":
        raise ValueError(
            "Private s3:// URIs aren't supported yet. Use a presigned/public "
            "https:// URL, or a local file path."
        )

    p = Path(src).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {src}")
    return p


def _download(url: str, parsed) -> Path:
    import os
    import uuid
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Name the file from the URL path (strip query string, keep extension).
    name = Path(unquote(parsed.path)).name or "download"
    dest = DOWNLOAD_DIR / name
    # Already fetched by another job (same recording)? reuse it — dedup.
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    # Download to a per-call temp file, then atomically rename — so concurrent
    # workers fetching the same URL can't corrupt each other's file.
    tmp = DOWNLOAD_DIR / f".{uuid.uuid4().hex}.{name}.part"
    print(f"  ↓ downloading {url.split('?')[0]}")
    req = urllib.request.Request(url, headers={"User-Agent": "asr-tool/1.0"})
    try:
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
            while chunk := resp.read(1 << 20):  # 1 MB chunks
                f.write(chunk)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    size_mb = dest.stat().st_size / 1e6
    print(f"  ✓ saved {dest.name} ({size_mb:.1f} MB)")
    return dest
