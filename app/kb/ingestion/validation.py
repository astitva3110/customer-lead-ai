"""Document upload validation."""

from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from app.config import settings

ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx"}
EXTENSION_MIME = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    filename: str
    extension: str
    mime_type: str
    file_hash: str
    size_bytes: int
    error: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_mime(path: Path, extension: str) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or EXTENSION_MIME.get(extension, "application/octet-stream")


def validate_upload_file(path: Path, *, max_bytes: int | None = None) -> ValidationResult:
    if not path.exists():
        return ValidationResult(
            ok=False,
            filename=path.name,
            extension="",
            mime_type="",
            file_hash="",
            size_bytes=0,
            error=f"File not found: {path}",
        )
    if not path.is_file():
        return ValidationResult(
            ok=False,
            filename=path.name,
            extension="",
            mime_type="",
            file_hash="",
            size_bytes=0,
            error=f"Not a regular file: {path}",
        )

    extension = path.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return ValidationResult(
            ok=False,
            filename=path.name,
            extension=extension,
            mime_type="",
            file_hash="",
            size_bytes=path.stat().st_size,
            error=f"Unsupported extension {extension!r}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    size_bytes = path.stat().st_size
    limit = max_bytes if max_bytes is not None else settings.max_upload_bytes
    if size_bytes == 0:
        return ValidationResult(
            ok=False,
            filename=path.name,
            extension=extension,
            mime_type="",
            file_hash="",
            size_bytes=0,
            error="File is empty",
        )
    if size_bytes > limit:
        return ValidationResult(
            ok=False,
            filename=path.name,
            extension=extension,
            mime_type="",
            file_hash="",
            size_bytes=size_bytes,
            error=f"File exceeds size limit ({size_bytes} > {limit} bytes)",
        )

    mime_type = detect_mime(path, extension)
    expected = EXTENSION_MIME.get(extension)
    if expected and mime_type not in {expected, "application/octet-stream"}:
        # Allow octet-stream on Windows where MIME detection is unreliable.
        if not mime_type.startswith("application/"):
            return ValidationResult(
                ok=False,
                filename=path.name,
                extension=extension,
                mime_type=mime_type,
                file_hash="",
                size_bytes=size_bytes,
                error=f"MIME type mismatch: got {mime_type!r}, expected {expected!r}",
            )

    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        return ValidationResult(
            ok=False,
            filename=path.name,
            extension=extension,
            mime_type=mime_type,
            file_hash="",
            size_bytes=size_bytes,
            error=f"File is not readable: {exc}",
        )

    file_hash = sha256_file(path)
    return ValidationResult(
        ok=True,
        filename=path.name,
        extension=extension,
        mime_type=mime_type,
        file_hash=file_hash,
        size_bytes=size_bytes,
    )
