"""
Upload a PDF file to an HTTP endpoint.

Usage:
    python upload.py /path/to/file.pdf https://example.com/upload
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _is_pdf(file_path: Path) -> bool:
    """
    Validate that the file is a PDF.
    Checks extension and magic bytes.
    """
    if file_path.suffix.lower() != ".pdf":
        return False

    try:
        with file_path.open("rb") as f:
            return f.read(4) == b"%PDF"
    except OSError:
        return False


def upload_pdf(file_path: str, upload_url: str, form_field: str = "file") -> dict:
    """
    Upload a PDF file to a given URL using multipart/form-data.
    Returns the JSON response if available, otherwise returns text.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if not _is_pdf(path):
        raise ValueError(f"File is not a valid PDF: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    content_type = mime_type or "application/pdf"

    with path.open("rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    boundary = f"----PDFUploadBoundary{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{form_field}"; filename="{path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + pdf_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    request = Request(upload_url, data=body, headers=headers, method="POST")

    try:
        with urlopen(request, timeout=60) as response:
            response_bytes = response.read()
            status_code = response.getcode()
    except HTTPError as e:
        raise RuntimeError(f"Upload failed with status {e.code}: {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Unable to reach upload URL: {e.reason}") from e

    response_text = response_bytes.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(response_text) if response_text else {}
    except json.JSONDecodeError:
        parsed = {"response_text": response_text}

    parsed["status_code"] = status_code
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a PDF file to an API endpoint.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("upload_url", help="Upload endpoint URL")
    parser.add_argument(
        "--field",
        default="file",
        help="Form field name expected by server (default: file)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = upload_pdf(args.pdf_path, args.upload_url, args.field)
    print("Upload successful.")
    print(result)


if __name__ == "__main__":
    main()
