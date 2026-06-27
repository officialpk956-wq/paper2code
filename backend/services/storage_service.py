"""
backend/services/storage_service.py

Abstraction over PDF storage backends:
  - Cloudflare R2 (boto3 S3-compatible) when R2_* env vars are set
  - Local tempfile fallback for dev/test when R2 is not configured

Storage refs:
  - R2 mode:    "r2://papers/{uuid}_{filename}.pdf"
  - Local mode: absolute path to a tempfile (e.g. "/tmp/xxx.pdf")
"""

import os
import uuid
import logging
import tempfile

log = logging.getLogger(__name__)

_R2_ACCOUNT_ID       = os.getenv("R2_ACCOUNT_ID", "")
_R2_ACCESS_KEY_ID    = os.getenv("R2_ACCESS_KEY_ID", "")
_R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
_R2_BUCKET_NAME      = os.getenv("R2_BUCKET_NAME", "")
_R2_PUBLIC_URL       = os.getenv("R2_PUBLIC_URL", "")

R2_AVAILABLE = bool(
    _R2_ACCOUNT_ID and _R2_ACCESS_KEY_ID and _R2_SECRET_ACCESS_KEY and _R2_BUCKET_NAME
)


def _r2_client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=f"https://{_R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=_R2_ACCESS_KEY_ID,
        aws_secret_access_key=_R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def store_pdf(pdf_bytes: bytes, filename: str) -> str:
    """
    Persist PDF bytes and return a storage_ref string that can later
    be passed to fetch_pdf() and cleanup().
    """
    safe = os.path.basename(filename).replace(" ", "_")
    if R2_AVAILABLE:
        key = f"papers/{uuid.uuid4().hex}_{safe}"
        _r2_client().put_object(
            Bucket=_R2_BUCKET_NAME,
            Key=key,
            Body=pdf_bytes,
            ContentType="application/pdf",
        )
        log.info("Stored PDF in R2: %s", key)
        return f"r2://{key}"

    # Local fallback
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        path = tmp.name
    log.info("Stored PDF locally (R2 not configured): %s", path)
    return path


def fetch_pdf(storage_ref: str) -> bytes:
    """Download/read PDF bytes from a storage_ref."""
    if storage_ref.startswith("r2://"):
        key = storage_ref[5:]
        resp = _r2_client().get_object(Bucket=_R2_BUCKET_NAME, Key=key)
        return resp["Body"].read()
    with open(storage_ref, "rb") as fh:
        return fh.read()


def cleanup(storage_ref: str) -> None:
    """Delete the stored PDF after processing. Never raises."""
    try:
        if storage_ref.startswith("r2://"):
            _r2_client().delete_object(Bucket=_R2_BUCKET_NAME, Key=storage_ref[5:])
        else:
            os.unlink(storage_ref)
    except Exception as exc:
        log.warning("Storage cleanup failed for %s: %s", storage_ref, exc)


def presigned_download_url(storage_ref: str, expires_in: int = 3600) -> str:
    """
    Return a time-limited URL to download the PDF.
    Returns "" for local refs (not applicable outside dev).
    """
    if not storage_ref.startswith("r2://"):
        return ""
    key = storage_ref[5:]
    if _R2_PUBLIC_URL:
        return f"{_R2_PUBLIC_URL.rstrip('/')}/{key}"
    return _r2_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": _R2_BUCKET_NAME, "Key": key},
        ExpiresIn=expires_in,
    )


def r2_key_from_ref(storage_ref: str):
    """Return the R2 object key, or None for local refs."""
    if storage_ref.startswith("r2://"):
        return storage_ref[5:]
    return None


def generate_presigned_upload_url(
    filename: str,
    content_type: str = "application/pdf",
    expires_in: int = 900,
) -> dict:
    """
    Generate a presigned S3 PUT URL for direct client → R2 upload.
    The API server never touches the bytes; only the key is queued to Celery.
    Returns {url, key, expires_in} or raises RuntimeError if R2 not configured.
    """
    if not R2_AVAILABLE:
        raise RuntimeError("R2 not configured — presigned uploads unavailable")
    safe = os.path.basename(filename).replace(" ", "_")
    key = f"papers/{uuid.uuid4().hex}_{safe}"
    url = _r2_client().generate_presigned_url(
        "put_object",
        Params={
            "Bucket": _R2_BUCKET_NAME,
            "Key": key,
            "ContentType": content_type,
        },
        ExpiresIn=expires_in,
    )
    log.info("Generated presigned upload URL for key: %s", key)
    return {"url": url, "key": key, "expires_in": expires_in}


def get_object_size(key: str) -> int:
    """Return the byte size of an R2 object, or 0 on error."""
    if not R2_AVAILABLE:
        return 0
    try:
        resp = _r2_client().head_object(Bucket=_R2_BUCKET_NAME, Key=key)
        return resp.get("ContentLength", 0)
    except Exception as exc:
        log.warning("get_object_size failed for %s: %s", key, exc)
        return 0
