import os


def is_email_allowed(email: str) -> bool:
    """Return True if this email may register or sign in.

    Controlled by the ALLOWLIST_EMAILS env var (comma-separated list of
    emails). When it is empty or unset the allowlist is DISABLED and everyone
    is allowed — so local/dev and normal open-signup deployments are
    unaffected. Set ALLOWLIST_EMAILS to lock the site to a private preview.
    """
    raw = os.getenv("ALLOWLIST_EMAILS", "").strip()
    if not raw:
        return True
    allowed = {e.strip().lower() for e in raw.split(",") if e.strip()}
    return (email or "").strip().lower() in allowed
