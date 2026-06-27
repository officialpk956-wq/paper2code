import httpx
import os
import logging
from typing import Optional



log = logging.getLogger(__name__)

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_URL = "https://api.resend.com/emails"
FROM_ADDRESS = os.getenv("EMAIL_FROM", "Paper2Code <noreply@paper2code.com>")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

async def _send(to: str, subject: str, html: str) -> bool:
    if not RESEND_API_KEY:
        log.warning(f"[email] RESEND_API_KEY not set — would send '{subject}' to {to}")
        return False
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                RESEND_URL,
                headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
                json={"from": FROM_ADDRESS, "to": [to],
                      "subject": subject, "html": html},
            )
            r.raise_for_status()
            return True
    except Exception as e:
        log.error(f"[email] send failed to {to}: {e}")
        return False

async def send_verification_email(to: str, token: str) -> bool:
    link = f"{FRONTEND_URL}/auth/verify-email?token={token}"
    html = f"""
    <div style="font-family:Inter,sans-serif;max-width:560px;margin:auto;padding:32px">
      <h1 style="font-size:24px;color:#0F172A">Verify your Paper2Code account</h1>
      <p style="color:#475569">Click the button below to confirm your email address.
         This link expires in 24 hours.</p>
      <a href="{link}" style="display:inline-block;margin:24px 0;padding:12px 24px;
         background:#7C3AED;color:#fff;border-radius:8px;text-decoration:none;font-weight:600">
        Verify Email →
      </a>
      <p style="color:#94A3B8;font-size:12px">If you didn't create an account, ignore this email.</p>
    </div>"""
    return await _send(to, "Verify your Paper2Code email", html)

async def send_password_reset_email(to: str, token: str) -> bool:
    link = f"{FRONTEND_URL}/auth/reset-password?token={token}"
    html = f"""
    <div style="font-family:Inter,sans-serif;max-width:560px;margin:auto;padding:32px">
      <h1 style="font-size:24px;color:#0F172A">Reset your password</h1>
      <p style="color:#475569">Click below to set a new password. 
         This link expires in <strong>15 minutes</strong> and can only be used once.</p>
      <a href="{link}" style="display:inline-block;margin:24px 0;padding:12px 24px;
         background:#7C3AED;color:#fff;border-radius:8px;text-decoration:none;font-weight:600">
        Reset Password →
      </a>
      <p style="color:#94A3B8;font-size:12px">If you didn't request this, you can safely ignore it.
         Your password won't change.</p>
    </div>"""
    return await _send(to, "Reset your Paper2Code password", html)

async def send_welcome_email(to: str, name: str) -> bool:
    html = f"""
    <div style="font-family:Inter,sans-serif;max-width:560px;margin:auto;padding:32px">
      <h1 style="font-size:24px;color:#0F172A">Welcome to Paper2Code, {name}!</h1>
      <p style="color:#475569">You're all set. Here's what to try first:</p>
      <ul style="color:#475569;line-height:2">
        <li>Upload a research paper and get PyTorch code in minutes</li>
        <li>Solve a Dojo problem to test your ML skills</li>
        <li>Ask the AI Tutor anything about an architecture</li>
      </ul>
      <a href="{FRONTEND_URL}/dashboard" style="display:inline-block;margin:24px 0;
         padding:12px 24px;background:#7C3AED;color:#fff;border-radius:8px;
         text-decoration:none;font-weight:600">
        Open Dashboard →
      </a>
    </div>"""
    return await _send(to, "Welcome to Paper2Code", html)


async def send_paper_done_email(to: str, paper_name: str, paper_id: int) -> bool:
    link = f"{FRONTEND_URL}/papers/{paper_id}"
    html = f"""
    <div style="font-family:Inter,sans-serif;max-width:560px;margin:auto;padding:32px">
      <h1 style="font-size:24px;color:#0F172A">Your paper is ready</h1>
      <p style="color:#475569">
        <strong>{paper_name}</strong> has been processed and PyTorch code has been generated.
      </p>
      <a href="{link}" style="display:inline-block;margin:24px 0;padding:12px 24px;
         background:#7C3AED;color:#fff;border-radius:8px;text-decoration:none;font-weight:600">
        View Code →
      </a>
      <p style="color:#94A3B8;font-size:12px">Paper2Code — turn research into runnable code.</p>
    </div>"""
    return await _send(to, f"Paper ready: {paper_name}", html)


_DRIP_TEMPLATES = {
    1: {
        "subject": "Welcome! Here's how to get started with Paper2Code",
        "cta_text": "Upload Your First Paper →",
        "cta_url": "/papers",
        "body": "Start by uploading any research paper PDF. We'll extract the architecture and generate PyTorch code within minutes.",
    },
    3: {
        "subject": "Ready to test your ML skills? Try the Dojo",
        "cta_text": "Open the Dojo →",
        "cta_url": "/dojo",
        "body": "The Coding Dojo has hand-crafted problems on Transformers, CNNs, and more. Solve one today to build your streak.",
    },
    7: {
        "subject": "7 days with Paper2Code — check the leaderboard",
        "cta_text": "View Leaderboard →",
        "cta_url": "/leaderboard",
        "body": "You've been with us for a week! See how you rank against other researchers and practitioners.",
    },
}


def send_drip_email_sync(to: str, name: str, day: int) -> bool:
    tpl = _DRIP_TEMPLATES.get(day)
    if not tpl:
        return False
    if not RESEND_API_KEY:
        log.warning("[email] RESEND_API_KEY not set — skipping drip day=%d to %s", day, to)
        return False
    link = f"{FRONTEND_URL}{tpl['cta_url']}"
    html = f"""
    <div style="font-family:Inter,sans-serif;max-width:560px;margin:auto;padding:32px">
      <h1 style="font-size:24px;color:#0F172A">Hi {name}!</h1>
      <p style="color:#475569">{tpl['body']}</p>
      <a href="{link}" style="display:inline-block;margin:24px 0;padding:12px 24px;
         background:#7C3AED;color:#fff;border-radius:8px;text-decoration:none;font-weight:600">
        {tpl['cta_text']}
      </a>
      <p style="color:#94A3B8;font-size:12px">Paper2Code — turn research into runnable code.</p>
    </div>"""
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(
                RESEND_URL,
                headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
                json={"from": FROM_ADDRESS, "to": [to], "subject": tpl["subject"], "html": html},
            )
            r.raise_for_status()
            return True
    except Exception as exc:
        log.error("[email] drip day=%d email failed to %s: %s", day, to, exc)
        return False


def send_streak_at_risk_email_sync(to: str, name: str, streak: int) -> bool:
    if not RESEND_API_KEY:
        log.warning("[email] RESEND_API_KEY not set — skipping streak-at-risk email to %s", to)
        return False
    html = f"""
    <div style="font-family:Inter,sans-serif;max-width:560px;margin:auto;padding:32px">
      <h1 style="font-size:24px;color:#0F172A">Your {streak}-day streak is at risk, {name}!</h1>
      <p style="color:#475569">
        You haven't been active today. Log in now to keep your streak alive — it resets at midnight UTC.
      </p>
      <a href="{FRONTEND_URL}/dojo" style="display:inline-block;margin:24px 0;padding:12px 24px;
         background:#DC2626;color:#fff;border-radius:8px;text-decoration:none;font-weight:600">
        Save My Streak →
      </a>
      <p style="color:#94A3B8;font-size:12px">Paper2Code — daily practice builds expertise.</p>
    </div>"""
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(
                RESEND_URL,
                headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
                json={"from": FROM_ADDRESS, "to": [to],
                      "subject": f"🔥 Your {streak}-day streak is at risk!", "html": html},
            )
            r.raise_for_status()
            return True
    except Exception as exc:
        log.error("[email] streak-at-risk email failed to %s: %s", to, exc)
        return False


def send_paper_done_email_sync(to: str, paper_name: str, paper_id: int) -> bool:
    """Synchronous wrapper for use inside Celery tasks."""
    if not RESEND_API_KEY:
        log.warning("[email] RESEND_API_KEY not set — skipping paper-done email to %s", to)
        return False
    try:
        link = f"{FRONTEND_URL}/papers/{paper_id}"
        html = f"""
        <div style="font-family:Inter,sans-serif;max-width:560px;margin:auto;padding:32px">
          <h1 style="font-size:24px;color:#0F172A">Your paper is ready</h1>
          <p style="color:#475569">
            <strong>{paper_name}</strong> has been processed and PyTorch code has been generated.
          </p>
          <a href="{link}" style="display:inline-block;margin:24px 0;padding:12px 24px;
             background:#7C3AED;color:#fff;border-radius:8px;text-decoration:none;font-weight:600">
            View Code →
          </a>
        </div>"""
        with httpx.Client(timeout=10.0) as client:
            r = client.post(
                RESEND_URL,
                headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
                json={"from": FROM_ADDRESS, "to": [to],
                      "subject": f"Paper ready: {paper_name}", "html": html},
            )
            r.raise_for_status()
            return True
    except Exception as exc:
        log.error("[email] sync paper-done email failed to %s: %s", to, exc)
        return False
