import os
import logging
from typing import Optional
import resend

logger = logging.getLogger(__name__)

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
FROM_EMAIL     = os.getenv("FROM_EMAIL", "Paper2Code <noreply@paper2code.com>")
FRONTEND_URL   = os.getenv("FRONTEND_URL", "http://localhost:5173")

if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY


def send_email_sync(to_email: str, subject: str, html_body: str) -> bool:
    if not RESEND_API_KEY:
        logger.info("MOCK EMAIL to %s: %s", to_email, subject)
        return True
    try:
        resend.Emails.send({"from": FROM_EMAIL, "to": to_email, "subject": subject, "html": html_body})
        return True
    except Exception as e:
        logger.error("Failed to send email to %s: %s", to_email, e)
        return False


def _base_template(content: str) -> str:
    """Branded wrapper applied to every outgoing email."""
    return f"""
<div style="font-family:Inter,system-ui,sans-serif;max-width:560px;margin:40px auto;padding:0 16px">
  <div style="background:#7C3AED;padding:24px 32px;border-radius:12px 12px 0 0">
    <h1 style="margin:0;color:#fff;font-size:20px;font-weight:700">Paper2Code</h1>
  </div>
  <div style="background:#fff;border:1px solid #E2E8F0;border-top:none;
              border-radius:0 0 12px 12px;padding:32px">
    {content}
  </div>
  <p style="text-align:center;color:#94A3B8;font-size:11px;margin-top:16px">
    © 2025 Paper2Code ·
    <a href="{FRONTEND_URL}/unsubscribe" style="color:#94A3B8">Unsubscribe</a>
  </p>
</div>"""


# --- TEMPLATES ---

def send_verification_email_sync(to_email: str, token: str) -> bool:
    link = f"{FRONTEND_URL}/verify-email?token={token}"
    html = _base_template(f"""
      <h2 style="margin:0 0 8px;color:#0F172A;font-size:22px">Verify your email</h2>
      <p style="color:#475569;margin:0 0 24px">
        One click and you're in. This link expires in 24 hours.
      </p>
      <a href="{link}" style="display:inline-block;padding:12px 28px;background:#7C3AED;
         color:#fff;border-radius:8px;text-decoration:none;font-weight:600;font-size:15px">
        Verify Email →
      </a>
      <p style="color:#94A3B8;font-size:12px;margin-top:24px">
        Didn't create an account? You can ignore this email.
      </p>""")
    return send_email_sync(to_email, "Verify your Paper2Code email", html)


def send_welcome_email_sync(to_email: str, name: str = "there") -> bool:
    html = _base_template(f"""
      <h2 style="margin:0 0 8px;color:#0F172A;font-size:22px">Welcome, {name}!</h2>
      <p style="color:#475569;margin:0 0 16px">
        You're now part of Paper2Code. Here's how to get started:
      </p>
      <ul style="color:#475569;padding-left:20px;line-height:1.8">
        <li><strong>Upload a paper</strong> — we generate PyTorch code for you</li>
        <li><strong>Try the Dojo</strong> — coding challenges ranked by difficulty</li>
        <li><strong>Ask the Tutor</strong> — instant explanations of any architecture</li>
      </ul>
      <a href="{FRONTEND_URL}/papers/upload"
         style="display:inline-block;margin-top:24px;padding:12px 28px;background:#7C3AED;
                color:#fff;border-radius:8px;text-decoration:none;font-weight:600;font-size:15px">
        Upload your first paper →
      </a>""")
    return send_email_sync(to_email, f"Welcome to Paper2Code, {name}!", html)


def send_paper_done_email_sync(to_email: str, paper_name: str, paper_id) -> bool:
    link = f"{FRONTEND_URL}/papers/{paper_id}"
    html = _base_template(f"""
      <h2 style="margin:0 0 8px;color:#0F172A;font-size:22px">Your paper is ready</h2>
      <p style="color:#475569;margin:0 0 8px">
        We've finished processing <strong>{paper_name}</strong>.
      </p>
      <p style="color:#475569;margin:0 0 24px">
        PyTorch code has been generated. Open it to explore the architecture.
      </p>
      <a href="{link}" style="display:inline-block;padding:12px 28px;background:#7C3AED;
         color:#fff;border-radius:8px;text-decoration:none;font-weight:600;font-size:15px">
        View Paper →
      </a>""")
    return send_email_sync(to_email, f"✅ {paper_name} is ready", html)


def send_achievement_unlocked_email_sync(
    to_email: str, achievement_name: str, description: str, name: str = "there"
) -> bool:
    html = _base_template(f"""
      <h2 style="margin:0 0 8px;color:#0F172A;font-size:22px">Achievement unlocked 🏆</h2>
      <p style="color:#475569;margin:0 0 24px">Great work, {name}! You just earned:</p>
      <div style="background:#F5F3FF;border:1px solid #DDD6FE;border-radius:8px;
                  padding:20px 24px;margin-bottom:16px">
        <p style="margin:0;font-size:18px;font-weight:700;color:#7C3AED">{achievement_name}</p>
        <p style="margin:8px 0 0;color:#6B7280;font-size:14px">{description}</p>
      </div>
      <a href="{FRONTEND_URL}/profile"
         style="display:inline-block;padding:12px 28px;background:#7C3AED;
                color:#fff;border-radius:8px;text-decoration:none;font-weight:600;font-size:15px">
        See your profile →
      </a>""")
    return send_email_sync(to_email, f"🏆 You earned: {achievement_name}", html)


def send_password_reset_email_sync(to_email: str, token: str) -> bool:
    link = f"{FRONTEND_URL}/reset-password?token={token}"
    html = _base_template(f"""
      <h2 style="margin:0 0 8px;color:#0F172A;font-size:22px">Reset your password</h2>
      <p style="color:#475569;margin:0 0 24px">
        Click below to choose a new password. This link expires in 1 hour.
      </p>
      <a href="{link}" style="display:inline-block;padding:12px 28px;background:#7C3AED;
         color:#fff;border-radius:8px;text-decoration:none;font-weight:600;font-size:15px">
        Reset Password →
      </a>
      <p style="color:#94A3B8;font-size:12px;margin-top:24px">
        Didn't request this? You can safely ignore this email.
      </p>""")
    return send_email_sync(to_email, "Reset your Paper2Code password", html)


def send_drip_email_sync(to_email: str, name: str, day: int):
    return send_email_sync(to_email, f"Day {day} at Paper2Code", f"<p>Hi {name}</p>")

def send_streak_at_risk_email_sync(to_email: str, name: str, streak: int):
    return send_email_sync(to_email, "Keep your streak alive!", f"<p>Hi {name}, you have a {streak} day streak at risk.</p>")

def send_weekly_digest_email_sync(to_email: str, name: str, stats: dict):
    return send_email_sync(to_email, "Your Weekly Digest", f"<p>Hi {name}, here are your stats: {stats}</p>")




_DRIP_TEMPLATES = {
    1: {'subject': 'Day 1', 'cta_text': 'Start', 'body': '...'},
    3: {'subject': 'Day 3', 'cta_text': 'Continue', 'body': '...'},
    7: {'subject': 'Day 7', 'cta_text': 'Finish', 'body': '...'},
}

