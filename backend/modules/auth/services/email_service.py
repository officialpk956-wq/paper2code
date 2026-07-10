import logging
import os

logger = logging.getLogger(__name__)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")


def _send(to: str, subject: str, html: str) -> None:
    api_key = os.getenv("RESEND_API_KEY", "")
    if not api_key:
        # Dev fallback — log the link so it can be used manually
        logger.info("RESEND_API_KEY not set — email not sent to %s | %s", to, html[:200])
        return
    try:
        import resend

        resend.api_key = api_key
        resend.Emails.send(
            {
                "from": os.getenv("RESEND_FROM_EMAIL", "noreply@paper2code.ai"),
                "to": [to],
                "subject": subject,
                "html": html,
            }
        )
    except Exception as exc:
        logger.error("Resend error sending to %s: %s", to, exc)


class EmailService:
    @staticmethod
    def send_verification_email(email: str, token: str) -> None:
        link = f"{FRONTEND_URL}/auth/verified?token={token}"
        _send(
            to=email,
            subject="Verify your paper2code email",
            html=f"""
            <p>Thanks for signing up!</p>
            <p><a href="{link}">Click here to verify your email</a></p>
            <p>Or copy this link: {link}</p>
            <p>This link expires in 24 hours.</p>
            """,
        )

    @staticmethod
    def send_reset_password_email(email: str, token: str) -> None:
        link = f"{FRONTEND_URL}/auth/reset-password?token={token}"
        _send(
            to=email,
            subject="Reset your paper2code password",
            html=f"""
            <p>You requested a password reset.</p>
            <p><a href="{link}">Click here to reset your password</a></p>
            <p>Or copy this link: {link}</p>
            <p>This link expires in 1 hour. If you didn't request this, ignore this email.</p>
            """,
        )

    @staticmethod
    def send_account_deleted_email(email: str, name: str) -> None:
        _send(
            to=email,
            subject="Your paper2code account has been deleted",
            html=f"<p>Hello {name},</p><p>Your paper2code account has been deleted. If this was a mistake, contact support.</p>",
        )
