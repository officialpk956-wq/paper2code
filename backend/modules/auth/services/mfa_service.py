import base64
import hashlib
import io
import secrets

import pyotp
import qrcode
from sqlalchemy.orm import Session

from backend.models import User
from backend.modules.auth.services.audit_service import AuditService


def hash_backup_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


class MFAService:
    def __init__(self, db: Session):
        self.db = db
        self.audit_service = AuditService(db)

    def generate_mfa_setup(self, user: User) -> tuple[str, str, list[str]]:
        """
        Generate setup details:
        - TOTP Secret
        - QR Code base64 Data URI
        - Backup codes (plaintext to display to the user, hashes stored in DB)
        """
        secret = pyotp.random_base32()

        # Create provisioning URI
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(name=user.email, issuer_name="Paper2Code")

        # Generate QR Code Image
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        qr_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        qr_data_uri = f"data:image/png;base64,{qr_base64}"

        # Generate backup codes
        plaintext_backups = [secrets.token_hex(5) for _ in range(8)]  # 10-char hex codes
        hashed_backups = [hash_backup_code(c) for c in plaintext_backups]

        # Temporarily save secret and backup codes, but do not enable MFA yet
        # MFA is enabled only after user completes verification check
        user.totp_secret = secret
        user.backup_codes = hashed_backups
        self.db.commit()

        return secret, qr_data_uri, plaintext_backups

    def verify_and_enable(
        self, user: User, code: str, ip_address: str | None, user_agent: str | None
    ) -> bool:
        if not user.totp_secret:
            return False

        totp = pyotp.TOTP(user.totp_secret)
        if totp.verify(code):
            user.mfa_enabled = True
            self.db.commit()

            self.audit_service.log(
                action="mfa_enabled", user_id=user.id, ip_address=ip_address, device=user_agent
            )
            return True
        return False

    def disable_mfa(self, user: User, ip_address: str | None, user_agent: str | None) -> None:
        user.mfa_enabled = False
        user.totp_secret = None
        user.backup_codes = None
        self.db.commit()

        self.audit_service.log(
            action="mfa_disabled", user_id=user.id, ip_address=ip_address, device=user_agent
        )

    def verify_totp(self, user: User, code: str) -> bool:
        if not user.mfa_enabled or not user.totp_secret:
            return False
        totp = pyotp.TOTP(user.totp_secret)
        return totp.verify(code)

    def verify_backup_code(self, user: User, code: str) -> bool:
        """
        Verify if code is a valid backup code.
        If valid, consume it (remove from user list) and return True.
        """
        if not user.mfa_enabled or not user.backup_codes:
            return False

        hashed_code = hash_backup_code(code)
        codes: list[str] = list(user.backup_codes)

        if hashed_code in codes:
            codes.remove(hashed_code)
            user.backup_codes = codes
            self.db.commit()
            return True
        return False
