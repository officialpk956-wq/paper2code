import datetime
import secrets

from sqlalchemy.orm import Session

from backend.modules.auth.services.audit_service import AuditService
from backend.modules.authz.models import ApiKey
from backend.modules.authz.repositories.apikey_repository import ApiKeyRepository


class ApiKeyService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = ApiKeyRepository(db)
        self.audit = AuditService(db)

    def create_api_key(
        self,
        user_id: int,
        org_id: int | None,
        name: str,
        scopes: list[str],
        ttl_days: int | None = None,
        ip_address: str | None = None,
        device: str | None = None,
    ) -> tuple[str, ApiKey]:
        # Prefix the raw key for easier identification (e.g. p2c_...)
        raw_key = f"p2c_{secrets.token_urlsafe(32)}"
        expires_at = None
        if ttl_days:
            expires_at = datetime.datetime.utcnow() + datetime.timedelta(days=ttl_days)

        api_key = self.repo.create_key(
            user_id=user_id,
            organization_id=org_id,
            name=name,
            scopes=scopes,
            key_plaintext=raw_key,
            expires_at=expires_at,
        )
        self.db.commit()

        self.audit.log(
            "api_key_created",
            user_id=user_id,
            ip_address=ip_address,
            device=device,
            metadata_dict={"key_id": api_key.id, "name": name, "scopes": scopes},
        )
        return raw_key, api_key

    def revoke_api_key(
        self, key_id: str, user_id: int, ip_address: str | None = None, device: str | None = None
    ) -> bool:
        res = self.repo.revoke_key(key_id, user_id)
        if res:
            self.db.commit()
            self.audit.log(
                "api_key_revoked",
                user_id=user_id,
                ip_address=ip_address,
                device=device,
                metadata_dict={"key_id": key_id},
            )
        return res
