from typing import Optional
from sqlalchemy.orm import Session
from backend.modules.auth.repositories.audit_repository import AuditRepository

class AuditService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = AuditRepository(db)

    def log(
        self,
        action: str,
        user_id: Optional[int],
        ip_address: Optional[str],
        device: Optional[str],
        metadata_dict: Optional[dict] = None,
    ) -> None:
        self.repo.log_event(
            action=action,
            user_id=user_id,
            ip_address=ip_address,
            device=device,
            metadata_dict=metadata_dict
        )
        self.db.commit()
