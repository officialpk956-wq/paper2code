from typing import Optional
from sqlalchemy.orm import Session
from backend.modules.auth.models import AuditLog

class AuditRepository:
    def __init__(self, db: Session):
        self.db = db

    def log_event(
        self,
        action: str,
        user_id: Optional[int],
        ip_address: Optional[str],
        device: Optional[str],
        metadata_dict: Optional[dict] = None,
    ) -> AuditLog:
        log = AuditLog(
            user_id=user_id,
            action=action,
            ip_address=ip_address,
            device=device,
            metadata_json=metadata_dict
        )
        self.db.add(log)
        self.db.flush()
        return log
