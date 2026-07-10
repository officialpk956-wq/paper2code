from sqlalchemy.orm import Session

from backend.modules.auth.models import AuditLog


class AuditRepository:
    def __init__(self, db: Session):
        self.db = db

    def log_event(
        self,
        action: str,
        user_id: int | None,
        ip_address: str | None,
        device: str | None,
        metadata_dict: dict | None = None,
    ) -> AuditLog:
        log = AuditLog(
            user_id=user_id,
            action=action,
            ip_address=ip_address,
            device=device,
            metadata_json=metadata_dict,
        )
        self.db.add(log)
        self.db.flush()
        return log
