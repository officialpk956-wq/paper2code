from sqlalchemy.orm import Session

from backend.modules.auth.repositories.audit_repository import AuditRepository


class AuditService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = AuditRepository(db)

    def log(
        self,
        action: str,
        user_id: int | None,
        ip_address: str | None,
        device: str | None,
        metadata_dict: dict | None = None,
    ) -> None:
        self.repo.log_event(
            action=action,
            user_id=user_id,
            ip_address=ip_address,
            device=device,
            metadata_dict=metadata_dict,
        )
        self.db.commit()
