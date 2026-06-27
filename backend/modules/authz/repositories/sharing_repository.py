from typing import Optional, Sequence
from sqlalchemy.orm import Session
from backend.modules.authz.models import ResourceShare

class SharingRepository:
    def __init__(self, db: Session):
        self.db = db

    def share_resource(
        self,
        resource_type: str,
        resource_id: str,
        shared_with_type: str,
        shared_with_id: Optional[str],
        access_level: str
    ) -> ResourceShare:
        share = db_share = self.db.query(ResourceShare).filter_by(
            resource_type=resource_type,
            resource_id=resource_id,
            shared_with_type=shared_with_type,
            shared_with_id=shared_with_id
        ).first()

        if db_share:
            db_share.access_level = access_level
        else:
            db_share = ResourceShare(
                resource_type=resource_type,
                resource_id=resource_id,
                shared_with_type=shared_with_type,
                shared_with_id=shared_with_id,
                access_level=access_level
            )
            self.db.add(db_share)
        
        self.db.flush()
        return db_share

    def revoke_share(
        self,
        resource_type: str,
        resource_id: str,
        shared_with_type: str,
        shared_with_id: Optional[str]
    ) -> bool:
        share = self.db.query(ResourceShare).filter_by(
            resource_type=resource_type,
            resource_id=resource_id,
            shared_with_type=shared_with_type,
            shared_with_id=shared_with_id
        ).first()
        if share:
            self.db.delete(share)
            self.db.flush()
            return True
        return False

    def get_shares_for_resource(self, resource_type: str, resource_id: str) -> Sequence[ResourceShare]:
        return self.db.query(ResourceShare).filter_by(resource_type=resource_type, resource_id=str(resource_id)).all()
