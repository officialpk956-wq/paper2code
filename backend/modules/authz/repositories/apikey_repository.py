import uuid
import datetime
import hashlib
from typing import Optional, List
from sqlalchemy.orm import Session
from backend.modules.authz.models import ApiKey

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

class ApiKeyRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_key(
        self,
        user_id: int,
        organization_id: Optional[int],
        name: str,
        scopes: List[str],
        key_plaintext: str,
        expires_at: Optional[datetime.datetime] = None
    ) -> ApiKey:
        key_hash = hash_key(key_plaintext)
        api_key = ApiKey(
            id=str(uuid.uuid4()),
            user_id=user_id,
            organization_id=organization_id,
            key_hash=key_hash,
            name=name,
            scopes=scopes,
            expires_at=expires_at,
            revoked=False
        )
        self.db.add(api_key)
        self.db.flush()
        return api_key

    def get_key_by_hash(self, key_hash: str) -> Optional[ApiKey]:
        return self.db.query(ApiKey).filter_by(key_hash=key_hash, revoked=False).first()

    def revoke_key(self, key_id: str, user_id: int) -> bool:
        key = self.db.query(ApiKey).filter_by(id=key_id, user_id=user_id).first()
        if key:
            key.revoked = True
            self.db.flush()
            return True
        return False

    def get_keys_for_user(self, user_id: int) -> List[ApiKey]:
        return self.db.query(ApiKey).filter_by(user_id=user_id).all()
