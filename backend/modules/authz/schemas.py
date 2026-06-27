from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import datetime

class OrgCreate(BaseModel):
    name: str
    slug: str

class OrgResponse(BaseModel):
    id: int
    name: str
    slug: str
    owner_id: int
    subscription_tier: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class OrgInviteRequest(BaseModel):
    email: str
    role: str

class OrgInviteAccept(BaseModel):
    token: str

class TeamCreate(BaseModel):
    name: str
    slug: str
    parent_id: Optional[int] = None

class TeamResponse(BaseModel):
    id: int
    organization_id: int
    parent_id: Optional[int]
    name: str
    slug: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ShareRequest(BaseModel):
    shared_with_type: str  # user, team, organization, public
    shared_with_id: Optional[str] = None
    access_level: str  # read, comment, edit, admin, owner

class ApiKeyCreateRequest(BaseModel):
    name: str
    scopes: List[str]
    ttl_days: Optional[int] = None

class ApiKeyResponse(BaseModel):
    id: str
    name: str
    scopes: List[str]
    expires_at: Optional[datetime]
    revoked: bool
    usage_count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
