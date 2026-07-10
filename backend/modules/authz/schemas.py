from datetime import datetime

from pydantic import BaseModel, ConfigDict


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
    parent_id: int | None = None


class TeamResponse(BaseModel):
    id: int
    organization_id: int
    parent_id: int | None
    name: str
    slug: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ShareRequest(BaseModel):
    shared_with_type: str  # user, team, organization, public
    shared_with_id: str | None = None
    access_level: str  # read, comment, edit, admin, owner


class ApiKeyCreateRequest(BaseModel):
    name: str
    scopes: list[str]
    ttl_days: int | None = None


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    scopes: list[str]
    expires_at: datetime | None
    revoked: bool
    usage_count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
