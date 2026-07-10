import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import backref, relationship

from backend.database import Base


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    slug = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    subscription_tier = Column(
        String(50), default="Free", nullable=False
    )  # Free, Pro, Team, Enterprise

    owner = relationship("User", foreign_keys=[owner_id])


class OrganizationMember(Base):
    __tablename__ = "organization_members"
    __table_args__ = (UniqueConstraint("organization_id", "user_id", name="uq_org_user"),)

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role = Column(String(50), nullable=False)  # Owner, Admin, Member, etc.
    joined_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    organization = relationship(
        "Organization", backref=backref("members", cascade="all, delete-orphan")
    )
    user = relationship("User", backref=backref("org_memberships", cascade="all, delete-orphan"))


class OrganizationInvitation(Base):
    __tablename__ = "organization_invitations"

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    email = Column(String(255), nullable=False, index=True)
    role = Column(String(50), nullable=False)
    token_hash = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    accepted = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    organization = relationship(
        "Organization", backref=backref("invitations", cascade="all, delete-orphan")
    )


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    parent_id = Column(
        Integer, ForeignKey("teams.id", ondelete="SET NULL"), nullable=True, index=True
    )
    name = Column(String(255), nullable=False)
    slug = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    organization = relationship(
        "Organization", backref=backref("teams", cascade="all, delete-orphan")
    )
    parent = relationship("Team", remote_side=[id], backref=backref("children", cascade="all"))


class TeamMember(Base):
    __tablename__ = "team_members"
    __table_args__ = (UniqueConstraint("team_id", "user_id", name="uq_team_user"),)

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(
        Integer, ForeignKey("teams.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role = Column(String(50), default="Member", nullable=False)
    joined_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    team = relationship("Team", backref=backref("members", cascade="all, delete-orphan"))
    user = relationship("User", backref=backref("team_memberships", cascade="all, delete-orphan"))


class TeamInvitation(Base):
    __tablename__ = "team_invitations"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(
        Integer, ForeignKey("teams.id", ondelete="CASCADE"), nullable=False, index=True
    )
    email = Column(String(255), nullable=False, index=True)
    role = Column(String(50), default="Member", nullable=False)
    token_hash = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    accepted = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    team = relationship("Team", backref=backref("invitations", cascade="all, delete-orphan"))


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    organization_id = Column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True
    )
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    scopes = Column(JSON, nullable=False)  # JSON list of permissions, e.g. ["project.read"]
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    revoked = Column(Boolean, default=False, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0, nullable=False)

    user = relationship("User", backref=backref("api_keys", cascade="all, delete-orphan"))
    organization = relationship("Organization")


class ResourceShare(Base):
    __tablename__ = "resource_shares"
    __table_args__ = (
        UniqueConstraint(
            "resource_type",
            "resource_id",
            "shared_with_type",
            "shared_with_id",
            name="uq_resource_share",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    resource_type = Column(String(50), nullable=False, index=True)  # project, paper, model, etc.
    resource_id = Column(String(255), nullable=False, index=True)
    shared_with_type = Column(String(50), nullable=False)  # user, team, organization, public
    shared_with_id = Column(String(255), nullable=True)  # null if shared_with_type is public
    access_level = Column(String(50), nullable=False)  # read, comment, edit, admin, owner
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
