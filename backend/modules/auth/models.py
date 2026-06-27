import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship, backref
from backend.database import Base

class UserSession(Base):
    __tablename__ = "user_sessions"

    id                 = Column(String(36), primary_key=True, index=True)
    user_id            = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    refresh_token_hash = Column(String(255), nullable=False, index=True)
    ip_address         = Column(String(45), nullable=True)
    user_agent         = Column(String(512), nullable=True)
    browser            = Column(String(100), nullable=True)
    os                 = Column(String(100), nullable=True)
    created_at         = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    last_used_at       = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    expires_at         = Column(DateTime, nullable=False)
    revoked            = Column(Boolean, default=False, nullable=False)
    rotated_at         = Column(DateTime, nullable=True)

    user = relationship("User", backref=backref("auth_sessions", cascade="all, delete-orphan"))


class VerificationToken(Base):
    __tablename__ = "verification_tokens"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash = Column(String(255), nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    used       = Column(Boolean, default=False, nullable=False)

    user = relationship("User", backref=backref("verification_tokens", cascade="all, delete-orphan"))


class ResetToken(Base):
    __tablename__ = "reset_tokens"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash = Column(String(255), nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    used       = Column(Boolean, default=False, nullable=False)

    user = relationship("User", backref=backref("reset_tokens", cascade="all, delete-orphan"))


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    action        = Column(String(100), nullable=False)
    ip_address    = Column(String(45), nullable=True)
    device        = Column(String(255), nullable=True)
    timestamp     = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    metadata_json = Column(JSON, nullable=True)

    user = relationship("User", backref="audit_logs")
