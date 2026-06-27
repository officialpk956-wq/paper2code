"""
backend/models.py

SQLAlchemy ORM models for Paper2Code.

All ORM models inherit from Base (SQLAlchemy 2.x DeclarativeBase).
Use Alembic for schema migrations. Base.metadata.create_all() is for dev/test only.
"""

import enum
import uuid
import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
    Numeric,
)
from sqlalchemy.orm import relationship
from backend.database import Base


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Difficulty(enum.Enum):
    Easy   = "Easy"
    Medium = "Medium"
    Hard   = "Hard"


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    email           = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=True)
    name            = Column(String(255), nullable=False)
    avatar_url      = Column(String(512))
    points          = Column(Integer, default=0, nullable=False)
    streak          = Column(Integer, default=0, nullable=False)
    last_active     = Column(DateTime, nullable=True)
    created_at      = Column(DateTime, server_default=func.now(), nullable=False)
    
    is_admin           = Column(Boolean, nullable=False, default=False, server_default='false')
    is_email_verified  = Column(Boolean, nullable=False, default=False, server_default='false')
    email_verified_at  = Column(DateTime(timezone=True), nullable=True)

    # Security and MFA fields
    is_verified           = Column(Boolean, default=False, nullable=False)
    token_version         = Column(Integer, default=1, nullable=False)
    totp_secret           = Column(String(255), nullable=True)
    mfa_enabled           = Column(Boolean, default=False, nullable=False)
    backup_codes          = Column(JSON, nullable=True)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    lockout_until         = Column(DateTime, nullable=True)

    submissions: list["DojoSubmission"] = relationship(
        "DojoSubmission", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r} points={self.points}>"

    @property
    def xp_level(self) -> int:
        return (self.points or 0) // 100 + 1


# ---------------------------------------------------------------------------
# Paper
# ---------------------------------------------------------------------------

class Paper(Base):
    __tablename__ = "papers"

    id                 = Column(Integer, primary_key=True, index=True)
    title              = Column(String(512), nullable=False, unique=True)
    authors            = Column(String(1024), nullable=True)
    abstract           = Column(Text, nullable=True)
    architecture_graph = Column(JSON, nullable=True)
    flops_analysis     = Column(JSON, nullable=True)
    created_at         = Column(DateTime, server_default=func.now(), nullable=False)

    # Sprint B: visibility, ToS acceptance, ownership, R2 storage
    visibility        = Column(String(20), nullable=False, default='public', server_default='public')
    terms_accepted_at = Column(DateTime, nullable=True)
    uploaded_by       = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    r2_key            = Column(String(512), nullable=True)

    modules: list["PaperModule"] = relationship(
        "PaperModule",
        back_populates="paper",
        cascade="all, delete-orphan",
        order_by="PaperModule.order_index",
    )

    def __repr__(self) -> str:
        return f"<Paper id={self.id} title={self.title!r}>"


# ---------------------------------------------------------------------------
# PaperModule
# ---------------------------------------------------------------------------

class PaperModule(Base):
    __tablename__ = "paper_modules"

    id            = Column(Integer, primary_key=True, index=True)
    paper_id      = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    layer_name    = Column(String(255), nullable=False)
    module_type   = Column(String(128), nullable=False)
    explanation   = Column(Text, nullable=True)
    tensor_flow   = Column(JSON, nullable=True)
    graph_nodes   = Column(JSON, nullable=True)
    flops_context = Column(JSON, nullable=True)
    order_index   = Column(Integer, default=0, nullable=False)

    paper: "Paper" = relationship("Paper", back_populates="modules")

    def __repr__(self) -> str:
        return f"<PaperModule id={self.id} paper_id={self.paper_id} layer_name={self.layer_name!r}>"


# ---------------------------------------------------------------------------
# LearnerProgress — tracks completion of any entity (topic, problem, paper)
# ---------------------------------------------------------------------------

class LearnerProgress(Base):
    __tablename__ = "learner_progress"
    __table_args__ = (
        UniqueConstraint("learner_id", "entity_type", "entity_id", name="uq_progress"),
        Index("ix_progress_learner_entity", "learner_id", "entity_type", "entity_id"),
    )

    id                 = Column(Integer, primary_key=True, index=True)
    learner_id         = Column(String(255), index=True, nullable=False)
    entity_type        = Column(String(50), nullable=False, server_default="paper_module")
    entity_id          = Column(String(255), nullable=False)
    status             = Column(String(50), default="not_started", nullable=False)
    started_at         = Column(DateTime, nullable=True)
    completed_at       = Column(DateTime, nullable=True)
    time_spent_seconds = Column(Integer, default=0, nullable=False)


# ---------------------------------------------------------------------------
# AssessmentAttempt
# ---------------------------------------------------------------------------

class AssessmentAttempt(Base):
    __tablename__ = "assessment_attempts"

    id              = Column(Integer, primary_key=True, index=True)
    learner_id      = Column(String(255), index=True, nullable=False)
    assessment_type = Column(String(100), nullable=False)
    architecture    = Column(String(255), nullable=True)
    difficulty      = Column(String(50), nullable=True)
    question_text   = Column(Text, nullable=True)
    user_answer     = Column(Text, nullable=True)
    correct_answer  = Column(Text, nullable=True)
    explanation     = Column(Text, nullable=True)
    score           = Column(Integer, nullable=False, default=0)
    attempt_count   = Column(Integer, default=1, nullable=False)
    is_correct      = Column(Boolean, default=False, nullable=False)
    created_at      = Column(DateTime, server_default=func.now(), nullable=False)


# ---------------------------------------------------------------------------
# TutorAnalytics
# ---------------------------------------------------------------------------

class TutorAnalytics(Base):
    __tablename__ = "tutor_analytics"

    id             = Column(Integer, primary_key=True, index=True)
    learner_id     = Column(String(255), index=True, nullable=True)
    architecture   = Column(String(255), nullable=True)
    module         = Column(String(255), nullable=True)
    reasoning_type = Column(String(100), nullable=True)
    question_count = Column(Integer, default=1, nullable=False)
    created_at     = Column(DateTime, server_default=func.now(), nullable=False)


# ---------------------------------------------------------------------------
# Problem (Dojo challenges)
# ---------------------------------------------------------------------------

class Problem(Base):
    __tablename__ = "problems"

    id                    = Column(String(50), primary_key=True, index=True)
    slug                  = Column(String(100), unique=True, index=True)
    title                 = Column(String(255))
    category              = Column(String(50))
    difficulty            = Column(String(50))
    estimated_time        = Column(Integer)
    description           = Column(Text)
    tags                  = Column(JSON)
    related_architectures = Column(JSON)
    related_papers        = Column(JSON)
    related_math          = Column(JSON)
    learning_points       = Column(JSON)
    visualization_url     = Column(String(512), nullable=True)
    python_template       = Column(Text)
    test_cases            = Column(JSON)
    hints                 = Column(JSON)
    explanation           = Column(JSON)
    # Sprint B: soft-delete for admin-managed problems
    is_retired            = Column(Boolean, nullable=False, default=False, server_default='false')

    submissions: list["DojoSubmission"] = relationship(
        "DojoSubmission", back_populates="problem", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# DojoSubmission — stores each user code submission + Piston result
# ---------------------------------------------------------------------------

class DojoSubmission(Base):
    __tablename__ = "dojo_submissions"
    __table_args__ = (
        Index("ix_submission_user_problem", "user_id", "problem_id"),
        Index("ix_submission_created", "created_at"),
    )

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    problem_id = Column(String(50), ForeignKey("problems.id", ondelete="CASCADE"), nullable=False, index=True)
    code       = Column(Text, nullable=False)
    passed     = Column(Boolean, nullable=False, default=False)
    stdout     = Column(Text, nullable=True)
    stderr     = Column(Text, nullable=True)
    time_ms    = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    user:    "User"    = relationship("User", back_populates="submissions")
    problem: "Problem" = relationship("Problem", back_populates="submissions")


# ---------------------------------------------------------------------------
# Task — async job tracking (paper code generation, etc.)
# ---------------------------------------------------------------------------

class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        Index("ix_task_user_status", "user_id", "status"),
    )

    id         = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    type       = Column(String(50), nullable=False)
    status     = Column(String(20), default="pending", nullable=False)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    input_ref  = Column(String(255), nullable=True)
    result     = Column(JSON, nullable=True)
    error      = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, onupdate=func.now(), nullable=True)


# ---------------------------------------------------------------------------
# InterviewQuestion
# ---------------------------------------------------------------------------

class InterviewQuestion(Base):
    __tablename__ = "interview_questions"

    id              = Column(String(50), primary_key=True, index=True)
    question        = Column(Text)
    difficulty      = Column(String(50))
    category        = Column(String(100))
    companies       = Column(JSON)
    tags            = Column(JSON)
    key_points      = Column(JSON)
    expected_answer = Column(Text)
    hints           = Column(JSON)


# ---------------------------------------------------------------------------
# Roadmap
# ---------------------------------------------------------------------------

class Roadmap(Base):
    __tablename__ = "roadmaps"

    id          = Column(String(50), primary_key=True, index=True)
    title       = Column(String(255))
    description = Column(Text)
    nodes       = Column(JSON)


# ---------------------------------------------------------------------------
# Email Verification, Password Reset, and Usage Logs
# ---------------------------------------------------------------------------

class EmailVerificationToken(Base):
    __tablename__ = "email_verification_tokens"
    id         = Column(Integer, primary_key=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token      = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used_at    = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"
    id         = Column(Integer, primary_key=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token      = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used_at    = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class UsageLog(Base):
    __tablename__ = "usage_log"
    id                 = Column(Integer, primary_key=True)
    user_id            = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action             = Column(String(100), nullable=False)
    prompt_tokens      = Column(Integer, nullable=False, default=0)
    completion_tokens  = Column(Integer, nullable=False, default=0)
    cost_usd           = Column(Numeric(10, 6), nullable=False, default=0)
    model              = Column(String(100), nullable=True)
    created_at         = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('ix_usage_log_user_created', 'user_id', 'created_at'),
        Index('ix_usage_log_action', 'action'),
    )


# ---------------------------------------------------------------------------
# Notification — in-app alerts (paper ready, achievements, etc.)
# ---------------------------------------------------------------------------

class Notification(Base):
    __tablename__ = "notifications"
    __table_args__ = (
        Index("ix_notification_user_created", "user_id", "created_at"),
    )

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    type       = Column(String(50), nullable=False)   # e.g. "paper.done", "achievement.unlocked"
    title      = Column(String(255), nullable=False)
    body       = Column(Text, nullable=True)
    is_read    = Column(Boolean, nullable=False, default=False, server_default='false')
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    payload    = Column(JSON, nullable=True)          # e.g. {"paper_id": 7}


# ---------------------------------------------------------------------------
# Sprint D: OAuth accounts
# ---------------------------------------------------------------------------

class OAuthAccount(Base):
    __tablename__ = "oauth_accounts"
    __table_args__ = (
        UniqueConstraint("provider", "provider_user_id", name="uq_oauth_provider_uid"),
        Index("ix_oauth_user", "user_id"),
    )

    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    provider         = Column(String(50), nullable=False)        # "google" | "github"
    provider_user_id = Column(String(255), nullable=False)
    provider_email   = Column(String(255), nullable=True)
    created_at       = Column(DateTime, server_default=func.now(), nullable=False)


# ---------------------------------------------------------------------------
# Sprint D: Achievement system
# ---------------------------------------------------------------------------

class Achievement(Base):
    __tablename__ = "achievements"

    id          = Column(Integer, primary_key=True, index=True)
    slug        = Column(String(100), unique=True, nullable=False, index=True)
    title       = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    icon        = Column(String(50), nullable=True)       # emoji or icon key
    xp_reward   = Column(Integer, nullable=False, default=0)
    category    = Column(String(50), nullable=False)      # "Learning" | "Consistency" | "Community"


class UserAchievement(Base):
    __tablename__ = "user_achievements"
    __table_args__ = (
        UniqueConstraint("user_id", "achievement_id", name="uq_user_achievement"),
        Index("ix_userachievement_user", "user_id"),
    )

    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    achievement_id = Column(Integer, ForeignKey("achievements.id", ondelete="CASCADE"), nullable=False)
    earned_at      = Column(DateTime, server_default=func.now(), nullable=False)
    payload        = Column(JSON, nullable=True)   # extra context {"problem_id": "..."} etc.


# ---------------------------------------------------------------------------
# Sprint D: Email drip tracking
# ---------------------------------------------------------------------------

class EmailDripLog(Base):
    __tablename__ = "email_drip_log"
    __table_args__ = (
        UniqueConstraint("user_id", "drip_day", name="uq_drip_user_day"),
    )

    id       = Column(Integer, primary_key=True, index=True)
    user_id  = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    drip_day = Column(Integer, nullable=False)    # 1, 3, or 7
    sent_at  = Column(DateTime, server_default=func.now(), nullable=False)


# Register modular authentication models
from backend.modules.auth.models import UserSession, VerificationToken, ResetToken, AuditLog

