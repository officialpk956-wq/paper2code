"""
backend/models.py

SQLAlchemy ORM models for TensorTonic.

Upgraded from the original stub to:
  - Use sqlalchemy.orm.DeclarativeBase (SQLAlchemy 2.x style)
  - Add server_default timestamps (created_at, updated_at)
  - Add ORM relationships with backrefs
  - Keep Difficulty enum and all original column semantics intact
"""

import enum
import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


# ---------------------------------------------------------------------------
# Declarative Base — SQLAlchemy 2.x style
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """
    Project-wide SQLAlchemy declarative base.

    All ORM models inherit from this class.  The Base carries the
    metadata registry used by Alembic and Base.metadata.create_all().
    """
    __allow_unmapped__ = True
    pass


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

    id         = Column(Integer, primary_key=True, index=True)
    email      = Column(String(255), unique=True, nullable=False, index=True)
    name       = Column(String(255), nullable=False)
    avatar_url = Column(String(512))
    points     = Column(Integer, default=0, nullable=False)
    streak     = Column(Integer, default=0, nullable=False)
    last_active = Column(DateTime, nullable=True)
    created_at  = Column(DateTime, server_default=func.now(), nullable=False)

    # We are dropping submissions for now in V1 Pivot.
    # We can add user_progress back later.

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r} points={self.points}>"


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

    # Relationships
    modules: list["PaperModule"] = relationship(
        "PaperModule", back_populates="paper", cascade="all, delete-orphan", order_by="PaperModule.order_index"
    )

    def __repr__(self) -> str:
        return f"<Paper id={self.id} title={self.title!r}>"


# ---------------------------------------------------------------------------
# PaperModule
# ---------------------------------------------------------------------------

class PaperModule(Base):
    __tablename__ = "paper_modules"

    id             = Column(Integer, primary_key=True, index=True)
    paper_id       = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    layer_name     = Column(String(255), nullable=False)
    module_type    = Column(String(128), nullable=False)
    explanation    = Column(Text, nullable=True)
    tensor_flow    = Column(JSON, nullable=True)
    graph_nodes    = Column(JSON, nullable=True)
    flops_context  = Column(JSON, nullable=True)
    order_index    = Column(Integer, default=0, nullable=False)

    # Relationships
    paper: "Paper" = relationship("Paper", back_populates="modules")

    def __repr__(self) -> str:
        return f"<PaperModule id={self.id} paper_id={self.paper_id} layer_name={self.layer_name!r}>"

# ---------------------------------------------------------------------------
# Phase 8: Active Learning & Analytics Models
# ---------------------------------------------------------------------------

class LearnerProgress(Base):
    __tablename__ = "learner_progress"

    id = Column(Integer, primary_key=True, index=True)
    learner_id = Column(String(255), index=True, nullable=False)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    module_id = Column(Integer, ForeignKey("paper_modules.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(50), default="not_started", nullable=False) # not_started, in_progress, completed
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    time_spent_seconds = Column(Integer, default=0, nullable=False)

class AssessmentAttempt(Base):
    __tablename__ = "assessment_attempts"

    id = Column(Integer, primary_key=True, index=True)
    learner_id = Column(String(255), index=True, nullable=False)
    assessment_type = Column(String(100), nullable=False)   # architecture, tensor, flops, comparison
    architecture = Column(String(255), nullable=True)        # E.g., ResNet
    difficulty = Column(String(50), nullable=True)           # beginner, intermediate, advanced
    question_text = Column(Text, nullable=True)              # Full question shown to learner
    user_answer = Column(Text, nullable=True)                # Raw answer submitted
    correct_answer = Column(Text, nullable=True)             # Ground-truth answer
    explanation = Column(Text, nullable=True)                # Why the answer is correct
    score = Column(Integer, nullable=False, default=0)       # 0 or 1
    attempt_count = Column(Integer, default=1, nullable=False)
    is_correct = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

class TutorAnalytics(Base):
    __tablename__ = "tutor_analytics"

    id = Column(Integer, primary_key=True, index=True)
    learner_id = Column(String(255), index=True, nullable=True)
    architecture = Column(String(255), nullable=True)
    module = Column(String(255), nullable=True)
    reasoning_type = Column(String(100), nullable=True)
    question_count = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
