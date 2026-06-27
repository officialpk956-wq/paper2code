"""sprint_f: xp_events, tutor_feedback, tutor_session_records

Revision ID: g1h2i3j4k5l6
Revises: f1g2h3i4j5k6
Create Date: 2026-06-27
"""

from alembic import op
import sqlalchemy as sa

revision = "g1h2i3j4k5l6"
down_revision = "f1g2h3i4j5k6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "xp_events",
        sa.Column("id",        sa.Integer(), primary_key=True),
        sa.Column("user_id",   sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("action",    sa.String(100), nullable=False),
        sa.Column("amount",    sa.Integer(), nullable=False),
        sa.Column("entity_id", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_xp_events_user_id", "xp_events", ["user_id"])
    op.create_index("ix_xp_events_user_created", "xp_events", ["user_id", "created_at"])

    op.create_table(
        "tutor_feedback",
        sa.Column("id",            sa.Integer(), primary_key=True),
        sa.Column("user_id",       sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id",    sa.String(36), nullable=False),
        sa.Column("message_index", sa.Integer(), nullable=False),
        sa.Column("rating",        sa.Integer(), nullable=False),
        sa.Column("created_at",    sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("session_id", "message_index", "user_id", name="uq_tutor_feedback"),
    )
    op.create_index("ix_tutor_feedback_user", "tutor_feedback", ["user_id"])
    op.create_index("ix_tutor_feedback_session", "tutor_feedback", ["session_id"])

    op.create_table(
        "tutor_session_records",
        sa.Column("id",             sa.Integer(), primary_key=True),
        sa.Column("user_id",        sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id",     sa.String(36), nullable=False, unique=True),
        sa.Column("context_type",   sa.String(100), nullable=True),
        sa.Column("messages",       sa.JSON(), nullable=True),
        sa.Column("created_at",     sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("last_active_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_tutor_session_record_user", "tutor_session_records", ["user_id"])
    op.create_index("ix_tutor_session_record_session", "tutor_session_records", ["session_id"])
    op.create_index("ix_tutor_session_record_user_created", "tutor_session_records", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_table("tutor_session_records")
    op.drop_table("tutor_feedback")
    op.drop_table("xp_events")
