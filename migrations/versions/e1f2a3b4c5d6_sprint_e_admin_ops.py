"""sprint_e_admin_ops

Revision ID: e1f2a3b4c5d6
Revises: d1e2f3a4b5c6
Create Date: 2026-06-27

Sprint E additions:
  - papers.is_flagged, papers.flag_reason   — content moderation
  - users.weekly_points                      — leaderboard weekly reset
  - leaderboard_archive table               — weekly snapshots before reset
"""

from alembic import op
import sqlalchemy as sa

revision = "e1f2a3b4c5d6"
down_revision = "d1e2f3a4b5c6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("papers") as batch:
        batch.add_column(sa.Column("is_flagged", sa.Boolean, nullable=False, server_default="false"))
        batch.add_column(sa.Column("flag_reason", sa.String(255), nullable=True))

    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("weekly_points", sa.Integer, nullable=False, server_default="0"))

    op.create_table(
        "leaderboard_archive",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("week_start", sa.DateTime, nullable=False),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("weekly_points", sa.Integer, nullable=False, server_default="0"),
        sa.Column("rank", sa.Integer, nullable=True),
        sa.Column("snapshot_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("week_start", "user_id", name="uq_lb_archive_week_user"),
    )
    op.create_index("ix_lb_archive_week", "leaderboard_archive", ["week_start"])


def downgrade() -> None:
    op.drop_table("leaderboard_archive")
    with op.batch_alter_table("users") as batch:
        batch.drop_column("weekly_points")
    with op.batch_alter_table("papers") as batch:
        batch.drop_column("flag_reason")
        batch.drop_column("is_flagged")
