"""sprint_d_growth_retention

Revision ID: d1e2f3a4b5c6
Revises: c1d2e3f4a5b6
Create Date: 2026-06-27

Sprint D tables:
  - oauth_accounts     : link provider accounts to users
  - achievements       : achievement catalogue
  - user_achievements  : which achievements each user has earned
  - email_drip_log     : prevent duplicate drip sends
"""

from alembic import op
import sqlalchemy as sa

revision = "d1e2f3a4b5c6"
down_revision = "c1d2e3f4a5b6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "oauth_accounts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("provider_user_id", sa.String(255), nullable=False),
        sa.Column("provider_email", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("provider", "provider_user_id", name="uq_oauth_provider_uid"),
    )
    op.create_index("ix_oauth_user", "oauth_accounts", ["user_id"])

    op.create_table(
        "achievements",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("slug", sa.String(100), unique=True, nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("icon", sa.String(50), nullable=True),
        sa.Column("xp_reward", sa.Integer, nullable=False, server_default="0"),
        sa.Column("category", sa.String(50), nullable=False),
    )
    op.create_index("ix_achievements_slug", "achievements", ["slug"])

    op.create_table(
        "user_achievements",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("achievement_id", sa.Integer, sa.ForeignKey("achievements.id", ondelete="CASCADE"), nullable=False),
        sa.Column("earned_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column("payload", sa.JSON, nullable=True),
        sa.UniqueConstraint("user_id", "achievement_id", name="uq_user_achievement"),
    )
    op.create_index("ix_userachievement_user", "user_achievements", ["user_id"])

    op.create_table(
        "email_drip_log",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("drip_day", sa.Integer, nullable=False),
        sa.Column("sent_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("user_id", "drip_day", name="uq_drip_user_day"),
    )
    op.create_index("ix_email_drip_user", "email_drip_log", ["user_id"])


def downgrade() -> None:
    op.drop_table("email_drip_log")
    op.drop_table("user_achievements")
    op.drop_table("achievements")
    op.drop_table("oauth_accounts")
