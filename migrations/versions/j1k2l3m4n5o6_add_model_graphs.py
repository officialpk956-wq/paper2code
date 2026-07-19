"""add model_graphs table for saved architecture visualizations

Revision ID: j1k2l3m4n5o6
Revises: a1ad010652f1
Create Date: 2026-07-19
"""

from alembic import op
import sqlalchemy as sa

revision = "j1k2l3m4n5o6"
down_revision = "a1ad010652f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_graphs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("format", sa.String(20), nullable=False),
        sa.Column("graph_data", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_graphs_id", "model_graphs", ["id"])
    op.create_index("ix_model_graphs_user_id", "model_graphs", ["user_id"])
    op.create_index("ix_model_graphs_user_created", "model_graphs", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_model_graphs_user_created", table_name="model_graphs")
    op.drop_index("ix_model_graphs_user_id", table_name="model_graphs")
    op.drop_index("ix_model_graphs_id", table_name="model_graphs")
    op.drop_table("model_graphs")
