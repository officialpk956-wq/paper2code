"""Persist generated paper-to-code artifacts.

Revision ID: l3m4n5o6p7q8
Revises: k2l3m4n5o6p7
Create Date: 2026-08-28
"""

from alembic import op
import sqlalchemy as sa


revision = "l3m4n5o6p7q8"
down_revision = "k2l3m4n5o6p7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("papers") as batch:
        batch.add_column(sa.Column("generated_code_source", sa.Text(), nullable=True))
        batch.add_column(sa.Column("generated_code_compiled", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("generation_status", sa.String(length=20), nullable=True))
        batch.add_column(sa.Column("verification_report", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("last_generation_error", sa.Text(), nullable=True))
        batch.create_check_constraint(
            "ck_papers_generation_status",
            "generation_status IS NULL OR generation_status IN "
            "('pending', 'success', 'failed', 'needs_review')",
        )


def downgrade() -> None:
    with op.batch_alter_table("papers") as batch:
        batch.drop_constraint("ck_papers_generation_status", type_="check")
        batch.drop_column("last_generation_error")
        batch.drop_column("verification_report")
        batch.drop_column("generation_status")
        batch.drop_column("generated_code_compiled")
        batch.drop_column("generated_code_source")
