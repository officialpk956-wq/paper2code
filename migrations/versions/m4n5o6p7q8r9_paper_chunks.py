"""Persist page-level paper chunks for evidence-grounded extraction.

Revision ID: m4n5o6p7q8r9
Revises: l3m4n5o6p7q8
Create Date: 2026-08-30
"""

from alembic import op
import sqlalchemy as sa


revision = "m4n5o6p7q8r9"
down_revision = "l3m4n5o6p7q8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "paper_chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("paper_id", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(length=32), nullable=False, server_default="other"),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("chunk_type", sa.String(length=16), nullable=False, server_default="text"),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("source_offset_start", sa.Integer(), nullable=True),
        sa.Column("source_offset_end", sa.Integer(), nullable=True),
        sa.Column("embedding_id", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_paper_chunks_paper_id", "paper_chunks", ["paper_id"])
    op.create_index("ix_paper_chunks_paper_section", "paper_chunks", ["paper_id", "section"])
    op.create_index("ix_paper_chunks_paper_page", "paper_chunks", ["paper_id", "page"])


def downgrade() -> None:
    op.drop_index("ix_paper_chunks_paper_page", table_name="paper_chunks")
    op.drop_index("ix_paper_chunks_paper_section", table_name="paper_chunks")
    op.drop_index("ix_paper_chunks_paper_id", table_name="paper_chunks")
    op.drop_table("paper_chunks")
