"""storage_quota

Revision ID: f1g2h3i4j5k6
Revises: e1f2a3b4c5d6
Create Date: 2026-06-27

Storage quota additions:
  - users.storage_bytes_used   — running total of bytes stored by this user
  - papers.file_size_bytes     — byte size of the uploaded PDF (for quota accounting on delete)
"""

from alembic import op
import sqlalchemy as sa

revision = "f1g2h3i4j5k6"
down_revision = "e1f2a3b4c5d6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("storage_bytes_used", sa.Integer, nullable=False, server_default="0"))

    with op.batch_alter_table("papers") as batch:
        batch.add_column(sa.Column("file_size_bytes", sa.Integer, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("papers") as batch:
        batch.drop_column("file_size_bytes")
    with op.batch_alter_table("users") as batch:
        batch.drop_column("storage_bytes_used")
