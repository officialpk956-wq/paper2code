"""sprint_g: problem versioning + timeout, submission is_best + version

Revision ID: h1i2j3k4l5m6
Revises: g1h2i3j4k5l6
Create Date: 2026-06-27
"""

from alembic import op
import sqlalchemy as sa

revision = "h1i2j3k4l5m6"
down_revision = "g1h2i3j4k5l6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("problems") as batch:
        batch.add_column(sa.Column("version",         sa.Integer(),      nullable=False, server_default="1"))
        batch.add_column(sa.Column("time_limit_ms",   sa.Integer(),      nullable=True))
        batch.add_column(sa.Column("acceptance_rate", sa.Numeric(5, 4),  nullable=True))

    with op.batch_alter_table("dojo_submissions") as batch:
        batch.add_column(sa.Column("is_best",         sa.Boolean(),  nullable=False, server_default="false"))
        batch.add_column(sa.Column("problem_version", sa.Integer(),  nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("dojo_submissions") as batch:
        batch.drop_column("problem_version")
        batch.drop_column("is_best")

    with op.batch_alter_table("problems") as batch:
        batch.drop_column("acceptance_rate")
        batch.drop_column("time_limit_ms")
        batch.drop_column("version")
