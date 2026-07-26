"""dojo v2: structured test cases — reference_solution + memory_limit_mb on
problems, cases_json/num_passed/num_total on dojo_submissions.

This also MERGES the two open heads (sprint_g_dojo + add_model_graphs).

Revision ID: k2l3m4n5o6p7
Revises: h1i2j3k4l5m6, j1k2l3m4n5o6
Create Date: 2026-07-24
"""

from alembic import op
import sqlalchemy as sa

revision = "k2l3m4n5o6p7"
down_revision = ("h1i2j3k4l5m6", "j1k2l3m4n5o6")
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("problems") as batch:
        batch.add_column(sa.Column("reference_solution", sa.Text(), nullable=True))
        batch.add_column(sa.Column("memory_limit_mb", sa.Integer(), nullable=True))

    with op.batch_alter_table("dojo_submissions") as batch:
        batch.add_column(sa.Column("cases_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("num_passed", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("num_total", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("dojo_submissions") as batch:
        batch.drop_column("num_total")
        batch.drop_column("num_passed")
        batch.drop_column("cases_json")

    with op.batch_alter_table("problems") as batch:
        batch.drop_column("memory_limit_mb")
        batch.drop_column("reference_solution")
