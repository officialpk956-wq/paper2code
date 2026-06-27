"""sprint_b_content_storage

Revision ID: b9c3f2e1d4a5
Revises: a9ba9ab5a1d6
Create Date: 2026-06-27 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b9c3f2e1d4a5'
down_revision: Union[str, Sequence[str], None] = 'a9ba9ab5a1d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('papers', schema=None) as batch_op:
        batch_op.add_column(sa.Column('visibility', sa.String(length=20), server_default='public', nullable=False))
        batch_op.add_column(sa.Column('terms_accepted_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('uploaded_by', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('r2_key', sa.String(length=512), nullable=True))
        batch_op.create_foreign_key('fk_paper_uploaded_by', 'users', ['uploaded_by'], ['id'], ondelete='SET NULL')
        batch_op.create_index('ix_papers_uploaded_by', ['uploaded_by'], unique=False)

    with op.batch_alter_table('problems', schema=None) as batch_op:
        batch_op.add_column(sa.Column('is_retired', sa.Boolean(), server_default='false', nullable=False))


def downgrade() -> None:
    with op.batch_alter_table('problems', schema=None) as batch_op:
        batch_op.drop_column('is_retired')

    with op.batch_alter_table('papers', schema=None) as batch_op:
        batch_op.drop_index('ix_papers_uploaded_by')
        batch_op.drop_constraint('fk_paper_uploaded_by', type_='foreignkey')
        batch_op.drop_column('r2_key')
        batch_op.drop_column('uploaded_by')
        batch_op.drop_column('terms_accepted_at')
        batch_op.drop_column('visibility')
