"""sprint_c_discovery_ops

Revision ID: c1d2e3f4a5b6
Revises: b9c3f2e1d4a5
Create Date: 2026-06-27 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c1d2e3f4a5b6'
down_revision: Union[str, Sequence[str], None] = 'b9c3f2e1d4a5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'notifications',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('type', sa.String(length=50), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('body', sa.Text(), nullable=True),
        sa.Column('is_read', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('notifications', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_notifications_id'), ['id'], unique=False)
        batch_op.create_index(batch_op.f('ix_notifications_user_id'), ['user_id'], unique=False)
        batch_op.create_index('ix_notification_user_created', ['user_id', 'created_at'], unique=False)

    # PostgreSQL-only: GIN indexes for full-text search.
    # These are created at runtime by the application on first use if missing.
    # Run manually on PostgreSQL:
    #   CREATE INDEX ix_papers_fts   ON papers   USING GIN(to_tsvector('english', coalesce(title,'') || ' ' || coalesce(abstract,'')));
    #   CREATE INDEX ix_problems_fts ON problems USING GIN(to_tsvector('english', coalesce(title,'') || ' ' || coalesce(description,'')));


def downgrade() -> None:
    with op.batch_alter_table('notifications', schema=None) as batch_op:
        batch_op.drop_index('ix_notification_user_created')
        batch_op.drop_index(batch_op.f('ix_notifications_user_id'))
        batch_op.drop_index(batch_op.f('ix_notifications_id'))
    op.drop_table('notifications')
