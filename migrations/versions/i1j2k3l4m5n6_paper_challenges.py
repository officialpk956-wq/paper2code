"""paper_challenges

Revision ID: i1j2k3l4m5n6
Revises: h1i2j3k4l5m6
Create Date: 2026-06-28 23:39:35.686557

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'i1j2k3l4m5n6'
down_revision: Union[str, Sequence[str], None] = 'h1i2j3k4l5m6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # paper_challenges
    op.create_table(
        'paper_challenges',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('paper_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('order_idx', sa.Integer(), nullable=False, default=0),
        sa.Column('is_published', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_paper_challenges_id'), 'paper_challenges', ['id'], unique=False)
    op.create_index(op.f('ix_paper_challenges_paper_id'), 'paper_challenges', ['paper_id'], unique=False)

    # paper_challenge_parts
    op.create_table(
        'paper_challenge_parts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('challenge_id', sa.Integer(), nullable=False),
        sa.Column('order_idx', sa.Integer(), nullable=False, default=0),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description_md', sa.Text(), nullable=False),
        sa.Column('paper_section_md', sa.Text(), nullable=True),
        sa.Column('setup_code', sa.Text(), nullable=True),
        sa.Column('starter_code', sa.Text(), nullable=False),
        sa.Column('solution_code', sa.Text(), nullable=True),
        sa.Column('test_code', sa.Text(), nullable=False),
        sa.Column('unlock_requires_part_id', sa.Integer(), nullable=True),
        sa.Column('xp_reward', sa.Integer(), nullable=False, default=50),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['challenge_id'], ['paper_challenges.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['unlock_requires_part_id'], ['paper_challenge_parts.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_paper_challenge_parts_id'), 'paper_challenge_parts', ['id'], unique=False)
    op.create_index(op.f('ix_paper_challenge_parts_challenge_id'), 'paper_challenge_parts', ['challenge_id'], unique=False)

    # paper_part_submissions
    op.create_table(
        'paper_part_submissions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('part_id', sa.Integer(), nullable=False),
        sa.Column('code', sa.Text(), nullable=False),
        sa.Column('passed', sa.Boolean(), nullable=False, default=False),
        sa.Column('stdout', sa.Text(), nullable=True),
        sa.Column('stderr', sa.Text(), nullable=True),
        sa.Column('time_ms', sa.Integer(), nullable=True),
        sa.Column('is_best', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['part_id'], ['paper_challenge_parts.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_paper_part_submissions_id'), 'paper_part_submissions', ['id'], unique=False)
    op.create_index(op.f('ix_paper_part_submissions_part_id'), 'paper_part_submissions', ['part_id'], unique=False)
    op.create_index(op.f('ix_paper_part_submissions_user_id'), 'paper_part_submissions', ['user_id'], unique=False)
    op.create_index('ix_pps_user_part', 'paper_part_submissions', ['user_id', 'part_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_pps_user_part', table_name='paper_part_submissions')
    op.drop_index(op.f('ix_paper_part_submissions_user_id'), table_name='paper_part_submissions')
    op.drop_index(op.f('ix_paper_part_submissions_part_id'), table_name='paper_part_submissions')
    op.drop_index(op.f('ix_paper_part_submissions_id'), table_name='paper_part_submissions')
    op.drop_table('paper_part_submissions')
    
    op.drop_index(op.f('ix_paper_challenge_parts_challenge_id'), table_name='paper_challenge_parts')
    op.drop_index(op.f('ix_paper_challenge_parts_id'), table_name='paper_challenge_parts')
    op.drop_table('paper_challenge_parts')
    
    op.drop_index(op.f('ix_paper_challenges_paper_id'), table_name='paper_challenges')
    op.drop_index(op.f('ix_paper_challenges_id'), table_name='paper_challenges')
    op.drop_table('paper_challenges')
