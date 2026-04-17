"""allow partial photo analysis metrics

Revision ID: b7d4e2f9c1a0
Revises: a8c9f6d1b2e3
Create Date: 2026-04-16 20:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b7d4e2f9c1a0"
down_revision: Union[str, Sequence[str], None] = "a8c9f6d1b2e3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("pose_analyses", "hip_slope", existing_type=sa.Float(), nullable=True)
    op.alter_column("pose_analyses", "spine_alignment", existing_type=sa.Float(), nullable=True)
    op.alter_column("pose_analyses", "asymmetry_score", existing_type=sa.Float(), nullable=True)


def downgrade() -> None:
    op.alter_column("pose_analyses", "asymmetry_score", existing_type=sa.Float(), nullable=False)
    op.alter_column("pose_analyses", "spine_alignment", existing_type=sa.Float(), nullable=False)
    op.alter_column("pose_analyses", "hip_slope", existing_type=sa.Float(), nullable=False)
