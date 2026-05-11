"""add posture score to pose analyses

Revision ID: b9e2c4d7a810
Revises: a6d5f1e8c932
Create Date: 2026-05-11 10:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "b9e2c4d7a810"
down_revision: Union[str, Sequence[str], None] = "a6d5f1e8c932"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("pose_analyses", sa.Column("posture_score", sa.Float(), nullable=True))
    op.add_column("pose_analyses", sa.Column("score_grade", sa.String(length=30), nullable=True))
    op.add_column(
        "pose_analyses",
        sa.Column("score_breakdown", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column("pose_analyses", sa.Column("score_version", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("pose_analyses", "score_version")
    op.drop_column("pose_analyses", "score_breakdown")
    op.drop_column("pose_analyses", "score_grade")
    op.drop_column("pose_analyses", "posture_score")
