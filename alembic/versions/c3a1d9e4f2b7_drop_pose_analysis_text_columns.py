"""drop pose analysis text columns

Revision ID: c3a1d9e4f2b7
Revises: b7d4e2f9c1a0
Create Date: 2026-04-16 21:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c3a1d9e4f2b7"
down_revision: Union[str, Sequence[str], None] = "b7d4e2f9c1a0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("pose_analyses", "recommendations")
    op.drop_column("pose_analyses", "issues")


def downgrade() -> None:
    op.add_column(
        "pose_analyses",
        sa.Column("issues", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
    )
    op.add_column(
        "pose_analyses",
        sa.Column("recommendations", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
    )
