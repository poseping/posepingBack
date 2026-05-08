"""add issues to pose analyses

Revision ID: f2c8d4a91b70
Revises: e5a9b2c7d104
Create Date: 2026-05-07 10:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "f2c8d4a91b70"
down_revision: Union[str, Sequence[str], None] = "e5a9b2c7d104"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "pose_analyses",
        sa.Column(
            "issues",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.alter_column("pose_analyses", "issues", server_default=None)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("pose_analyses", "issues")
