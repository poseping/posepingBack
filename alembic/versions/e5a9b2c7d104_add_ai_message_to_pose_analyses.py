"""add ai message to pose analyses

Revision ID: e5a9b2c7d104
Revises: d4b6c2e9a113
Create Date: 2026-05-07 10:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e5a9b2c7d104"
down_revision: Union[str, Sequence[str], None] = "d4b6c2e9a113"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("pose_analyses", sa.Column("ai_message", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("pose_analyses", "ai_message")
