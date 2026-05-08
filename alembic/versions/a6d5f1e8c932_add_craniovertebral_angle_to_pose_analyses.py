"""add craniovertebral angle to pose analyses

Revision ID: a6d5f1e8c932
Revises: f2c8d4a91b70
Create Date: 2026-05-08 14:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a6d5f1e8c932"
down_revision: Union[str, Sequence[str], None] = "f2c8d4a91b70"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("pose_analyses", sa.Column("craniovertebral_angle", sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("pose_analyses", "craniovertebral_angle")
