"""add_ai_comment_mode_to_user_webcam_settings

Revision ID: f3b1e9d2c845
Revises: d1e7f3a5b920
Create Date: 2026-05-28 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "f3b1e9d2c845"
down_revision: Union[str, Sequence[str], None] = "e7a2c9f4b610"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "user_webcam_settings",
        sa.Column("ai_comment_mode", sa.String(20), nullable=False, server_default="ai"),
    )


def downgrade() -> None:
    op.drop_column("user_webcam_settings", "ai_comment_mode")
