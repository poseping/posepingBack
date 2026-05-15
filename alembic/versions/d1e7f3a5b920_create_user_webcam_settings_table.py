"""create_user_webcam_settings_table

Revision ID: d1e7f3a5b920
Revises: c9f3a1b8d2e0
Create Date: 2026-05-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "d1e7f3a5b920"
down_revision: Union[str, Sequence[str], None] = "c9f3a1b8d2e0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_webcam_settings",
        sa.Column("setting_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "member_id",
            sa.BigInteger(),
            sa.ForeignKey("members.member_id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
            index=True,
        ),
        sa.Column("posture_sensitivity", sa.String(10), nullable=False, server_default="medium"),
        sa.Column("ai_comment_threshold_sec", sa.Integer(), nullable=False, server_default="60"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("user_webcam_settings")
