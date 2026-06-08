"""add_user_api_settings

Revision ID: a1b2c3d4e5f6
Revises: f3b1e9d2c845
Create Date: 2026-06-08 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "f3b1e9d2c845"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_api_settings",
        sa.Column("setting_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "member_id",
            sa.BigInteger(),
            sa.ForeignKey("members.member_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("ai_api_key_enc", sa.Text(), nullable=True),
        sa.Column("is_ai_enabled", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("member_id", name="uq_user_api_settings_member_id"),
    )
    op.drop_column("user_webcam_settings", "ai_comment_mode")


def downgrade() -> None:
    op.add_column(
        "user_webcam_settings",
        sa.Column("ai_comment_mode", sa.String(20), nullable=False, server_default="ai"),
    )
    op.drop_table("user_api_settings")
