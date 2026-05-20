"""add_sleep_position_to_user_lifestyle_habits

Revision ID: e7a2c9f4b610
Revises: d1e7f3a5b920
Create Date: 2026-05-20 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e7a2c9f4b610"
down_revision: Union[str, Sequence[str], None] = "d1e7f3a5b920"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "user_lifestyle_habits",
        sa.Column("sleep_position", sa.String(length=100), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_lifestyle_habits", "sleep_position")
