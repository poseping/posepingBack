"""create_user_lifestyle_habits_table

Revision ID: d4b6c2e9a113
Revises: ff4b55a3600f
Create Date: 2026-04-24 18:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d4b6c2e9a113"
down_revision: Union[str, Sequence[str], None] = "ff4b55a3600f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "user_lifestyle_habits",
        sa.Column("habit_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("member_id", sa.BigInteger(), nullable=False),
        sa.Column("sitting_hours_per_day", sa.String(length=100), nullable=True),
        sa.Column("exercise_days_per_week", sa.String(length=100), nullable=True),
        sa.Column("pain_areas", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["member_id"], ["members.member_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("habit_id"),
        sa.UniqueConstraint("member_id", name="uq_user_lifestyle_habits_member_id"),
    )
    op.create_index(
        op.f("ix_user_lifestyle_habits_member_id"),
        "user_lifestyle_habits",
        ["member_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_user_lifestyle_habits_member_id"), table_name="user_lifestyle_habits")
    op.drop_table("user_lifestyle_habits")
