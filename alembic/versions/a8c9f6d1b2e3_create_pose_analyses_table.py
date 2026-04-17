"""create pose analyses table

Revision ID: a8c9f6d1b2e3
Revises: 754d691b5318
Create Date: 2026-04-16 19:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a8c9f6d1b2e3"
down_revision: Union[str, Sequence[str], None] = "754d691b5318"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "pose_analyses",
        sa.Column("analysis_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("member_id", sa.BigInteger(), nullable=False),
        sa.Column("side_view", sa.String(length=10), nullable=False),
        sa.Column("overall_status", sa.String(length=20), nullable=False),
        sa.Column("overall_confidence", sa.Float(), nullable=False),
        sa.Column("front_confidence", sa.Float(), nullable=False),
        sa.Column("side_confidence", sa.Float(), nullable=False),
        sa.Column("neck_forward_angle", sa.Float(), nullable=False),
        sa.Column("shoulder_slope", sa.Float(), nullable=False),
        sa.Column("hip_slope", sa.Float(), nullable=False),
        sa.Column("spine_alignment", sa.Float(), nullable=False),
        sa.Column("asymmetry_score", sa.Float(), nullable=False),
        sa.Column("forward_head_detected", sa.Boolean(), nullable=False),
        sa.Column("issues", sa.JSON(), nullable=False),
        sa.Column("recommendations", sa.JSON(), nullable=False),
        sa.Column("analyzed_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("overall_status IN ('good', 'warning', 'bad')", name="chk_pose_analyses_status"),
        sa.CheckConstraint("side_view IN ('left', 'right')", name="chk_pose_analyses_side_view"),
        sa.ForeignKeyConstraint(["member_id"], ["members.member_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("analysis_id"),
    )
    op.create_index(op.f("ix_pose_analyses_member_id"), "pose_analyses", ["member_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_pose_analyses_member_id"), table_name="pose_analyses")
    op.drop_table("pose_analyses")
