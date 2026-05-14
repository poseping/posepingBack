"""rename_webcam_session_counts_to_frames

Revision ID: c9f3a1b8d2e0
Revises: b9e2c4d7a810
Create Date: 2026-05-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

revision: str = "c9f3a1b8d2e0"
down_revision: Union[str, Sequence[str], None] = "b9e2c4d7a810"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("webcam_sessions", "good_count",    new_column_name="good_frames")
    op.alter_column("webcam_sessions", "warning_count", new_column_name="warning_frames")
    op.alter_column("webcam_sessions", "bad_count",     new_column_name="bad_frames")


def downgrade() -> None:
    op.alter_column("webcam_sessions", "good_frames",    new_column_name="good_count")
    op.alter_column("webcam_sessions", "warning_frames", new_column_name="warning_count")
    op.alter_column("webcam_sessions", "bad_frames",     new_column_name="bad_count")
