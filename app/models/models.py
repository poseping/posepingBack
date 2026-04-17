from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Member(Base):
    __tablename__ = "members"

    __table_args__ = (
        UniqueConstraint("provider", "provider_user_id", name="uq_members_provider_user"),
        CheckConstraint(
            "provider IN ('KAKAO', 'GOOGLE')",
            name="chk_members_provider",
        ),
        CheckConstraint(
            "role IN ('USER', 'ADMIN')",
            name="chk_members_role",
        ),
        CheckConstraint(
            "status IN ('ACTIVE', 'INACTIVE', 'SUSPENDED', 'WITHDRAWN')",
            name="chk_members_status",
        ),
        CheckConstraint(
            "deleted_at IS NULL OR status = 'WITHDRAWN'",
            name="chk_members_deleted_at_status",
        ),
    )

    member_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(20), nullable=False)
    provider_user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    nickname: Mapped[str | None] = mapped_column(String(50), nullable=True)
    profile_image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="ACTIVE")
    role: Mapped[str] = mapped_column(String(20), nullable=False, server_default="USER")
    email_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    last_login_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    deleted_at: Mapped[datetime | None] = mapped_column(nullable=True)


class PoseAnalysis(Base):
    __tablename__ = "pose_analyses"

    __table_args__ = (
        CheckConstraint(
            "side_view IN ('left', 'right')",
            name="chk_pose_analyses_side_view",
        ),
        CheckConstraint(
            "overall_status IN ('good', 'warning', 'bad')",
            name="chk_pose_analyses_status",
        ),
    )

    analysis_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    member_id: Mapped[int] = mapped_column(
        ForeignKey("members.member_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    side_view: Mapped[str] = mapped_column(String(10), nullable=False)
    overall_status: Mapped[str] = mapped_column(String(20), nullable=False)
    overall_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    front_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    side_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    neck_forward_angle: Mapped[float] = mapped_column(Float, nullable=False)
    shoulder_slope: Mapped[float] = mapped_column(Float, nullable=False)
    hip_slope: Mapped[float | None] = mapped_column(Float, nullable=True)
    spine_alignment: Mapped[float | None] = mapped_column(Float, nullable=True)
    asymmetry_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    forward_head_detected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    analyzed_at: Mapped[datetime] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
