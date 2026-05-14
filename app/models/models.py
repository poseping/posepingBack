from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
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
    craniovertebral_angle: Mapped[float | None] = mapped_column(Float, nullable=True)
    shoulder_slope: Mapped[float] = mapped_column(Float, nullable=False)
    hip_slope: Mapped[float | None] = mapped_column(Float, nullable=True)
    spine_alignment: Mapped[float | None] = mapped_column(Float, nullable=True)
    asymmetry_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    forward_head_detected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    issues: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    ai_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    posture_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_grade: Mapped[str | None] = mapped_column(String(30), nullable=True)
    score_breakdown: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    score_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    analyzed_at: Mapped[datetime] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())


class WebcamAlertType(Base):
    __tablename__ = "webcam_alert_type"

    alert_type_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    alert_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class WebcamSession(Base):
    __tablename__ = "webcam_sessions"

    session_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    member_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("members.member_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    profile_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("user_posture_profiles.profile_id", ondelete="SET NULL"),
        nullable=True,
    )
    started_at: Mapped[datetime] = mapped_column(nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(nullable=True)
    good_frames: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    warning_frames: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    bad_frames: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    cause_counts: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class UserPostureProfile(Base):
    __tablename__ = "user_posture_profiles"

    profile_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    member_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("members.member_id", ondelete="CASCADE"),
        nullable=False,
    )
    profile_name: Mapped[str] = mapped_column(String(255), nullable=False, server_default="기본 자세")
    monitor_label: Mapped[str | None] = mapped_column(String(100), nullable=True)
    display_order: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    reference_landmarks: Mapped[dict] = mapped_column(JSONB, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())


class UserLifestyleHabit(Base):
    __tablename__ = "user_lifestyle_habits"

    __table_args__ = (
        UniqueConstraint("member_id", name="uq_user_lifestyle_habits_member_id"),
    )

    habit_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    member_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("members.member_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sitting_hours_per_day: Mapped[str | None] = mapped_column(String(100), nullable=True)
    exercise_days_per_week: Mapped[str | None] = mapped_column(String(100), nullable=True)
    pain_areas: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
