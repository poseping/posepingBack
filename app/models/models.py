from datetime import datetime

from sqlalchemy import BigInteger, Boolean, CheckConstraint, String, Text, UniqueConstraint, func
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
