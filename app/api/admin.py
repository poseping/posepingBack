from datetime import datetime
from enum import Enum
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.api.dependencies import require_admin
from app.db.session import get_db
from app.models.models import Member

router = APIRouter()


class AdminMemberRole(str, Enum):
    ADMIN = "ADMIN"
    USER = "USER"


class AdminMemberStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class AdminMemberResponse(BaseModel):
    member_id: int
    provider: str
    provider_user_id: str
    email: Optional[str]
    name: Optional[str]
    nickname: Optional[str]
    profile_image_url: Optional[str]
    status: str
    role: str
    email_verified: bool
    last_login_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class AdminMemberUpdateRequest(BaseModel):
    role: Optional[AdminMemberRole] = None
    status: Optional[AdminMemberStatus] = None


@router.get("/members", response_model=list[AdminMemberResponse])
async def list_members(
    _: Member = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return db.query(Member).order_by(Member.created_at.desc()).all()


@router.patch("/members/{member_id}", response_model=AdminMemberResponse)
async def update_member(
    member_id: int,
    request: AdminMemberUpdateRequest,
    _: Member = Depends(require_admin),
    db: Session = Depends(get_db),
):
    if request.role is None and request.status is None:
        raise HTTPException(status_code=400, detail="role 또는 status 중 하나는 필요합니다")

    member = db.query(Member).filter(Member.member_id == member_id).first()
    if not member:
        raise HTTPException(status_code=404, detail="회원을 찾을 수 없습니다")

    if request.role is not None:
        member.role = request.role.value

    if request.status is not None:
        member.status = request.status.value

    member.updated_at = datetime.now()
    db.commit()
    db.refresh(member)

    return member
