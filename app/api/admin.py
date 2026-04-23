from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.dependencies import require_admin
from app.db.session import get_db
from app.models.models import Member

router = APIRouter()


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

    class Config:
        from_attributes = True


@router.get("/members", response_model=list[AdminMemberResponse])
async def list_members(
    _: Member = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return db.query(Member).order_by(Member.created_at.desc()).all()
