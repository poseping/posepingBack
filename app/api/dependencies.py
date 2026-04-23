from typing import Optional

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.models import Member
from app.services.auth_service import JWTService


async def verify_auth(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> Member:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization 헤더가 필요합니다")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="유효하지 않은 Authorization 헤더")

    member_id = JWTService.verify_token(parts[1])
    if not member_id:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰")

    member = db.query(Member).filter(Member.member_id == member_id).first()
    if not member or member.status == "WITHDRAWN":
        raise HTTPException(status_code=401, detail="사용자를 찾을 수 없음")

    return member


async def require_admin(member: Member = Depends(verify_auth)) -> Member:
    if member.role != "ADMIN":
        raise HTTPException(status_code=403, detail="Admin access is required.")

    return member
