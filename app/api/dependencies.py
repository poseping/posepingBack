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
        raise HTTPException(status_code=401, detail="Authorization header is required.")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header.")

    member_id = JWTService.verify_token(parts[1])
    if not member_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    member = db.query(Member).filter(Member.member_id == member_id).first()
    if not member or member.status == "WITHDRAWN":
        raise HTTPException(status_code=401, detail="Member not found.")

    return member
