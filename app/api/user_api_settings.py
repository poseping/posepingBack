from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

from app.api.dependencies import verify_auth
from app.db.session import get_db
from app.models.models import UserApiSettings
from app.services.crypto_utils import encrypt_key, decrypt_key

router = APIRouter()


# ==================== 스키마 ====================

class UserApiSettingsResponse(BaseModel):
    is_ai_enabled: bool
    ai_api_key_masked: Optional[str]  # "AIza***...***", 미등록 시 null

    class Config:
        from_attributes = True


class UserApiSettingsUpdateRequest(BaseModel):
    is_ai_enabled: Optional[bool] = None
    ai_api_key: Optional[str] = None  # 빈 문자열 "" → 키 삭제


# ==================== 헬퍼 ====================

def _mask_key(raw: str) -> str:
    if len(raw) <= 8:
        return "***"
    return raw[:6] + "***" + raw[-4:]


def get_user_ai_key(db: Session, member_id: int) -> Optional[str]:
    row = db.query(UserApiSettings).filter_by(member_id=member_id).first()
    if row and row.is_ai_enabled and row.ai_api_key_enc:
        return decrypt_key(row.ai_api_key_enc)
    return None


# ==================== 엔드포인트 ====================

@router.get("/api-settings", response_model=UserApiSettingsResponse)
def get_api_settings(
    db: Session = Depends(get_db),
    current_user=Depends(verify_auth),
):
    row = db.query(UserApiSettings).filter_by(member_id=current_user.member_id).first()
    if not row:
        return UserApiSettingsResponse(is_ai_enabled=False, ai_api_key_masked=None)

    masked = _mask_key(decrypt_key(row.ai_api_key_enc)) if row.ai_api_key_enc else None
    return UserApiSettingsResponse(is_ai_enabled=row.is_ai_enabled, ai_api_key_masked=masked)


@router.patch("/api-settings", response_model=UserApiSettingsResponse)
def update_api_settings(
    body: UserApiSettingsUpdateRequest,
    db: Session = Depends(get_db),
    current_user=Depends(verify_auth),
):
    row = db.query(UserApiSettings).filter_by(member_id=current_user.member_id).first()
    if not row:
        row = UserApiSettings(member_id=current_user.member_id)
        db.add(row)

    # 키 처리
    if body.ai_api_key is not None:
        if body.ai_api_key == "":
            row.ai_api_key_enc = None
            row.is_ai_enabled = False  # 키 삭제 시 AI 모드 강제 off
        else:
            row.ai_api_key_enc = encrypt_key(body.ai_api_key)

    # 모드 토글 처리 (키가 있을 때만 허용)
    if body.is_ai_enabled is not None:
        if body.is_ai_enabled and not row.ai_api_key_enc:
            raise HTTPException(status_code=400, detail="API 키를 먼저 등록해주세요.")
        row.is_ai_enabled = body.is_ai_enabled

    row.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(row)

    masked = _mask_key(decrypt_key(row.ai_api_key_enc)) if row.ai_api_key_enc else None
    return UserApiSettingsResponse(is_ai_enabled=row.is_ai_enabled, ai_api_key_masked=masked)
