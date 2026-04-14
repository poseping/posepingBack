"""
인증 API 엔드포인트
카카오, 구글 소셜 로그인
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from app.db.session import get_db
from app.models.models import Member
from app.services.auth_service import (
    JWTService,
    KakaoOAuthService,
    GoogleOAuthService,
)
from app.services.nickname_service import generate_nickname_with_fallback

router = APIRouter()


# ==================== Request/Response 모델 ====================

class KakaoLoginRequest(BaseModel):
    """카카오 로그인 요청"""
    access_token: str  # 카카오 액세스 토큰


class GoogleLoginRequest(BaseModel):
    """구글 로그인 요청"""
    token: str  # 구글 ID Token 또는 인가 코드


class UserResponse(BaseModel):
    """사용자 정보 응답"""
    member_id: int
    provider: str
    nickname: Optional[str]
    profile_image_url: Optional[str]
    role: str

    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    """로그인 응답"""
    success: bool
    access_token: str
    token_type: str
    expires_in: int  # 초 단위
    user: UserResponse


class VerifyResponse(BaseModel):
    """토큰 검증 응답"""
    success: bool
    user: Optional[UserResponse] = None
    error: Optional[str] = None


# ==================== 카카오 로그인 ====================

@router.post("/kakao", response_model=LoginResponse)
async def kakao_login(request: KakaoLoginRequest, db: Session = Depends(get_db)):
    """
    카카오 로그인 엔드포인트

    Args:
        request: 카카오 액세스 토큰
        db: DB 세션

    Returns:
        LoginResponse: JWT 토큰 및 사용자 정보
    """
    print(f"🔍 카카오 액세스 토큰: {request.access_token[:20]}...")

    # 1️⃣ 카카오에서 사용자 정보 조회 (액세스 토큰 사용)
    user_info = KakaoOAuthService.get_user_info(request.access_token)
    print(f"🔍 사용자정보 조회 결과: {user_info}")

    if not user_info:
        raise HTTPException(status_code=401, detail="카카오 인증 실패")

    # 2️⃣ DB에서 기존 회원 조회
    member = db.query(Member).filter(
        Member.provider == "KAKAO",
        Member.provider_user_id == user_info["social_id"],
    ).first()

    # 3️⃣ 신규 회원 생성 또는 기존 회원 업데이트
    if not member:
        # 신규 회원 - 닉네임 자동 생성
        # 카카오는 닉네임 정보가 없으므로 "형용사+동물" 형식으로 자동 생성
        auto_nickname = generate_nickname_with_fallback(db)

        member = Member(
            provider="KAKAO",
            provider_user_id=user_info["social_id"],
            nickname=auto_nickname,
            profile_image_url=user_info.get("profile_image_url"),
            status="ACTIVE",
            role="USER",
            last_login_at=datetime.now(timezone.utc),
        )
        db.add(member)
        db.commit()
        db.refresh(member)
    else:
        # 기존 회원 업데이트
        member.last_login_at = datetime.now(timezone.utc)
        # 기존 회원은 닉네임 변경 안 함
        member.profile_image_url = user_info.get("profile_image_url") or member.profile_image_url
        db.commit()

    # 4️⃣ JWT 토큰 생성
    access_token = JWTService.create_access_token(member.member_id)

    return LoginResponse(
        success=True,
        access_token=access_token,
        token_type="Bearer",
        expires_in=3600,  # 1시간
        user=UserResponse.from_orm(member),
    )


# ==================== 구글 로그인 ====================

@router.post("/google", response_model=LoginResponse)
async def google_login(request: GoogleLoginRequest, db: Session = Depends(get_db)):
    """
    구글 로그인 엔드포인트

    Args:
        request: 구글 ID Token
        db: DB 세션

    Returns:
        LoginResponse: JWT 토큰 및 사용자 정보
    """
    # 1️⃣ 구글 ID Token 검증
    user_info_google = GoogleOAuthService.verify_id_token(request.token)
    if not user_info_google:
        raise HTTPException(status_code=401, detail="구글 인증 실패")

    # 2️⃣ 필요한 정보 추출
    social_id = user_info_google.get("sub")  # Google User ID
    nickname = user_info_google.get("name")
    profile_image_url = user_info_google.get("picture")

    if not social_id:
        raise HTTPException(status_code=401, detail="사용자 정보 조회 실패")

    # 3️⃣ DB에서 기존 회원 조회
    member = db.query(Member).filter(
        Member.provider == "GOOGLE",
        Member.provider_user_id == social_id,
    ).first()

    # 4️⃣ 신규 회원 생성 또는 기존 회원 업데이트
    if not member:
        # 신규 회원 - 닉네임 설정
        # 구글은 name이 있으면 그것을 우선 사용, 중복되면 자동 생성
        final_nickname = generate_nickname_with_fallback(db, preferred_nickname=nickname)

        member = Member(
            provider="GOOGLE",
            provider_user_id=social_id,
            nickname=final_nickname,
            profile_image_url=profile_image_url,
            status="ACTIVE",
            role="USER",
            last_login_at=datetime.now(timezone.utc),
        )
        db.add(member)
        db.commit()
        db.refresh(member)
    else:
        # 기존 회원 업데이트
        member.last_login_at = datetime.now(timezone.utc)
        # 기존 회원은 닉네임 변경 안 함
        member.profile_image_url = profile_image_url or member.profile_image_url
        db.commit()

    # 5️⃣ JWT 토큰 생성
    access_token = JWTService.create_access_token(member.member_id)

    return LoginResponse(
        success=True,
        access_token=access_token,
        token_type="Bearer",
        expires_in=3600,  # 1시간
        user=UserResponse.from_orm(member),
    )


# ==================== 토큰 검증 ====================

@router.get("/verify", response_model=VerifyResponse)
async def verify_token(token: str, db: Session = Depends(get_db)):
    """
    JWT 토큰 검증

    Args:
        token: JWT 토큰 (Authorization 헤더에서 제거한 순수 토큰)
        db: DB 세션

    Returns:
        VerifyResponse: 검증 결과 및 사용자 정보
    """
    member_id = JWTService.verify_token(token)
    if not member_id:
        return VerifyResponse(success=False, error="Invalid token")

    member = db.query(Member).filter(Member.member_id == member_id).first()
    if not member or member.status == "WITHDRAWN":
        return VerifyResponse(success=False, error="User not found or withdrawn")

    return VerifyResponse(
        success=True,
        user=UserResponse.from_orm(member),
    )


# ==================== 로그아웃 ====================

@router.post("/logout")
async def logout(token: str):
    """
    로그아웃 (현재는 클라이언트에서 토큰 삭제만 하면 됨)

    Note:
        JWT는 서버에서 관리하지 않으므로
        클라이언트에서 로컬스토리지의 토큰 삭제로 처리됨
    """
    return {"success": True, "message": "로그아웃 되었습니다"}
