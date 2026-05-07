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
from app.api.dependencies import verify_auth

router = APIRouter()
DEV_ALWAYS_FIRST_LOGIN_MEMBER_IDS = {13}


# ==================== Request/Response 모델 ====================

class UpdateProfileRequest(BaseModel):
    """프로필 수정 요청"""
    nickname: str


class KakaoLoginRequest(BaseModel):
    """카카오 로그인 요청"""
    code: str  # 카카오 인가 코드


class GoogleLoginRequest(BaseModel):
    """구글 로그인 요청"""
    code: str  # 구글 인가 코드


class AdminLoginRequest(BaseModel):
    """임시 관리자 로그인 요청"""
    admin_id: str
    password: str


class DevLoginRequest(BaseModel):
    """개발용 회원 선택 로그인 요청"""
    member_id: int


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
    is_new_member: bool


class VerifyResponse(BaseModel):
    """토큰 검증 응답"""
    success: bool
    user: Optional[UserResponse] = None
    error: Optional[str] = None


class DevMemberResponse(BaseModel):
    """개발용 선택 로그인 회원 응답"""
    member_id: int
    provider: str
    nickname: Optional[str]
    role: str

    class Config:
        from_attributes = True


class NicknameResponse(BaseModel):
    """랜덤 닉네임 응답"""
    nickname: str


def should_show_first_login(member: Member | None, created_in_request: bool = False) -> bool:
    if created_in_request:
        return True
    if member is None:
        return False
    if member.last_login_at is None:
        return True
    return member.last_login_at <= member.created_at


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
    print(f"🔍 카카오 인가 코드: {request.code[:20]}...")

    # 1️⃣ 인가 코드로 카카오 액세스 토큰 교환
    token_data = KakaoOAuthService.get_access_token(request.code)
    if not token_data or "access_token" not in token_data:
        raise HTTPException(status_code=401, detail="카카오 인증 실패")

    # 2️⃣ 카카오에서 사용자 정보 조회
    user_info = KakaoOAuthService.get_user_info(token_data["access_token"])
    print(f"🔍 사용자정보 조회 결과: {user_info}")

    if not user_info:
        raise HTTPException(status_code=401, detail="카카오 인증 실패")

    # 3️⃣ DB에서 기존 회원 조회
    member = db.query(Member).filter(
        Member.provider == "KAKAO",
        Member.provider_user_id == user_info["social_id"],
    ).first()

    # 4️⃣ 신규 회원 생성 또는 기존 회원 업데이트
    is_new_member = should_show_first_login(member, created_in_request=member is None)

    if is_new_member:
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

    # 5️⃣ JWT 토큰 생성
    access_token = JWTService.create_access_token(member.member_id)

    return LoginResponse(
        success=True,
        access_token=access_token,
        token_type="Bearer",
        expires_in=3600,  # 1시간
        user=UserResponse.from_orm(member),
        is_new_member=is_new_member,
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
    # 1️⃣ 인가 코드로 구글 토큰 교환 (팝업 flow → redirect_uri="postmessage")
    token_data = GoogleOAuthService.get_access_token(request.code, redirect_uri="postmessage")
    if not token_data or "id_token" not in token_data:
        raise HTTPException(status_code=401, detail="구글 인증 실패")

    # 2️⃣ ID Token으로 사용자 정보 검증
    user_info_google = GoogleOAuthService.verify_id_token(token_data["id_token"])
    if not user_info_google:
        raise HTTPException(status_code=401, detail="구글 인증 실패")

    # 3️⃣ 필요한 정보 추출
    social_id = user_info_google.get("sub")  # Google User ID
    profile_image_url = user_info_google.get("picture")

    if not social_id:
        raise HTTPException(status_code=401, detail="사용자 정보 조회 실패")

    # 4️⃣ DB에서 기존 회원 조회
    member = db.query(Member).filter(
        Member.provider == "GOOGLE",
        Member.provider_user_id == social_id,
    ).first()

    # 5️⃣ 신규 회원 생성 또는 기존 회원 업데이트
    is_new_member = should_show_first_login(member, created_in_request=member is None)

    if is_new_member:
        # 신규 회원 - 닉네임 자동 생성
        final_nickname = generate_nickname_with_fallback(db)

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

    # 6️⃣ JWT 토큰 생성
    access_token = JWTService.create_access_token(member.member_id)

    return LoginResponse(
        success=True,
        access_token=access_token,
        token_type="Bearer",
        expires_in=3600,  # 1시간
        user=UserResponse.from_orm(member),
        is_new_member=is_new_member,
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


# ==================== 개발용 로그인 ====================

@router.post("/admin-login", response_model=LoginResponse)
async def admin_login(request: AdminLoginRequest, db: Session = Depends(get_db)):
    """
    임시 관리자 로그인.
    소셜 로그인 없이 관리자 JWT를 발급하기 위한 개발용 엔드포인트.
    """
    from app.core.config import settings
    if settings.app_env != "dev":
        raise HTTPException(status_code=404, detail="Not found")

    if request.admin_id != "admin" or request.password != "admin1234":
        raise HTTPException(status_code=401, detail="Invalid admin credentials")

    member = db.query(Member).filter(
        Member.provider == "KAKAO",
        Member.provider_user_id == "admin-local-user",
    ).first()

    is_new_member = should_show_first_login(member, created_in_request=member is None)

    if is_new_member:
        member = Member(
            provider="KAKAO",
            provider_user_id="admin-local-user",
            email="admin@local.dev",
            name="관리자",
            nickname="관리자",
            status="ACTIVE",
            role="ADMIN",
            email_verified=True,
            last_login_at=datetime.now(timezone.utc),
        )
        db.add(member)
        db.commit()
        db.refresh(member)
    else:
        member.status = "ACTIVE"
        member.role = "ADMIN"
        member.last_login_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(member)

    access_token = JWTService.create_access_token(member.member_id)

    return LoginResponse(
        success=True,
        access_token=access_token,
        token_type="Bearer",
        expires_in=3600,
        user=UserResponse.from_orm(member),
        is_new_member=is_new_member,
    )


@router.get("/dev-members", response_model=list[DevMemberResponse])
async def get_dev_members(db: Session = Depends(get_db)):
    """
    개발용 선택 로그인 대상 회원 목록 조회.
    member_id 6~15 범위의 회원과 기존 개발용 계정을 함께 반환한다.
    """
    from app.core.config import settings
    if settings.app_env != "dev":
        raise HTTPException(status_code=404, detail="Not found")

    legacy_dev_member = db.query(Member).filter(
        Member.provider == "KAKAO",
        Member.provider_user_id == "dev-test-user",
    ).first()

    if not legacy_dev_member:
        legacy_dev_member = Member(
            provider="KAKAO",
            provider_user_id="dev-test-user",
            nickname="개발자",
            status="ACTIVE",
            role="USER",
            last_login_at=datetime.now(timezone.utc),
        )
        db.add(legacy_dev_member)
        db.commit()
        db.refresh(legacy_dev_member)

    members = (
        db.query(Member)
        .filter(Member.member_id >= 6, Member.member_id <= 15)
        .order_by(Member.member_id.asc())
        .all()
    )

    member_map = {member.member_id: member for member in members}
    member_map[legacy_dev_member.member_id] = legacy_dev_member

    return [
        DevMemberResponse.from_orm(member)
        for member in sorted(member_map.values(), key=lambda member: member.member_id)
    ]


@router.post("/dev-login", response_model=LoginResponse)
async def dev_login(request: DevLoginRequest, db: Session = Depends(get_db)):
    """
    개발용 선택 로그인 (DEBUG 환경에서만 동작)
    member_id 6~15 범위의 기존 회원으로 JWT 토큰 발급
    """
    from app.core.config import settings
    if settings.app_env != "dev":
        raise HTTPException(status_code=404, detail="Not found")

    if request.member_id < 6 or request.member_id > 15:
        raise HTTPException(status_code=400, detail="허용된 개발용 회원 ID 범위를 벗어났습니다.")

    member = db.query(Member).filter(Member.member_id == request.member_id).first()
    if not member:
        raise HTTPException(status_code=404, detail="선택한 회원을 찾을 수 없습니다.")

    is_new_member = (
        member.member_id in DEV_ALWAYS_FIRST_LOGIN_MEMBER_IDS
        or should_show_first_login(member)
    )

    member.last_login_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(member)

    access_token = JWTService.create_access_token(member.member_id)

    return LoginResponse(
        success=True,
        access_token=access_token,
        token_type="Bearer",
        expires_in=3600,
        user=UserResponse.from_orm(member),
        is_new_member=is_new_member,
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



# ==================== 내 정보 수정 ====================

@router.patch("/me", response_model=UserResponse)
async def update_profile(
    request: UpdateProfileRequest,
    member: Member = Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """닉네임 변경"""
    nickname = request.nickname.strip()
    if not nickname:
        raise HTTPException(status_code=400, detail="닉네임을 입력해주세요")
    if len(nickname) > 20:
        raise HTTPException(status_code=400, detail="닉네임은 20자 이하여야 합니다")
    member.nickname = nickname
    db.commit()
    db.refresh(member)
    return UserResponse.from_orm(member)


@router.delete("/me")
async def delete_account(
    member: Member = Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """회원 탈퇴 (status WITHDRAWN 처리)"""
    member.status = "WITHDRAWN"
    db.commit()
    return {"success": True, "message": "회원 탈퇴가 완료되었습니다"}


@router.get("/me/random-nickname", response_model=NicknameResponse)
async def get_random_nickname(db: Session = Depends(get_db)):
    """회원 수정 시 랜덤 닉네임 생성"""
    nickname = generate_nickname_with_fallback(db)
    return NicknameResponse(nickname=nickname)