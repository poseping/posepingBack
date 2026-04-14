"""
인증 서비스
JWT 토큰 생성/검증, OAuth 처리
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import requests
from jose import JWTError, jwt
import os
from dotenv import load_dotenv

load_dotenv()

# JWT 설정
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# OAuth 설정
KAKAO_CLIENT_ID = os.getenv("KAKAO_CLIENT_ID")
KAKAO_CLIENT_SECRET = os.getenv("KAKAO_CLIENT_SECRET")
KAKAO_REDIRECT_URI = os.getenv("KAKAO_REDIRECT_URI")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")


class JWTService:
    """JWT 토큰 관리"""

    @staticmethod
    def create_access_token(member_id: int, expires_delta: Optional[timedelta] = None) -> str:
        """
        JWT 액세스 토큰 생성

        Args:
            member_id: 회원 ID
            expires_delta: 만료 시간 (기본: ACCESS_TOKEN_EXPIRE_MINUTES)

        Returns:
            JWT 토큰 문자열
        """
        if expires_delta is None:
            expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        expire = datetime.now(timezone.utc) + expires_delta
        to_encode = {
            "sub": str(member_id),
            "exp": expire,
            "iat": datetime.now(timezone.utc)
        }

        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def verify_token(token: str) -> Optional[int]:
        """
        JWT 토큰 검증

        Args:
            token: JWT 토큰

        Returns:
            성공하면 member_id, 실패하면 None
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            member_id: int = int(payload.get("sub"))
            if member_id is None:
                return None
            return member_id
        except JWTError:
            return None


class KakaoOAuthService:
    """카카오 OAuth 통합"""

    @staticmethod
    def get_access_token(code: str) -> Optional[Dict[str, Any]]:
        """
        인가 코드로 카카오 액세스 토큰 받기

        Args:
            code: 카카오에서 받은 인가 코드

        Returns:
            access_token 등 응답 데이터, 실패하면 None
        """
        try:
            url = "https://kauth.kakao.com/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = {
                "grant_type": "authorization_code",
                "client_id": KAKAO_CLIENT_ID,
                "client_secret": KAKAO_CLIENT_SECRET,
                "code": code,
                "redirect_uri": KAKAO_REDIRECT_URI,
            }

            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"❌ 카카오 토큰 요청 실패: {e}")
            return None

    @staticmethod
    def get_user_info(access_token: str) -> Optional[Dict[str, Any]]:
        """
        카카오 액세스 토큰으로 사용자 정보 조회

        Args:
            access_token: 카카오 액세스 토큰

        Returns:
            사용자 정보 (id, nickname, profile_image_url 등)
        """
        try:
            url = "https://kapi.kakao.com/v2/user/me"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/x-www-form-urlencoded",
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            # 필요한 정보 추출
            return {
                "social_id": str(data.get("id")),
                "nickname": data.get("kakao_account", {}).get("profile", {}).get("nickname"),
                "profile_image_url": data.get("kakao_account", {}).get("profile", {}).get("profile_image_url"),
            }
        except requests.RequestException as e:
            print(f"❌ 카카오 사용자 정보 조회 실패: {e}")
            return None


class GoogleOAuthService:
    """구글 OAuth 통합"""

    @staticmethod
    def verify_id_token(id_token: str) -> Optional[Dict[str, Any]]:
        """
        구글 ID Token 검증 (옵션 1: 구글 API 호출)

        Args:
            id_token: 구글에서 받은 ID Token

        Returns:
            사용자 정보 (sub, email, name, picture)
        """
        try:
            url = f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"❌ 구글 ID Token 검증 실패: {e}")
            return None

    @staticmethod
    def get_access_token(code: str) -> Optional[Dict[str, Any]]:
        """
        인가 코드로 구글 액세스 토큰 받기 (옵션 2: 서버에서 처리)

        Args:
            code: 구글에서 받은 인가 코드

        Returns:
            access_token 등 응답 데이터
        """
        try:
            url = "https://oauth2.googleapis.com/token"
            data = {
                "grant_type": "authorization_code",
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "code": code,
                "redirect_uri": GOOGLE_REDIRECT_URI,
            }

            response = requests.post(url, data=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"❌ 구글 토큰 요청 실패: {e}")
            return None
