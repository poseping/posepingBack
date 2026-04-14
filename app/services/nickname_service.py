"""
닉네임 생성 서비스
형용사 + 동물명사 조합으로 유니크한 닉네임 자동 생성
"""

import random
from typing import Optional
from sqlalchemy.orm import Session
from app.models.models import Member


# 형용사 목록
ADJECTIVES = [
    "배고픈", "졸린", "행복한", "슬픈", "화난", "신난", "피곤한",
    "우아한", "멋진", "귀여운", "야무진", "똑똑한", "이상한", "신비한",
    "반짝이는", "조용한", "시끄러운", "밝은", "어두운", "따뜻한", "차가운",
    "빠른", "느린", "강한", "약한", "크기", "작은", "크큼한",
    "커다란", "아기", "풍부한", "빈", "높은", "낮은", "깊은",
    "천진한", "매력적인", "멋스러운", "활발한", "침착한", "거친", "부드러운",
    "밤색", "하얀", "검은", "빨간", "노란", "파란", "초록",
    "보라색", "분홍", "주황", "아무것도", "모든", "특별한", "보통의"
]

# 동물명사 목록
ANIMALS = [
    "얼룩말", "사자", "호랑이", "곰", "사슴", "토끼", "여우",
    "늑대", "독수리", "올빼미", "돌고래", "고래", "상어", "뱀",
    "악어", "개미", "나비", "꿀벌", "거미", "개구리", "오리",
    "펭귄", "코끼리", "기린", "원숭이", "판다", "캥거루", "낙타",
    "얼음곰", "북극여우", "라마", "알파카", "고슴도치", "너구리", "비버",
    "수달", "바다사자", "바다표범", "매", "참새", "까마귀", "비둘기",
    "흰백조", "타조", "잉어", "금붕어", "열대어", "게", "가재",
    "소라", "문어", "해파리", "불가사리", "성게", "진주조개", "굴"
]


def generate_unique_nickname(db: Session) -> str:
    """
    형용사 + 동물명사 조합으로 유니크한 닉네임 생성

    Args:
        db: DB 세션

    Returns:
        생성된 유니크한 닉네임 (예: "배고픈얼룩말32")
    """
    max_attempts = 100  # 최대 시도 횟수

    for attempt in range(max_attempts):
        # 1️⃣ 형용사 + 동물명사 조합
        adjective = random.choice(ADJECTIVES)
        animal = random.choice(ANIMALS)
        nickname = f"{adjective}{animal}"

        # 2️⃣ DB에서 기본 닉네임 존재 여부 확인
        existing = db.query(Member).filter(Member.nickname == nickname).first()

        if not existing:
            # 중복 없음 → 바로 반환
            return nickname

        # 3️⃣ 중복 있음 → 숫자 붙이기 (닉네임1, 닉네임2, ...)
        for suffix in range(1, 1000):
            nickname_with_suffix = f"{nickname}{suffix}"
            existing = db.query(Member).filter(
                Member.nickname == nickname_with_suffix
            ).first()

            if not existing:
                return nickname_with_suffix

    # 만약 100번 시도해도 실패하면 (거의 불가능) UUID 사용
    import uuid
    fallback_nickname = f"user_{uuid.uuid4().hex[:8]}"
    return fallback_nickname


def generate_nickname_with_fallback(
    db: Session,
    preferred_nickname: Optional[str] = None
) -> str:
    """
    선호하는 닉네임이 있으면 그것 사용, 없으면 자동 생성

    Args:
        db: DB 세션
        preferred_nickname: 선호하는 닉네임 (None이면 자동 생성)

    Returns:
        사용할 닉네임
    """
    # 1️⃣ 선호 닉네임이 있고, DB에 없으면 그것 사용
    if preferred_nickname:
        preferred_nickname = preferred_nickname.strip()
        if preferred_nickname:  # 공백 제거 후 확인
            existing = db.query(Member).filter(
                Member.nickname == preferred_nickname
            ).first()

            if not existing:
                return preferred_nickname

    # 2️⃣ 선호 닉네임 없거나 중복이면 자동 생성
    return generate_unique_nickname(db)
