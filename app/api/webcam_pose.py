from datetime import datetime
from typing import List, Optional

import base64
import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.dependencies import verify_auth
from app.db.session import get_db
from app.models.models import Member, UserPostureProfile, WebcamAlertType, WebcamSession
from app.services.webcam_ai_context import build_ai_context
from app.services.mediapipe_detector import MediaPipePoseDetector
from app.services.webcam_comparator import compare as compare_posture

router = APIRouter()


# ==================== 스키마 ====================

class LandmarkData(BaseModel):
    id: int
    name: str
    x: float
    y: float
    z: float
    visibility: float


class PostureProfileCreateRequest(BaseModel):
    reference_landmarks: List[LandmarkData]
    profile_name: Optional[str] = "기본 자세"
    monitor_label: Optional[str] = None
    display_order: Optional[int] = 1
    description: Optional[str] = None


class PostureProfileResponse(BaseModel):
    profile_id: int
    member_id: int
    profile_name: str
    monitor_label: Optional[str]
    display_order: int
    description: Optional[str]
    reference_landmarks: dict
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PostureProfileUpdateRequest(BaseModel):
    profile_name: Optional[str] = None
    monitor_label: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class WebcamSessionRequest(BaseModel):
    started_at: datetime
    ended_at: datetime
    good_count: int = 0
    warning_count: int = 0
    bad_count: int = 0
    cause_counts: Optional[dict] = None


class WebcamSessionHistoryItem(BaseModel):
    session_id: int
    started_at: datetime
    ended_at: Optional[datetime]
    good_count: int
    warning_count: int
    bad_count: int
    total_count: int
    good_ratio: float
    cause_counts: Optional[dict]

    class Config:
        from_attributes = True


class WebcamHistoryResponse(BaseModel):
    sessions: List[WebcamSessionHistoryItem]
    total: int


class AlertTypeResponse(BaseModel):
    alert_type_id: str
    alert_name: str
    description: Optional[str]

    class Config:
        from_attributes = True


class WebcamAnalyzeRequest(BaseModel):
    image_base64: str
    profile_id: Optional[int] = None  # None이면 활성화된 기준 자세 중 첫 번째 사용


class PointDeviation(BaseModel):
    landmark: str
    deviation: float


class WebcamAnalyzeResponse(BaseModel):
    status: str          # "good" | "warning" | "bad"
    deviation_score: float
    profile_id: int
    profile_name: str
    issues: List[str]
    per_point: dict
    ai_context: dict
    judgement_signature: str
    landmarks: List[LandmarkData]
    frame_width: int
    frame_height: int


_detector: Optional[MediaPipePoseDetector] = None


def _get_detector() -> MediaPipePoseDetector:
    global _detector
    if _detector is None:
        _detector = MediaPipePoseDetector()
    return _detector


# ==================== 엔드포인트 ====================

@router.post("/session", status_code=201)
async def create_webcam_session(
    request: WebcamSessionRequest,
    member: Member = Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """웹캠 세션 분석 결과 저장 (인증 필요)"""
    session = WebcamSession(
        member_id=member.member_id,
        started_at=request.started_at,
        ended_at=request.ended_at,
        good_count=request.good_count,
        warning_count=request.warning_count,
        bad_count=request.bad_count,
        cause_counts=request.cause_counts,
    )
    db.add(session)
    db.commit()
    return {"session_id": session.session_id}


@router.get("/history", response_model=WebcamHistoryResponse)
async def get_webcam_history(
    limit: int = 10,
    member: Member = Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """웹캠 세션 기록 조회 (인증 필요)"""
    total = db.query(WebcamSession).filter(WebcamSession.member_id == member.member_id).count()
    rows = (
        db.query(WebcamSession)
        .filter(WebcamSession.member_id == member.member_id)
        .order_by(WebcamSession.started_at.desc())
        .limit(limit)
        .all()
    )

    sessions = []
    for row in rows:
        total_count = row.good_count + row.warning_count + row.bad_count
        good_ratio = row.good_count / total_count if total_count > 0 else 0.0
        sessions.append(
            WebcamSessionHistoryItem(
                session_id=row.session_id,
                started_at=row.started_at,
                ended_at=row.ended_at,
                good_count=row.good_count,
                warning_count=row.warning_count,
                bad_count=row.bad_count,
                total_count=total_count,
                good_ratio=round(good_ratio, 4),
                cause_counts=row.cause_counts,
            )
        )

    return WebcamHistoryResponse(sessions=sessions, total=total)


@router.get("/alert-types", response_model=List[AlertTypeResponse])
async def get_alert_types(db: Session = Depends(get_db)):
    """자세 알림 유형 목록 조회 (인증 불필요)"""
    return db.query(WebcamAlertType).all()


@router.post("/posture-profile", response_model=PostureProfileResponse, status_code=201)
async def create_posture_profile(
    request: PostureProfileCreateRequest,
    member: Member = Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """기준 자세 등록 (인증 필요)"""
    active_count = (
        db.query(UserPostureProfile)
        .filter(UserPostureProfile.member_id == member.member_id, UserPostureProfile.is_active == True)
        .count()
    )
    if active_count >= 3:
        raise HTTPException(status_code=400, detail="활성화된 기준 자세는 최대 3개까지 등록할 수 있습니다.")

    landmarks_dict = {
        "landmarks": [lm.model_dump() for lm in request.reference_landmarks]
    }

    profile = UserPostureProfile(
        member_id=member.member_id,
        profile_name=request.profile_name,
        monitor_label=request.monitor_label,
        display_order=request.display_order,
        description=request.description,
        reference_landmarks=landmarks_dict,
        is_active=True,
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


@router.get("/posture-profile", response_model=List[PostureProfileResponse])
async def get_posture_profiles(
    member: Member = Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """내 기준 자세 목록 조회 (활성화된 것만, 인증 필요)"""
    profiles = (
        db.query(UserPostureProfile)
        .filter(UserPostureProfile.member_id == member.member_id)
        .order_by(UserPostureProfile.is_active.desc(), UserPostureProfile.display_order, UserPostureProfile.created_at)
        .all()
    )
    return profiles


@router.patch("/posture-profile/{profile_id}", response_model=PostureProfileResponse)
async def update_posture_profile(
    profile_id: int,
    request: PostureProfileUpdateRequest,
    member: Member = Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """기준 자세 수정 (인증 필요)"""
    profile = (
        db.query(UserPostureProfile)
        .filter(
            UserPostureProfile.profile_id == profile_id,
            UserPostureProfile.member_id == member.member_id,
        )
        .first()
    )
    if not profile:
        raise HTTPException(status_code=404, detail="기준 자세를 찾을 수 없습니다.")

    if request.profile_name is not None:
        profile.profile_name = request.profile_name
    if request.monitor_label is not None:
        profile.monitor_label = request.monitor_label
    if request.description is not None:
        profile.description = request.description
    if request.is_active is not None:
        if request.is_active and not profile.is_active:
            active_count = (
                db.query(UserPostureProfile)
                .filter(UserPostureProfile.member_id == member.member_id, UserPostureProfile.is_active == True)
                .count()
            )
            if active_count >= 3:
                raise HTTPException(status_code=400, detail="활성화된 기준 자세는 최대 3개까지 등록할 수 있습니다.")
        profile.is_active = request.is_active

    profile.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(profile)
    return profile


@router.delete("/posture-profile/{profile_id}", status_code=204)
async def delete_posture_profile(
    profile_id: int,
    member: Member = Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """기준 자세 삭제 (인증 필요)"""
    profile = (
        db.query(UserPostureProfile)
        .filter(
            UserPostureProfile.profile_id == profile_id,
            UserPostureProfile.member_id == member.member_id,
        )
        .first()
    )
    if not profile:
        raise HTTPException(status_code=404, detail="기준 자세를 찾을 수 없습니다.")

    db.delete(profile)
    db.commit()


@router.post("/analyze", response_model=WebcamAnalyzeResponse)
async def analyze_webcam(
    request: WebcamAnalyzeRequest,
    member: Member = Depends(verify_auth),
    db: Session = Depends(get_db),
):
    """웹캠 프레임을 기준 자세와 비교 분석 (인증 필요)"""
    # 기준 자세 프로필 조회
    query = db.query(UserPostureProfile).filter(
        UserPostureProfile.member_id == member.member_id,
        UserPostureProfile.is_active == True,
    )
    if request.profile_id:
        query = query.filter(UserPostureProfile.profile_id == request.profile_id)

    profiles = query.order_by(UserPostureProfile.display_order).all()
    if not profiles:
        raise HTTPException(status_code=404, detail="활성화된 기준 자세가 없습니다. 기준 자세를 먼저 등록해주세요.")

    # 이미지 디코딩
    try:
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 디코딩에 실패했습니다")

    if frame is None:
        raise HTTPException(status_code=400, detail="유효한 이미지가 아닙니다")

    # 현재 자세 감지
    pose_result = _get_detector().detect_pose(frame)
    if not pose_result.is_detected:
        raise HTTPException(status_code=422, detail="자세를 감지할 수 없습니다. 카메라를 조정해주세요.")

    # 모든 활성 프로필과 비교 → deviation_score가 가장 낮은 프로필 선택
    # (듀얼모니터 등 여러 기준 자세 중 현재 자세와 가장 가까운 것을 자동 매칭)
    best_profile = profiles[0]
    best_result = compare_posture(
        pose_result.landmarks,
        profiles[0].reference_landmarks.get("landmarks", []),
    )
    for p in profiles[1:]:
        candidate = compare_posture(
            pose_result.landmarks,
            p.reference_landmarks.get("landmarks", []),
        )
        if candidate.deviation_score < best_result.deviation_score:
            best_profile = p
            best_result = candidate

    profile = best_profile
    result = best_result

    landmarks_out = [
        LandmarkData(
            id=i,
            name=lm.name if hasattr(lm, "name") else str(i),
            x=lm.x,
            y=lm.y,
            z=lm.z,
            visibility=lm.visibility,
        )
        for i, lm in enumerate(pose_result.landmarks)
    ]

    ai_context = build_ai_context(result)

    return WebcamAnalyzeResponse(
        status=result.status,
        deviation_score=result.deviation_score,
        profile_id=profile.profile_id,
        profile_name=profile.profile_name,
        issues=result.issues,
        per_point=result.per_point,
        ai_context=ai_context,
        judgement_signature=ai_context["judgement_signature"],
        landmarks=landmarks_out,
        frame_width=frame.shape[1],
        frame_height=frame.shape[0],
    )
