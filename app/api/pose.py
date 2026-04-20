"""
자세 분석 API 엔드포인트

웹캠 자세 감지 및 분석 관련 API
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import cv2
import base64
import numpy as np
from sqlalchemy.orm import Session

from app.api.dependencies import verify_auth
from app.services.mediapipe_detector import MediaPipePoseDetector, PoseDetectionResult
from app.services.pose_analyzer import PoseAnalyzer, PostureAnalysisResult
from app.db.session import get_db
from app.models.models import Member


router = APIRouter()


class LandmarkResponse(BaseModel):
    """랜드마크 응답"""
    id: int
    name: str
    x: float
    y: float
    z: float
    visibility: float


class PostureAnalysisResponse(BaseModel):
    """자세 분석 응답"""
    status: str  # good, warning, bad
    confidence: float
    landmarks: List[LandmarkResponse]
    frame_width: int
    frame_height: int
    neck_forward_angle: float
    shoulder_slope: float
    spine_alignment: float
    issues: List[str]
    recommendations: List[str]


class PoseDetectionRequest(BaseModel):
    """포즈 감지 요청"""
    image_base64: Optional[str] = None  # Base64 인코딩된 이미지


# 전역 변수 (간단한 구현용)
detector = None
analyzer = None


def init_services():
    """서비스 초기화"""
    global detector, analyzer
    if detector is None:
        detector = MediaPipePoseDetector()
    if analyzer is None:
        analyzer = PoseAnalyzer()


@router.post("/analyze")
async def analyze_posture(request: PoseDetectionRequest, member: Member = Depends(verify_auth)):
    """
    웹캠 프레임에서 자세를 분석합니다. (인증 필요)

    Args:
        request: Base64 인코딩된 이미지
        member: 인증된 회원 정보 (Authorization 헤더에서 자동 검증)

    Returns:
        PostureAnalysisResponse: 자세 분석 결과
    """
    init_services()

    try:
        if not request.image_base64:
            raise HTTPException(status_code=400, detail="image_base64가 필요합니다")

        # Base64 디코딩
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="유효한 이미지가 아닙니다")

        # 포즈 감지
        pose_result = detector.detect_pose(frame)

        # 랜드마크를 Response 형식으로 변환
        landmarks_response = [
            LandmarkResponse(
                id=lm.id,
                name=lm.name,
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility
            )
            for lm in pose_result.landmarks
        ]

        # 자세 분석
        if not pose_result.is_detected:
            return PostureAnalysisResponse(
                status="warning",
                confidence=0.0,
                landmarks=landmarks_response,
                frame_width=pose_result.frame_width,
                frame_height=pose_result.frame_height,
                neck_forward_angle=0.0,
                shoulder_slope=0.0,
                spine_alignment=0.0,
                issues=["자세를 감지할 수 없습니다"],
                recommendations=["카메라를 조정하거나 자세를 명확히 해주세요"]
            )

        analysis = analyzer.analyze(pose_result.landmarks, pose_result.confidence)

        return PostureAnalysisResponse(
            status=analysis.status.value,
            confidence=analysis.confidence,
            landmarks=landmarks_response,
            frame_width=pose_result.frame_width,
            frame_height=pose_result.frame_height,
            neck_forward_angle=analysis.neck_forward_angle,
            shoulder_slope=analysis.shoulder_slope,
            spine_alignment=analysis.spine_alignment,
            issues=analysis.issues,
            recommendations=analysis.recommendations
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/landmarks")
async def get_landmarks(image_base64: Optional[str] = None, member: Member = Depends(verify_auth)):
    """
    포즈 랜드마크를 반환합니다. (인증 필요)

    Args:
        image_base64: Base64 인코딩된 이미지
        member: 인증된 회원 정보 (Authorization 헤더에서 자동 검증)

    Returns:
        dict: 랜드마크 데이터
    """
    init_services()

    try:
        if not image_base64:
            raise HTTPException(status_code=400, detail="image_base64가 필요합니다")

        # Base64 디코딩
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="유효한 이미지가 아닙니다")

        # 포즈 감지
        pose_result = detector.detect_pose(frame)

        return detector.to_dict(pose_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    """헬스 체크"""
    return {"status": "ok"}
