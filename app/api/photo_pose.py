from datetime import datetime
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.dependencies import verify_auth
from app.core.config import settings
from app.db.session import get_db
from app.models.models import Member, PoseAnalysis
from app.services.analysis_token_service import AnalysisTokenService
from app.services.mediapipe_detector import Landmark, MediaPipePoseDetector, PoseDetectionResult
from app.services.photo_posture_analyzer import PhotoPostureAnalyzer, SideView


router = APIRouter()


class PhotoGuideResponse(BaseModel):
    minimum_photo_count: int
    required_views: list[str]
    side_view_options: list[str]
    images_stored: bool
    capture_tips: list[str]
    fallback_tips: list[str]


class FrontMetricsResponse(BaseModel):
    confidence: float
    shoulder_slope: float | None
    hip_slope: float | None
    spine_alignment: float | None
    asymmetry_score: float | None


class SideMetricsResponse(BaseModel):
    confidence: float
    neck_forward_angle: float | None
    forward_head_detected: bool | None


class LandmarkResponse(BaseModel):
    id: int
    name: str
    x: float
    y: float
    z: float
    visibility: float


class AnalyzePhotosResponse(BaseModel):
    status: Literal["good", "warning", "bad", "action_required"]
    analysis_mode: Literal["full", "upper_body_only", "manual_adjustment_required"]
    confidence: float
    analyzed_at: str
    side_view: Literal["left", "right"]
    images_stored: bool
    can_save: bool
    alerts: list[str]
    missing_landmarks: list[str]
    available_actions: list[str]
    front: FrontMetricsResponse
    side: SideMetricsResponse
    front_landmarks: list[LandmarkResponse]
    side_landmarks: list[LandmarkResponse]
    issues: list[str]
    save_token: str | None


class SaveAnalysisRequest(BaseModel):
    save_token: str


class SaveAnalysisResponse(BaseModel):
    analysis_id: int
    saved_at: str
    status: Literal["good", "warning", "bad"]
    images_stored: bool


class ManualLandmarkInput(BaseModel):
    id: int = Field(..., ge=0, le=32)
    x: float
    y: float
    z: float = 0.0
    visibility: float = Field(default=1.0, ge=0.0, le=1.0)
    name: str | None = None


class ManualLandmarkAnalyzeRequest(BaseModel):
    side_view: SideView
    front_landmarks: list[ManualLandmarkInput]
    side_landmarks: list[ManualLandmarkInput]


detector: MediaPipePoseDetector | None = None
photo_analyzer: PhotoPostureAnalyzer | None = None
analysis_token_service: AnalysisTokenService | None = None


def init_services() -> None:
    global detector, photo_analyzer, analysis_token_service

    if detector is None:
        detector = MediaPipePoseDetector(running_mode="image")
    if photo_analyzer is None:
        photo_analyzer = PhotoPostureAnalyzer()
    if analysis_token_service is None:
        analysis_token_service = AnalysisTokenService(settings.secret_key)


@router.get("/photo-guide", response_model=PhotoGuideResponse)
async def photo_guide() -> PhotoGuideResponse:
    return PhotoGuideResponse(
        minimum_photo_count=2,
        required_views=["front", "side"],
        side_view_options=["left", "right"],
        images_stored=False,
        capture_tips=[
            "정면 전신 사진 1장과 측면 전신 사진 1장을 업로드하세요.",
            "머리, 어깨, 골반이 모두 보이도록 촬영하면 가장 정확합니다.",
            "카메라는 가슴 높이 정도에 두고 몸이 프레임 중앙에 오도록 맞춰 주세요.",
            "헐렁한 옷이나 강한 역광은 랜드마크 인식을 어렵게 만들 수 있습니다.",
        ],
        fallback_tips=[
            "골반이 잘 보이지 않으면 목과 어깨 중심의 부분 분석만 진행됩니다.",
            "머리나 어깨가 잘 보이지 않으면 수동 랜드마크 조정 또는 사진 재업로드가 필요합니다.",
        ],
    )


@router.post("/analyze-photos", response_model=AnalyzePhotosResponse)
async def analyze_photos(
    front_image: UploadFile = File(...),
    side_image: UploadFile = File(...),
    side_view: SideView = Form(...),
    member: Member = Depends(verify_auth),
) -> AnalyzePhotosResponse:
    init_services()

    try:
        front_frame = await _decode_upload_to_frame(front_image, "front_image")
        side_frame = await _decode_upload_to_frame(side_image, "side_image")
        front_result = detector.detect_pose(front_frame)
        side_result = detector.detect_pose(side_frame)
        analysis = photo_analyzer.analyze(front_result, side_result, side_view)
    finally:
        await front_image.close()
        await side_image.close()

    return _build_analysis_response(member.member_id, analysis, front_result, side_result)


@router.post("/analyze-manual-landmarks", response_model=AnalyzePhotosResponse)
async def analyze_manual_landmarks(
    request: ManualLandmarkAnalyzeRequest,
    member: Member = Depends(verify_auth),
) -> AnalyzePhotosResponse:
    init_services()

    front_result = _manual_landmarks_to_result(request.front_landmarks)
    side_result = _manual_landmarks_to_result(request.side_landmarks)
    analysis = photo_analyzer.analyze(front_result, side_result, request.side_view)

    return _build_analysis_response(member.member_id, analysis, front_result, side_result)


@router.post("/analyses", response_model=SaveAnalysisResponse)
async def save_analysis(
    request: SaveAnalysisRequest,
    member: Member = Depends(verify_auth),
    db: Session = Depends(get_db),
) -> SaveAnalysisResponse:
    init_services()

    try:
        payload = analysis_token_service.loads(request.save_token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if payload.get("member_id") != member.member_id:
        raise HTTPException(status_code=403, detail="This analysis token belongs to a different member.")

    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        raise HTTPException(status_code=400, detail="Invalid analysis payload.")

    if not analysis.get("can_save"):
        raise HTTPException(status_code=400, detail="This analysis is incomplete and cannot be saved yet.")

    front = analysis.get("front") or {}
    side = analysis.get("side") or {}
    analyzed_at = _parse_iso_datetime(analysis.get("analyzed_at"))

    record = PoseAnalysis(
        member_id=member.member_id,
        side_view=analysis["side_view"],
        overall_status=analysis["status"],
        overall_confidence=analysis["confidence"],
        front_confidence=front["confidence"],
        side_confidence=side["confidence"],
        neck_forward_angle=side["neck_forward_angle"],
        shoulder_slope=front["shoulder_slope"],
        hip_slope=front.get("hip_slope"),
        spine_alignment=front.get("spine_alignment"),
        asymmetry_score=front.get("asymmetry_score"),
        forward_head_detected=side["forward_head_detected"],
        analyzed_at=analyzed_at,
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return SaveAnalysisResponse(
        analysis_id=record.analysis_id,
        saved_at=record.created_at.isoformat(),
        status=record.overall_status,
        images_stored=False,
    )


def _build_analysis_response(
    member_id: int,
    analysis,
    front_result: PoseDetectionResult,
    side_result: PoseDetectionResult,
) -> AnalyzePhotosResponse:
    analysis_payload = analysis.to_dict()
    save_token = None

    if analysis.can_save:
        save_token = analysis_token_service.dumps(
            {
                "member_id": member_id,
                "analysis": analysis_payload,
                "version": 3,
            }
        )

    return AnalyzePhotosResponse(
        **analysis_payload,
        images_stored=False,
        front_landmarks=_landmarks_to_response(front_result.landmarks),
        side_landmarks=_landmarks_to_response(side_result.landmarks),
        save_token=save_token,
    )


def _landmarks_to_response(landmarks: list[Landmark]) -> list[LandmarkResponse]:
    return [
        LandmarkResponse(
            id=landmark.id,
            name=landmark.name,
            x=landmark.x,
            y=landmark.y,
            z=landmark.z,
            visibility=landmark.visibility,
        )
        for landmark in landmarks
    ]


def _manual_landmarks_to_result(landmarks: list[ManualLandmarkInput]) -> PoseDetectionResult:
    if not landmarks:
        return PoseDetectionResult(
            landmarks=[],
            confidence=0.0,
            frame_width=0,
            frame_height=0,
            timestamp=datetime.utcnow().isoformat() + "Z",
            is_detected=False,
        )

    max_index = max(24, max(landmark.id for landmark in landmarks))
    normalized_landmarks = [
        Landmark(
            id=index,
            name=f"landmark_{index}",
            x=0.0,
            y=0.0,
            z=0.0,
            visibility=0.0,
        )
        for index in range(max_index + 1)
    ]

    for landmark in landmarks:
        normalized_landmarks[landmark.id] = Landmark(
            id=landmark.id,
            name=landmark.name or f"landmark_{landmark.id}",
            x=landmark.x,
            y=landmark.y,
            z=landmark.z,
            visibility=landmark.visibility,
        )

    confidence = round(
        sum(landmark.visibility for landmark in normalized_landmarks) / len(normalized_landmarks),
        3,
    )
    return PoseDetectionResult(
        landmarks=normalized_landmarks,
        confidence=confidence,
        frame_width=0,
        frame_height=0,
        timestamp=datetime.utcnow().isoformat() + "Z",
        is_detected=True,
    )


def _parse_iso_datetime(value: str | None) -> datetime:
    if not value:
        raise HTTPException(status_code=400, detail="Invalid analyzed_at value.")
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


async def _decode_upload_to_frame(upload: UploadFile, field_name: str) -> np.ndarray:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"{field_name} must be an image file.")

    image_bytes = await upload.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty.")

    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail=f"{field_name} could not be decoded.")

    return frame
