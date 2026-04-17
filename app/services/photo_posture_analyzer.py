from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np

from app.services.mediapipe_detector import Landmark, PoseDetectionResult
from app.services.pose_analyzer import PoseAnalyzer


class AnalysisStatus(str, Enum):
    GOOD = "good"
    WARNING = "warning"
    BAD = "bad"
    ACTION_REQUIRED = "action_required"


class AnalysisMode(str, Enum):
    FULL = "full"
    UPPER_BODY_ONLY = "upper_body_only"
    MANUAL_ADJUSTMENT_REQUIRED = "manual_adjustment_required"


class SideView(str, Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass
class FrontViewMetrics:
    confidence: float
    shoulder_slope: float | None
    hip_slope: float | None
    spine_alignment: float | None
    asymmetry_score: float | None


@dataclass
class SideViewMetrics:
    confidence: float
    neck_forward_angle: float | None
    forward_head_detected: bool | None


@dataclass
class PhotoPostureAnalysis:
    status: AnalysisStatus
    analysis_mode: AnalysisMode
    confidence: float
    analyzed_at: str
    side_view: str
    front: FrontViewMetrics
    side: SideViewMetrics
    issues: list[str]
    alerts: list[str]
    missing_landmarks: list[str]
    available_actions: list[str]
    can_save: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["analysis_mode"] = self.analysis_mode.value
        return data


class PhotoPostureAnalyzer:
    MIN_VIEW_CONFIDENCE = 0.25
    HIP_SLOPE_THRESHOLD = 7.0
    ASYMMETRY_THRESHOLD = 5.0
    SPINE_ALIGNMENT_THRESHOLD = 0.75
    FORWARD_HEAD_THRESHOLD = 15.0
    MIN_LANDMARK_VISIBILITY = 0.2

    HIP_ALERT_MESSAGE = "사진에 골반이 제대로 보이지 않으면 목과 어깨까지의 정보만 분석된다"
    ESSENTIAL_ALERT_MESSAGE = "사진에서 머리와 골반이 명확하지 않다"

    def __init__(self) -> None:
        self.pose_analyzer = PoseAnalyzer()

    def analyze(
        self,
        front_result: PoseDetectionResult,
        side_result: PoseDetectionResult,
        side_view: SideView,
    ) -> PhotoPostureAnalysis:
        analyzed_at = front_result.timestamp
        confidence = round((front_result.confidence + side_result.confidence) / 2, 3)

        front_metrics = FrontViewMetrics(
            confidence=round(front_result.confidence, 3),
            shoulder_slope=None,
            hip_slope=None,
            spine_alignment=None,
            asymmetry_score=None,
        )
        side_metrics = SideViewMetrics(
            confidence=round(side_result.confidence, 3),
            neck_forward_angle=None,
            forward_head_detected=None,
        )

        required_missing = self._find_required_missing_landmarks(front_result, side_result, side_view)
        if required_missing:
            return PhotoPostureAnalysis(
                status=AnalysisStatus.ACTION_REQUIRED,
                analysis_mode=AnalysisMode.MANUAL_ADJUSTMENT_REQUIRED,
                confidence=confidence,
                analyzed_at=analyzed_at,
                side_view=side_view.value,
                front=front_metrics,
                side=side_metrics,
                issues=[],
                alerts=[self.ESSENTIAL_ALERT_MESSAGE],
                missing_landmarks=required_missing,
                available_actions=["manual_adjust", "reupload"],
                can_save=False,
            )

        front_metrics.shoulder_slope = self.pose_analyzer.calculate_shoulder_slope(front_result.landmarks)
        side_metrics.neck_forward_angle = self._calculate_side_neck_angle(side_result.landmarks, side_view)
        side_metrics.forward_head_detected = (
            side_metrics.neck_forward_angle > self.FORWARD_HEAD_THRESHOLD
            if side_metrics.neck_forward_angle is not None
            else None
        )

        hips_available = self._has_visible(front_result.landmarks, self.pose_analyzer.LEFT_HIP) and self._has_visible(
            front_result.landmarks, self.pose_analyzer.RIGHT_HIP
        )

        alerts: list[str] = []
        issues: list[str] = []
        analysis_mode = AnalysisMode.FULL

        if side_metrics.forward_head_detected:
            issues.append(f"거북목 경향 ({side_metrics.neck_forward_angle:.1f}°)")

        if (
            front_metrics.shoulder_slope is not None
            and front_metrics.shoulder_slope > self.pose_analyzer.SHOULDER_SLOPE_THRESHOLD
        ):
            issues.append(f"어깨 기울기 ({front_metrics.shoulder_slope:.1f}°)")

        if hips_available:
            front_metrics.hip_slope = self._calculate_hip_slope(front_result.landmarks)
            front_metrics.spine_alignment = self.pose_analyzer.calculate_spine_alignment(front_result.landmarks)
            front_metrics.asymmetry_score = self._calculate_asymmetry_score(front_result.landmarks)

            if front_metrics.hip_slope > self.HIP_SLOPE_THRESHOLD:
                issues.append(f"골반 기울기 ({front_metrics.hip_slope:.1f}°)")

            if (
                front_metrics.asymmetry_score is not None
                and front_metrics.asymmetry_score > self.ASYMMETRY_THRESHOLD
            ):
                issues.append(f"좌우 비대칭 ({front_metrics.asymmetry_score:.1f}%)")

            if (
                front_metrics.spine_alignment is not None
                and front_metrics.spine_alignment < self.SPINE_ALIGNMENT_THRESHOLD
            ):
                issues.append(f"척추 정렬 주의 (정렬도 {front_metrics.spine_alignment:.2f})")
        else:
            analysis_mode = AnalysisMode.UPPER_BODY_ONLY
            alerts.append(self.HIP_ALERT_MESSAGE)

        if len(issues) == 0:
            status = AnalysisStatus.GOOD
        elif len(issues) <= 2:
            status = AnalysisStatus.WARNING
        else:
            status = AnalysisStatus.BAD

        return PhotoPostureAnalysis(
            status=status,
            analysis_mode=analysis_mode,
            confidence=confidence,
            analyzed_at=analyzed_at,
            side_view=side_view.value,
            front=front_metrics,
            side=side_metrics,
            issues=issues,
            alerts=alerts,
            missing_landmarks=[],
            available_actions=["save", "reupload"],
            can_save=True,
        )

    def _find_required_missing_landmarks(
        self,
        front_result: PoseDetectionResult,
        side_result: PoseDetectionResult,
        side_view: SideView,
    ) -> list[str]:
        missing: list[str] = []

        if not front_result.is_detected or front_result.confidence < self.MIN_VIEW_CONFIDENCE:
            missing.extend(["front_left_shoulder", "front_right_shoulder"])
        else:
            if not self._has_visible(front_result.landmarks, self.pose_analyzer.LEFT_SHOULDER):
                missing.append("front_left_shoulder")
            if not self._has_visible(front_result.landmarks, self.pose_analyzer.RIGHT_SHOULDER):
                missing.append("front_right_shoulder")

        side_ear_index = self.pose_analyzer.LEFT_EAR if side_view == SideView.LEFT else self.pose_analyzer.RIGHT_EAR
        side_shoulder_index = (
            self.pose_analyzer.LEFT_SHOULDER if side_view == SideView.LEFT else self.pose_analyzer.RIGHT_SHOULDER
        )

        if not side_result.is_detected or side_result.confidence < self.MIN_VIEW_CONFIDENCE:
            missing.extend([f"{side_view.value}_ear", f"{side_view.value}_shoulder"])
        else:
            if not self._has_visible(side_result.landmarks, side_ear_index):
                missing.append(f"{side_view.value}_ear")
            if not self._has_visible(side_result.landmarks, side_shoulder_index):
                missing.append(f"{side_view.value}_shoulder")

        return missing

    def _calculate_hip_slope(self, landmarks: list[Landmark]) -> float:
        left_hip = self._landmark(landmarks, self.pose_analyzer.LEFT_HIP)
        right_hip = self._landmark(landmarks, self.pose_analyzer.RIGHT_HIP)
        return self._line_angle(left_hip, right_hip)

    def _calculate_asymmetry_score(self, landmarks: list[Landmark]) -> float:
        left_shoulder = self._landmark(landmarks, self.pose_analyzer.LEFT_SHOULDER)
        right_shoulder = self._landmark(landmarks, self.pose_analyzer.RIGHT_SHOULDER)
        left_hip = self._landmark(landmarks, self.pose_analyzer.LEFT_HIP)
        right_hip = self._landmark(landmarks, self.pose_analyzer.RIGHT_HIP)

        shoulder_ratio = self._vertical_ratio(left_shoulder, right_shoulder)
        hip_ratio = self._vertical_ratio(left_hip, right_hip)
        return round(((shoulder_ratio + hip_ratio) / 2) * 100, 2)

    def _calculate_side_neck_angle(self, landmarks: list[Landmark], side_view: SideView) -> float:
        if side_view == SideView.LEFT:
            ear_index = self.pose_analyzer.LEFT_EAR
            shoulder_index = self.pose_analyzer.LEFT_SHOULDER
        else:
            ear_index = self.pose_analyzer.RIGHT_EAR
            shoulder_index = self.pose_analyzer.RIGHT_SHOULDER

        ear = self._landmark(landmarks, ear_index)
        shoulder = self._landmark(landmarks, shoulder_index)

        dx = abs(ear.x - shoulder.x)
        dy = abs(ear.y - shoulder.y) + 1e-6
        return round(float(np.degrees(np.arctan2(dx, dy))), 2)

    def _has_visible(self, landmarks: list[Landmark], index: int) -> bool:
        if len(landmarks) <= index:
            return False
        return landmarks[index].visibility >= self.MIN_LANDMARK_VISIBILITY

    @staticmethod
    def _line_angle(left: Landmark, right: Landmark) -> float:
        dx = right.x - left.x
        dy = right.y - left.y
        slope_angle = np.degrees(np.arctan2(dy, dx))
        return round(float(min(abs(slope_angle), 180 - abs(slope_angle))), 2)

    @staticmethod
    def _vertical_ratio(left: Landmark, right: Landmark) -> float:
        width = abs(right.x - left.x) + 1e-6
        return abs(right.y - left.y) / width

    @staticmethod
    def _landmark(landmarks: list[Landmark], index: int) -> Landmark:
        if len(landmarks) <= index:
            return Landmark(id=index, name=f"landmark_{index}", x=0.0, y=0.0, z=0.0, visibility=0.0)
        return landmarks[index]
