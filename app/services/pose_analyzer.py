"""
자세 분석 서비스

MediaPipe 랜드마크를 분석하여 거북목, 자세 이상 등을 감지합니다.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PostureStatus(str, Enum):
    """자세 상태"""
    GOOD = "good"  # 좋은 자세
    WARNING = "warning"  # 주의
    BAD = "bad"  # 나쁜 자세


@dataclass
class PostureAnalysisResult:
    """자세 분석 결과"""
    status: PostureStatus
    confidence: float
    neck_forward_angle: float  # 목 앞으로 구부러진 각도 (도)
    shoulder_slope: float  # 어깨 기울기 (도)
    spine_alignment: float  # 척추 정렬도 (0~1, 1이 완벽)
    issues: List[str]  # 감지된 문제점
    recommendations: List[str]  # 개선 권장사항


class PoseAnalyzer:
    """
    포즈 분석기

    MediaPipe 랜드마크를 기반으로 자세를 분석합니다.
    """

    # 랜드마크 인덱스 (MediaPipe Pose)
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_EAR = 7
    RIGHT_EAR = 8

    # 임계값
    NECK_FORWARD_THRESHOLD = 20  # 도 (거북목 판정 기준)
    SHOULDER_SLOPE_THRESHOLD = 10  # 도 (어깨 기울기 기준)
    CONFIDENCE_THRESHOLD = 0.3  # 신뢰도 기준

    def __init__(self):
        """PoseAnalyzer 초기화"""
        pass

    @staticmethod
    def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """두 점 사이의 거리 계산"""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def calculate_angle(p1: Tuple[float, float],
                       p2: Tuple[float, float],
                       p3: Tuple[float, float]) -> float:
        """
        세 점으로 이루어진 각도 계산 (p2를 중심)

        Args:
            p1, p2, p3: (x, y) 튜플

        Returns:
            각도 (도)
        """
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        # cos 값을 [-1, 1] 범위로 clamp
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)

    def calculate_neck_forward_angle(self, landmarks: List) -> float:
        """
        목이 앞으로 구부러진 각도 계산

        머리(코)와 어깨의 상대 위치로 판단
        0도에 가까울수록 좋은 자세

        Args:
            landmarks: Landmark 객체 리스트

        Returns:
            각도 (도, 작을수록 좋음)
        """
        if len(landmarks) < max(self.NOSE, self.LEFT_SHOULDER, self.RIGHT_SHOULDER) + 1:
            return 0.0

        nose = (landmarks[self.NOSE].x, landmarks[self.NOSE].y)
        left_shoulder = (landmarks[self.LEFT_SHOULDER].x, landmarks[self.LEFT_SHOULDER].y)
        right_shoulder = (landmarks[self.RIGHT_SHOULDER].x, landmarks[self.RIGHT_SHOULDER].y)

        # 어깨 중심점
        shoulder_center = (
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2
        )

        # 코에서 어깨 중심으로의 벡터
        dx = shoulder_center[0] - nose[0]
        dy = shoulder_center[1] - nose[1]

        # 수평선으로부터의 각도 (-90~90도)
        # 0도 = 정확히 아래쪽, 음수 = 왼쪽으로 기울어짐, 양수 = 오른쪽으로 기울어짐
        angle_from_vertical = np.degrees(np.arctan2(dx, dy))

        # 절댓값 (좌우 대칭이므로)
        neck_forward = abs(angle_from_vertical)

        return round(neck_forward, 2)

    def calculate_shoulder_slope(self, landmarks: List) -> float:
        """
        어깨 기울기 계산

        좌우 어깨의 높이 차이로 판단
        0도에 가까울수록 좋은 자세

        Args:
            landmarks: Landmark 객체 리스트

        Returns:
            각도 (도, 0~90, 0이 가장 좋음)
        """
        if len(landmarks) < max(self.LEFT_SHOULDER, self.RIGHT_SHOULDER) + 1:
            return 0.0

        left_shoulder = (landmarks[self.LEFT_SHOULDER].x, landmarks[self.LEFT_SHOULDER].y)
        right_shoulder = (landmarks[self.RIGHT_SHOULDER].x, landmarks[self.RIGHT_SHOULDER].y)

        # 어깨를 잇는 직선의 기울기
        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]

        # 수평선으로부터의 각도 (-90~90도)
        slope_angle = np.degrees(np.arctan2(dy, dx))

        # 0도에서 벗어난 정도 (정규화: 0~90도)
        # 양수든 음수든 절댓값을 취하되, 180도 근처는 다시 0도로
        slope = min(abs(slope_angle), 180 - abs(slope_angle))

        return round(slope, 2)

    def calculate_spine_alignment(self, landmarks: List) -> float:
        """
        척추 정렬도 계산 (0~1, 1이 완벽)

        어깨-허리의 정렬 상태로 판단

        Args:
            landmarks: Landmark 객체 리스트

        Returns:
            정렬도 (0~1)
        """
        if len(landmarks) < max(self.LEFT_HIP, self.RIGHT_HIP,
                               self.LEFT_SHOULDER, self.RIGHT_SHOULDER) + 1:
            return 0.0

        # 어깨 중심
        left_shoulder = (landmarks[self.LEFT_SHOULDER].x, landmarks[self.LEFT_SHOULDER].y)
        right_shoulder = (landmarks[self.RIGHT_SHOULDER].x, landmarks[self.RIGHT_SHOULDER].y)
        shoulder_center = (
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2
        )

        # 허리 중심
        left_hip = (landmarks[self.LEFT_HIP].x, landmarks[self.LEFT_HIP].y)
        right_hip = (landmarks[self.RIGHT_HIP].x, landmarks[self.RIGHT_HIP].y)
        hip_center = (
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2
        )

        # 어깨-허리 중심의 x 좌표 차이 (클수록 기울어짐)
        x_diff = abs(shoulder_center[0] - hip_center[0])

        # 어깨-허리 거리
        distance = self.calculate_distance(shoulder_center, hip_center)

        # 정렬도 계산 (차이가 작을수록 높음)
        alignment = max(0, 1 - (x_diff / (distance + 1e-6)))

        return round(alignment, 3)

    def analyze(self, landmarks: List, overall_confidence: float) -> PostureAnalysisResult:
        """
        전체 자세 분석

        Args:
            landmarks: Landmark 객체 리스트
            overall_confidence: 전체 감지 신뢰도 (0~1)

        Returns:
            PostureAnalysisResult: 분석 결과
        """
        issues = []
        recommendations = []

        # 신뢰도 체크
        if overall_confidence < self.CONFIDENCE_THRESHOLD:
            return PostureAnalysisResult(
                status=PostureStatus.WARNING,
                confidence=overall_confidence,
                neck_forward_angle=0.0,
                shoulder_slope=0.0,
                spine_alignment=0.0,
                issues=["자세 감지 신뢰도가 낮습니다"],
                recommendations=["카메라를 조정하거나 밝기를 확인하세요"]
            )

        # 각 지표 계산
        neck_angle = self.calculate_neck_forward_angle(landmarks)
        shoulder_slope = self.calculate_shoulder_slope(landmarks)
        spine_alignment = self.calculate_spine_alignment(landmarks)

        # 문제점 판단
        if neck_angle > self.NECK_FORWARD_THRESHOLD:
            issues.append(f"거북목 경향 ({neck_angle:.1f}°)")
            recommendations.append("목을 뒤로 당기고 턱을 안으로 집어넣으세요")

        if shoulder_slope > self.SHOULDER_SLOPE_THRESHOLD:
            issues.append(f"어깨 기울기 ({shoulder_slope:.1f}°)")
            recommendations.append("양쪽 어깨의 높이를 맞추세요")

        if spine_alignment < 0.7:
            issues.append(f"척추 정렬 문제 (정렬도: {spine_alignment:.2f})")
            recommendations.append("허리를 펴고 어깨를 펴세요")

        # 상태 판정
        if len(issues) == 0:
            status = PostureStatus.GOOD
        elif len(issues) == 1:
            status = PostureStatus.WARNING
        else:
            status = PostureStatus.BAD

        return PostureAnalysisResult(
            status=status,
            confidence=round(overall_confidence, 3),
            neck_forward_angle=neck_angle,
            shoulder_slope=shoulder_slope,
            spine_alignment=spine_alignment,
            issues=issues,
            recommendations=recommendations
        )

    def to_dict(self, result: PostureAnalysisResult) -> Dict:
        """분석 결과를 딕셔너리로 변환"""
        return {
            "status": result.status.value,
            "confidence": result.confidence,
            "neck_forward_angle": result.neck_forward_angle,
            "shoulder_slope": result.shoulder_slope,
            "spine_alignment": result.spine_alignment,
            "issues": result.issues,
            "recommendations": result.recommendations
        }