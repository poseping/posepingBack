"""
MediaPipe Pose Detection Service (새로운 Task API)

웹캠에서 실시간 자세를 감지하고 정규화된 랜드마크 좌표를 반환합니다.
"""

import cv2
import numpy as np
import os
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from tempfile import gettempdir
from urllib.request import urlopen
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from app.services.pose_analyzer import PoseAnalyzer


@dataclass
class Landmark:
    """MediaPipe 랜드마크 데이터"""
    id: int
    name: str
    x: float  # 정규화된 x 좌표 (0.0-1.0)
    y: float  # 정규화된 y 좌표 (0.0-1.0)
    z: float  # 깊이 값
    visibility: float  # 감지 신뢰도 (0.0-1.0)


@dataclass
class PoseDetectionResult:
    """포즈 감지 결과"""
    landmarks: List[Landmark]
    confidence: float
    frame_width: int
    frame_height: int
    timestamp: str
    is_detected: bool


class MediaPipePoseDetector:
    """
    MediaPipe Task API를 사용한 포즈 감지기

    17개의 신체 포인트(nose, shoulders, elbows, wrists)를 감지합니다.
    """

    MODEL_ASSET_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )
    MODEL_CACHE_DIR = Path(gettempdir()) / "poseping-models"

    LANDMARK_NAMES = [
        "nose",
        "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear",
        "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist"
    ]

    def __init__(self, min_pose_detection_confidence: float = 0.5,
                 min_pose_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        MediaPipePoseDetector 초기화 (Task API 사용)

        Args:
            min_pose_detection_confidence: 최소 감지 신뢰도
            min_pose_presence_confidence: 최소 포즈 존재 신뢰도
            min_tracking_confidence: 최소 추적 신뢰도
        """
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = vision.PoseLandmarker
        PoseLandmarkerOptions = vision.PoseLandmarkerOptions
        VisionRunningMode = vision.RunningMode
        model_asset_path = self._resolve_model_asset_path()

        # ⭐ 모델 파일 경로 지정
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_asset_path)),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.detector = PoseLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    @classmethod
    def ensure_model_asset(cls) -> Path:
        return cls._resolve_model_asset_path()

    @classmethod
    def _resolve_model_asset_path(cls) -> Path:
        env_model_asset_path = os.getenv("POSE_LANDMARKER_PATH")

        if env_model_asset_path:
            model_asset_path = Path(env_model_asset_path).expanduser().resolve()
            if not model_asset_path.exists():
                raise FileNotFoundError(
                    f"POSE_LANDMARKER_PATH 파일을 찾을 수 없습니다: {model_asset_path}"
                )
            return model_asset_path

        model_asset_candidates = [
            Path(__file__).resolve().parents[1] / "pose_landmarker_lite.task",
            Path(__file__).resolve().with_name("pose_landmarker_lite.task"),
            Path(__file__).resolve().parents[2] / "pose_landmarker_lite.task",
            Path.cwd() / "pose_landmarker_lite.task",
            cls.MODEL_CACHE_DIR / "pose_landmarker_lite.task",
        ]
        existing_model_path = next((path for path in model_asset_candidates if path.exists()), None)

        if existing_model_path is not None:
            return existing_model_path

        cls.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        download_path = cls.MODEL_CACHE_DIR / "pose_landmarker_lite.task"
        temp_download_path = cls.MODEL_CACHE_DIR / "pose_landmarker_lite.task.tmp"

        try:
            with urlopen(cls.MODEL_ASSET_URL, timeout=60) as response, temp_download_path.open("wb") as file_obj:
                file_obj.write(response.read())
            temp_download_path.replace(download_path)
        except Exception as e:
            temp_download_path.unlink(missing_ok=True)
            raise RuntimeError(f"pose_landmarker_lite.task 다운로드에 실패했습니다: {e}")

        return download_path

    def detect_pose(self, frame: np.ndarray) -> PoseDetectionResult:
        """
        이미지 프레임에서 자세를 감지합니다.

        Args:
            frame: OpenCV 프레임 (BGR 형식)

        Returns:
            PoseDetectionResult: 감지된 포즈 결과
        """
        # BGR을 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape

        # MediaPipe Image 생성
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # 포즈 감지
        self.timestamp_ms += 33  # ~30fps
        detection_result = self.detector.detect_for_video(mp_image, self.timestamp_ms)

        landmarks_list = []
        confidence = 0.0
        is_detected = False

        if detection_result.pose_landmarks:
            is_detected = True
            visibilities = []

            for idx, landmark in enumerate(detection_result.pose_landmarks[0]):
                # 랜드마크는 이미 정규화된 좌표
                x = landmark.x
                y = landmark.y
                z = landmark.z
                visibility = landmark.visibility if landmark.visibility else 0.0

                visibilities.append(visibility)

                landmark_obj = Landmark(
                    id=idx,
                    name=self.LANDMARK_NAMES[idx] if idx < len(self.LANDMARK_NAMES) else f"landmark_{idx}",
                    x=round(x, 6),
                    y=round(y, 6),
                    z=round(z, 6),
                    visibility=round(visibility, 6)
                )
                landmarks_list.append(landmark_obj)

            # 전체 신뢰도는 visibility의 평균
            confidence = round(np.mean(visibilities), 6) if visibilities else 0.0

        result = PoseDetectionResult(
            landmarks=landmarks_list,
            confidence=confidence,
            frame_width=frame_width,
            frame_height=frame_height,
            timestamp=datetime.utcnow().isoformat() + "Z",
            is_detected=is_detected
        )

        return result

    def draw_landmarks_on_frame(self, frame: np.ndarray,
                               result: PoseDetectionResult,
                               draw_connections: bool = True) -> np.ndarray:
        """
        감지된 랜드마크를 프레임에 그립니다.

        Args:
            frame: OpenCV 프레임
            result: 포즈 감지 결과
            draw_connections: 포인트 간 연결선 그릴지 여부

        Returns:
            랜드마크가 그려진 프레임
        """
        frame_copy = frame.copy()

        if not result.is_detected or not result.landmarks:
            return frame_copy

        h, w = frame.shape[:2]

        # 랜드마크 포인트 그리기
        for landmark in result.landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            visibility = landmark.visibility

            # visibility가 낮으면 투명하게 표시
            if visibility > 0.5:
                color = (0, 255, 0)  # 초록색
            else:
                color = (0, 165, 255)  # 주황색 (신뢰도 낮음)

            cv2.circle(frame_copy, (x, y), 4, color, -1)
            cv2.putText(frame_copy, str(landmark.id), (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 연결선 그리기
        if draw_connections:
            connections = [
                (0, 1), (1, 2), (2, 3),  # 왼쪽 눈
                (0, 4), (4, 5), (5, 6),  # 오른쪽 눈
                (0, 7), (7, 8),  # 귀
                (9, 10),  # 입
                (11, 12),  # 어깨
                (11, 13), (13, 15),  # 왼쪽 팔
                (12, 14), (14, 16),  # 오른쪽 팔
            ]

            for start_id, end_id in connections:
                if start_id < len(result.landmarks) and end_id < len(result.landmarks):
                    start = result.landmarks[start_id]
                    end = result.landmarks[end_id]

                    if start.visibility > 0.3 and end.visibility > 0.3:
                        x1, y1 = int(start.x * w), int(start.y * h)
                        x2, y2 = int(end.x * w), int(end.y * h)
                        cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame_copy

    def to_dict(self, result: PoseDetectionResult) -> Dict:
        """
        포즈 감지 결과를 딕셔너리로 변환합니다.
        """
        return {
            "is_detected": result.is_detected,
            "confidence": result.confidence,
            "landmarks": [
                {
                    "id": lm.id,
                    "name": lm.name,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                }
                for lm in result.landmarks
            ],
            "frame_width": result.frame_width,
            "frame_height": result.frame_height,
            "timestamp": result.timestamp
        }

    def to_json(self, result: PoseDetectionResult) -> str:
        """
        포즈 감지 결과를 JSON 문자열로 변환합니다.
        """
        return json.dumps(self.to_dict(result), ensure_ascii=False)

    def release(self):
        """리소스 해제"""
        if hasattr(self, 'detector'):
            self.detector.close()


def test_with_webcam():
    """
    웹캠에서 실시간으로 포즈를 감지하고 자세를 분석하는 테스트 함수

    사용법: python -m backend.app.services.mediapipe_detector
    """
    print("MediaPipe Pose Detector 초기화 중...")
    detector = MediaPipePoseDetector()
    analyzer = PoseAnalyzer()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        return

    print("✅ 웹캠 포즈 감지 + 자세 분석 시작 (ESC를 눌러 종료)")
    print("-" * 50)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임을 읽을 수 없습니다.")
            break

        # 프레임 크기 조정 (빠른 처리)
        frame_resized = cv2.resize(frame, (640, 480))

        # 포즈 감지
        result = detector.detect_pose(frame_resized)

        # 결과 시각화
        annotated_frame = detector.draw_landmarks_on_frame(frame_resized, result)

        h, w = annotated_frame.shape[:2]
        y_offset = 30

        # 포즈 감지 여부 표시
        if result.is_detected:
            text = f"[POSE DETECTED] Confidence: {result.confidence:.2f}"
            color = (0, 255, 0)
        else:
            text = "[NO POSE]"
            color = (0, 0, 255)

        cv2.putText(annotated_frame, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        y_offset += 35

        # 자세 분석 (포즈가 감지되었을 때만)
        if result.is_detected:
            analysis = analyzer.analyze(result.landmarks, result.confidence)

            # 자세 상태 표시
            status_color = (0, 255, 0) if analysis.status.value == "good" else \
                          (0, 165, 255) if analysis.status.value == "warning" else (0, 0, 255)

            status_text = f"[{analysis.status.value.upper()}]"
            cv2.putText(annotated_frame, status_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 2)
            y_offset += 35

            # 자세 지표 표시
            metrics_text = f"Neck: {analysis.neck_forward_angle:.1f}deg | Shoulder: {analysis.shoulder_slope:.1f}deg"
            cv2.putText(annotated_frame, metrics_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            y_offset += 25

            spine_text = f"Spine: {analysis.spine_alignment:.2f}"
            cv2.putText(annotated_frame, spine_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            y_offset += 25

            # 문제점 표시
            if analysis.issues:
                cv2.putText(annotated_frame, "[ISSUES]", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 1)
                y_offset += 25
                for issue in analysis.issues:
                    cv2.putText(annotated_frame, f"- {issue}", (15, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
                    y_offset += 20

        # 화면에 표시
        cv2.imshow("Pose Detection + Analysis", annotated_frame)

        # 프레임 카운트 및 주기적 로그 출력
        frame_count += 1
        if frame_count % 30 == 0:
            if result.is_detected:
                print(f"[Frame {frame_count}] Status: {analysis.status.value}, "
                      f"Neck: {analysis.neck_forward_angle:.1f}°, "
                      f"Shoulder: {analysis.shoulder_slope:.1f}°, "
                      f"Spine: {analysis.spine_alignment:.2f}")
            else:
                print(f"[Frame {frame_count}] No pose detected")

        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            print("\n✅ 웹캠 포즈 분석 종료")
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.release()


if __name__ == "__main__":
    test_with_webcam()
