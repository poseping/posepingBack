import math
from dataclasses import dataclass

_KEY_POINTS: dict[int, tuple[str, float]] = {
    0:  ("코",        0.25),
    7:  ("왼쪽 귀",   0.15),
    8:  ("오른쪽 귀", 0.15),
    11: ("왼쪽 어깨", 0.15),
    12: ("오른쪽 어깨", 0.15),
}

# 랜드마크 인덱스 → webcam_alert_type.alert_type_id 매핑
_POINT_TO_ALERT: dict[int, str] = {
    0:  "NECK_FORWARD",
    7:  "HEAD_TILT",
    8:  "HEAD_TILT",
    11: "SHOULDER_SLOPE",
    12: "SHOULDER_SLOPE",
}

_SENSITIVITY_THRESHOLDS: dict[str, dict[str, float]] = {
    "low":    {"good": 0.15, "bad": 0.35, "issue": 0.20},
    "medium": {"good": 0.10, "bad": 0.25, "issue": 0.15},
    "high":   {"good": 0.07, "bad": 0.18, "issue": 0.10},
}
_DEFAULT_SENSITIVITY = "medium"

# 얼굴/어깨 비율이 기준 대비 이 값 이상이면 거북목 주의 (NECK_FORWARD 발생)
_FACE_PROXIMITY_WARNING = 1.15
# 이 값 이상이면 심한 거북목으로 status "bad" 강제
_FACE_PROXIMITY_BAD = 1.25

# 귀/눈 근접비율 계산 시 최소 visibility (0.2로 완화 — 정면에서도 귀가 잘 탈락)
_VISIBILITY_MIN = 0.2
# 메인 비교 루프에서 신뢰도가 낮은 랜드마크를 skip할 최소 visibility
_KEY_POINT_VIS_MIN = 0.3


def _shoulder_frame(landmarks: list) -> tuple[float, float, float] | None:
    """현재 랜드마크 리스트에서 어깨 기준 좌표계 (mid_x, mid_y, width) 추출"""
    if len(landmarks) <= 12:
        return None
    ls, rs = landmarks[11], landmarks[12]
    width = math.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2)
    if width < 1e-6:
        return None
    return (ls.x + rs.x) / 2, (ls.y + rs.y) / 2, width


def _shoulder_frame_ref(ref_map: dict) -> tuple[float, float, float] | None:
    """기준 랜드마크 딕셔너리에서 어깨 기준 좌표계 (mid_x, mid_y, width) 추출"""
    if 11 not in ref_map or 12 not in ref_map:
        return None
    ls, rs = ref_map[11], ref_map[12]
    width = math.sqrt((ls["x"] - rs["x"]) ** 2 + (ls["y"] - rs["y"]) ** 2)
    if width < 1e-6:
        return None
    return (ls["x"] + rs["x"]) / 2, (ls["y"] + rs["y"]) / 2, width


def _rel(x: float, y: float, frame: tuple[float, float, float]) -> tuple[float, float]:
    """절대 좌표 → 어깨 기준 상대 좌표 변환"""
    mid_x, mid_y, width = frame
    return (x - mid_x) / width, (y - mid_y) / width


def _get_face_proximity_ratio(current_landmarks: list, ref_map: dict) -> float | None:
    """
    얼굴 근접 비율(현재/기준) 반환. 두 지표 중 더 큰 값 사용.

    - 귀 간격 / 어깨 너비: 정면 응시 시 효과적
    - 눈-입 수직 높이 / 어깨 너비: 고개를 옆으로 돌려도 Y축은 안정적이라 측면 모니터에 효과적

    고개를 돌리면 귀 간격이 원근법으로 줄어들어 비율이 과소평가되는 문제를 보완.
    """
    LEFT_EAR, RIGHT_EAR = 7, 8
    LEFT_EYE, RIGHT_EYE = 2, 5
    MOUTH_LEFT, MOUTH_RIGHT = 9, 10
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

    def _dist(ax, ay, bx, by) -> float:
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    # 어깨 간격 (공통 분모)
    if any(idx >= len(current_landmarks) for idx in [LEFT_SHOULDER, RIGHT_SHOULDER]):
        return None
    if any(idx not in ref_map for idx in [LEFT_SHOULDER, RIGHT_SHOULDER]):
        return None

    cur_shoulder = _dist(
        current_landmarks[LEFT_SHOULDER].x, current_landmarks[LEFT_SHOULDER].y,
        current_landmarks[RIGHT_SHOULDER].x, current_landmarks[RIGHT_SHOULDER].y,
    )
    ref_shoulder = _dist(
        ref_map[LEFT_SHOULDER]["x"], ref_map[LEFT_SHOULDER]["y"],
        ref_map[RIGHT_SHOULDER]["x"], ref_map[RIGHT_SHOULDER]["y"],
    )
    if cur_shoulder < 1e-6 or ref_shoulder < 1e-6:
        return None

    ratios: dict[str, float] = {}

    # 지표 1: 귀 간격 / 어깨 너비 (양쪽 귀 모두 충분히 보일 때)
    ears_visible = (
        all(idx < len(current_landmarks) for idx in [LEFT_EAR, RIGHT_EAR])
        and getattr(current_landmarks[LEFT_EAR], "visibility", 1.0) >= _VISIBILITY_MIN
        and getattr(current_landmarks[RIGHT_EAR], "visibility", 1.0) >= _VISIBILITY_MIN
        and all(idx in ref_map for idx in [LEFT_EAR, RIGHT_EAR])
    )
    if ears_visible:
        cur_ear = _dist(
            current_landmarks[LEFT_EAR].x, current_landmarks[LEFT_EAR].y,
            current_landmarks[RIGHT_EAR].x, current_landmarks[RIGHT_EAR].y,
        )
        ref_ear = _dist(
            ref_map[LEFT_EAR]["x"], ref_map[LEFT_EAR]["y"],
            ref_map[RIGHT_EAR]["x"], ref_map[RIGHT_EAR]["y"],
        )
        if ref_ear > 1e-6:
            ratios["귀간격"] = round((cur_ear / cur_shoulder) / (ref_ear / ref_shoulder), 4)

    # 지표 2: 눈-입 수직 높이 / 어깨 너비 (고개 회전에 강건)
    face_h_ok = (
        all(idx < len(current_landmarks) for idx in [LEFT_EYE, RIGHT_EYE, MOUTH_LEFT, MOUTH_RIGHT])
        and getattr(current_landmarks[LEFT_EYE], "visibility", 1.0) >= _VISIBILITY_MIN
        and getattr(current_landmarks[RIGHT_EYE], "visibility", 1.0) >= _VISIBILITY_MIN
        and all(idx in ref_map for idx in [LEFT_EYE, RIGHT_EYE, MOUTH_LEFT, MOUTH_RIGHT])
    )
    if face_h_ok:
        cur_eye_y = (current_landmarks[LEFT_EYE].y + current_landmarks[RIGHT_EYE].y) / 2
        cur_mouth_y = (current_landmarks[MOUTH_LEFT].y + current_landmarks[MOUTH_RIGHT].y) / 2
        cur_face_h = abs(cur_eye_y - cur_mouth_y)

        ref_eye_y = (ref_map[LEFT_EYE]["y"] + ref_map[RIGHT_EYE]["y"]) / 2
        ref_mouth_y = (ref_map[MOUTH_LEFT]["y"] + ref_map[MOUTH_RIGHT]["y"]) / 2
        ref_face_h = abs(ref_eye_y - ref_mouth_y)

        if ref_face_h > 1e-6:
            ratios["얼굴높이"] = round((cur_face_h / cur_shoulder) / (ref_face_h / ref_shoulder), 4)

    if not ratios:
        return None
    return max(ratios.values()), ratios


@dataclass
class ComparisonResult:
    status: str           # "good" | "warning" | "bad"
    deviation_score: float
    issues: list[str]     # webcam_alert_type.alert_type_id 리스트 (중복 없음)
    per_point: dict[str, float]


def compare(current_landmarks: list, reference_landmarks: list[dict], sensitivity: str = _DEFAULT_SENSITIVITY) -> ComparisonResult:
    ref_map = {lm["id"]: lm for lm in reference_landmarks}
    t = _SENSITIVITY_THRESHOLDS.get(sensitivity, _SENSITIVITY_THRESHOLDS[_DEFAULT_SENSITIVITY])

    # 어깨 기준 좌표계 구성 (카메라 거리·위치 변화 제거)
    cur_frame = _shoulder_frame(current_landmarks)
    ref_frame = _shoulder_frame_ref(ref_map)
    if cur_frame is None or ref_frame is None:
        return ComparisonResult(status="warning", deviation_score=0.0, issues=[], per_point={})

    weighted_sum = 0.0
    weight_total = 0.0
    per_point: dict[str, float] = {}
    triggered: set[str] = set()

    for idx, (label, weight) in _KEY_POINTS.items():
        if idx >= len(current_landmarks) or idx not in ref_map:
            continue

        cur = current_landmarks[idx]
        if getattr(cur, "visibility", 1.0) < _KEY_POINT_VIS_MIN:
            continue

        ref = ref_map[idx]

        cur_rx, cur_ry = _rel(cur.x, cur.y, cur_frame)
        ref_rx, ref_ry = _rel(ref["x"], ref["y"], ref_frame)

        dist = math.sqrt((cur_rx - ref_rx) ** 2 + (cur_ry - ref_ry) ** 2)
        per_point[label] = round(dist, 4)
        weighted_sum += dist * weight
        weight_total += weight

        if dist >= t["issue"]:
            alert_id = _POINT_TO_ALERT.get(idx)
            if alert_id:
                triggered.add(alert_id)

    if weight_total == 0:
        return ComparisonResult(
            status="warning",
            deviation_score=0.0,
            issues=[],
            per_point={},
        )

    score = weighted_sum / weight_total

    # 얼굴/어깨 비율로 정면 거북목(카메라 접근) 추가 감지
    neck_proximity_bad = False
    proximity_result = _get_face_proximity_ratio(current_landmarks, ref_map)
    if proximity_result is not None:
        proximity_ratio, proximity_details = proximity_result
        per_point["얼굴근접비율"] = proximity_ratio
        per_point.update({f"근접_{k}": v for k, v in proximity_details.items()})
        if proximity_ratio >= _FACE_PROXIMITY_BAD:
            triggered.add("NECK_FORWARD")
            neck_proximity_bad = True
        elif proximity_ratio >= _FACE_PROXIMITY_WARNING:
            triggered.add("NECK_FORWARD")

    # status 결정: deviation_score + 심각 issue 반영
    if score >= t["bad"] or neck_proximity_bad:
        status = "bad"
        triggered.add("BAD_POSTURE")
    elif score >= t["good"]:
        status = "warning"
    else:
        status = "good"

    return ComparisonResult(
        status=status,
        deviation_score=round(score, 4),
        issues=sorted(triggered),
        per_point=per_point,
    )
