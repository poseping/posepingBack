import math
from dataclasses import dataclass

_KEY_POINTS: dict[int, tuple[str, float]] = {
    0:  ("코",        0.25),
    7:  ("왼쪽 귀",   0.15),
    8:  ("오른쪽 귀", 0.15),
    11: ("왼쪽 어깨", 0.15),
    12: ("오른쪽 어깨", 0.15),
    23: ("왼쪽 골반", 0.075),
    24: ("오른쪽 골반", 0.075),
}

# 랜드마크 인덱스 → webcam_alert_type.alert_type_id 매핑
_POINT_TO_ALERT: dict[int, str] = {
    0:  "NECK_FORWARD",
    7:  "HEAD_TILT",
    8:  "HEAD_TILT",
    11: "SHOULDER_SLOPE",
    12: "SHOULDER_SLOPE",
    23: "HIP_DEVIATION",
    24: "HIP_DEVIATION",
}

GOOD_THRESHOLD = 0.05
BAD_THRESHOLD = 0.10


@dataclass
class ComparisonResult:
    status: str           # "good" | "warning" | "bad"
    deviation_score: float
    issues: list[str]     # webcam_alert_type.alert_type_id 리스트 (중복 없음)
    per_point: dict[str, float]


def compare(current_landmarks: list, reference_landmarks: list[dict]) -> ComparisonResult:
    ref_map = {lm["id"]: lm for lm in reference_landmarks}

    weighted_sum = 0.0
    weight_total = 0.0
    per_point: dict[str, float] = {}
    triggered: set[str] = set()

    for idx, (label, weight) in _KEY_POINTS.items():
        if idx >= len(current_landmarks) or idx not in ref_map:
            continue

        cur = current_landmarks[idx]
        ref = ref_map[idx]

        dist = math.sqrt((cur.x - ref["x"]) ** 2 + (cur.y - ref["y"]) ** 2)
        per_point[label] = round(dist, 4)
        weighted_sum += dist * weight
        weight_total += weight

        if dist > GOOD_THRESHOLD:
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

    if score < GOOD_THRESHOLD:
        status = "good"
    elif score < BAD_THRESHOLD:
        status = "warning"
    else:
        status = "bad"
        triggered.add("BAD_POSTURE")

    return ComparisonResult(
        status=status,
        deviation_score=round(score, 4),
        issues=sorted(triggered),
        per_point=per_point,
    )
