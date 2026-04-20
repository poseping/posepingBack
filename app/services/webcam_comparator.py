import math
from dataclasses import dataclass
from typing import Optional

# 앉은 자세 비교에 사용할 핵심 랜드마크 (인덱스: 이름, 가중치)
_KEY_POINTS: dict[int, tuple[str, float]] = {
    0:  ("코",      0.25),
    7:  ("왼쪽 귀", 0.15),
    8:  ("오른쪽 귀", 0.15),
    11: ("왼쪽 어깨", 0.15),
    12: ("오른쪽 어깨", 0.15),
    23: ("왼쪽 골반", 0.075),
    24: ("오른쪽 골반", 0.075),
}

GOOD_THRESHOLD = 0.05
BAD_THRESHOLD = 0.10


@dataclass
class ComparisonResult:
    status: str          # "good" | "warning" | "bad"
    deviation_score: float
    issues: list[str]
    per_point: dict[str, float]  # 랜드마크별 이탈량


def compare(current_landmarks: list, reference_landmarks: list[dict]) -> ComparisonResult:
    """
    현재 랜드마크와 기준 랜드마크를 비교하여 자세 이탈 정도를 반환합니다.

    Args:
        current_landmarks: MediaPipe Landmark 객체 리스트
        reference_landmarks: DB에 저장된 랜드마크 딕셔너리 리스트

    Returns:
        ComparisonResult
    """
    ref_map = {lm["id"]: lm for lm in reference_landmarks}

    weighted_sum = 0.0
    weight_total = 0.0
    per_point: dict[str, float] = {}
    issues: list[str] = []

    for idx, (label, weight) in _KEY_POINTS.items():
        if idx >= len(current_landmarks) or idx not in ref_map:
            continue

        cur = current_landmarks[idx]
        ref = ref_map[idx]

        dist = math.sqrt((cur.x - ref["x"]) ** 2 + (cur.y - ref["y"]) ** 2)
        per_point[label] = round(dist, 4)
        weighted_sum += dist * weight
        weight_total += weight

    if weight_total == 0:
        return ComparisonResult(
            status="warning",
            deviation_score=0.0,
            issues=["랜드마크를 비교할 수 없습니다"],
            per_point={},
        )

    score = weighted_sum / weight_total

    # 개별 이탈량이 큰 부위 경고
    for label, dist in per_point.items():
        if dist > BAD_THRESHOLD:
            issues.append(f"{label} 위치가 크게 벗어났습니다")
        elif dist > GOOD_THRESHOLD:
            issues.append(f"{label} 위치가 약간 벗어났습니다")

    if score < GOOD_THRESHOLD:
        status = "good"
    elif score < BAD_THRESHOLD:
        status = "warning"
    else:
        status = "bad"

    return ComparisonResult(
        status=status,
        deviation_score=round(score, 4),
        issues=issues,
        per_point=per_point,
    )
