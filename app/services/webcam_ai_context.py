import hashlib
import json
from typing import Any

from app.services.webcam_comparator import ComparisonResult


def build_ai_context(result: ComparisonResult) -> dict[str, Any]:
    issue_priority = [
        "BAD_POSTURE",
        "NECK_FORWARD",
        "HEAD_TILT",
        "SHOULDER_SLOPE",
        "HIP_DEVIATION",
    ]
    issue_labels = {
        "BAD_POSTURE": "overall posture breakdown",
        "NECK_FORWARD": "forward head tendency",
        "HEAD_TILT": "head tilt",
        "SHOULDER_SLOPE": "shoulder imbalance",
        "HIP_DEVIATION": "hip imbalance",
    }

    ordered_issues = sorted(
        result.issues,
        key=lambda issue: issue_priority.index(issue) if issue in issue_priority else len(issue_priority),
    )
    primary_issue = ordered_issues[0] if ordered_issues else None
    primary_issue_label = issue_labels.get(primary_issue, "general posture deviation") if primary_issue else None

    if result.status == "bad":
        severity = "high"
    elif result.status == "warning":
        severity = "medium"
    else:
        severity = "low"

    context = {
        "status": result.status,
        "deviation_score": result.deviation_score,
        "severity": severity,
        "issues": ordered_issues,
        "issue_count": len(ordered_issues),
        "primary_issue": primary_issue,
        "primary_issue_label": primary_issue_label,
        "has_forward_head_signal": "NECK_FORWARD" in ordered_issues,
        "has_head_tilt_signal": "HEAD_TILT" in ordered_issues,
        "has_shoulder_imbalance_signal": "SHOULDER_SLOPE" in ordered_issues,
        "has_hip_imbalance_signal": "HIP_DEVIATION" in ordered_issues,
        "has_face_proximity_signal": "NECK_FORWARD" in ordered_issues,
    }
    context["judgement_signature"] = _build_judgement_signature(context)
    return context


def _build_judgement_signature(context: dict[str, Any]) -> str:
    signature_payload = {
        "status": context.get("status"),
        "severity": context.get("severity"),
        "issues": context.get("issues") or [],
        "primary_issue": context.get("primary_issue"),
        "has_forward_head_signal": context.get("has_forward_head_signal"),
        "has_head_tilt_signal": context.get("has_head_tilt_signal"),
        "has_shoulder_imbalance_signal": context.get("has_shoulder_imbalance_signal"),
        "has_hip_imbalance_signal": context.get("has_hip_imbalance_signal"),
    }
    raw = json.dumps(signature_payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
