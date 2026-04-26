from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from datetime import datetime

from app.api.dependencies import verify_auth
from app.core.config import settings
from app.db.session import get_db
from app.models.models import Member, UserLifestyleHabit
from app.services.gemini_service import (
    ChatMessage,
    GeminiAssistantService,
    OnboardingChatResult,
    PhotoCommentResult,
    WebcamCommentResult,
)
from app.services.onboarding_fields import ONBOARDING_MAX_TURNS
from app.services.onboarding_session import (
    build_onboarding_context,
    build_fallback_question,
    count_user_turns,
    get_missing_fields,
    merge_collected_fields,
    normalize_collected_fields,
)

router = APIRouter()

assistant_service: GeminiAssistantService | None = None


def get_assistant_service() -> GeminiAssistantService:
    global assistant_service

    if assistant_service is None:
        assistant_service = GeminiAssistantService(
            api_key=settings.vertex_ai_api_key,
            model_name=settings.gemini_model,
        )
    return assistant_service


class WebcamCommentRequest(BaseModel):
    status: str
    deviation_score: float
    issues: list[str] = Field(default_factory=list)
    ai_context: dict = Field(default_factory=dict)
    profile_name: str | None = None
    judgement_signature: str | None = None
    previous_judgement_signature: str | None = None


class WebcamCommentResponse(BaseModel):
    requested: bool
    judgement_changed: bool
    judgement_signature: str | None = None
    comment: str | None = None


class PhotoCommentRequest(BaseModel):
    status: str
    analysis_mode: str
    confidence: float
    side_view: str
    issues: list[str] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)
    missing_landmarks: list[str] = Field(default_factory=list)
    front: dict = Field(default_factory=dict)
    side: dict = Field(default_factory=dict)


class PhotoCommentResponse(BaseModel):
    comment: str


class OnboardingChatRequest(BaseModel):
    user_prompt: str = Field(..., min_length=1, max_length=500)
    chat_history: list[ChatMessage] = Field(default_factory=list)
    collected_fields: dict[str, str] = Field(default_factory=dict)


class OnboardingChatResponse(BaseModel):
    reply: str
    done: bool
    stop_reason: str | None = None
    collected_fields: dict[str, str] = Field(default_factory=dict)
    missing_fields: list[dict[str, str]] = Field(default_factory=list)
    turn_count: int
    max_turns: int


@router.post("/webcam-comment", response_model=WebcamCommentResponse)
async def generate_webcam_comment(
    request: WebcamCommentRequest,
    member: Member = Depends(verify_auth),
) -> WebcamCommentResponse:
    _ = member
    service = get_assistant_service()
    current_signature = request.judgement_signature or request.ai_context.get("judgement_signature")
    judgement_changed = current_signature != request.previous_judgement_signature

    if not judgement_changed:
        return WebcamCommentResponse(
            requested=False,
            judgement_changed=False,
            judgement_signature=current_signature,
            comment=None,
        )

    try:
        result: WebcamCommentResult = await service.generate_webcam_comment(
            analysis_context={
                "status": request.status,
                "deviation_score": request.deviation_score,
                "issues": request.issues,
                "profile_name": request.profile_name,
                "ai_context": request.ai_context,
            },
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return WebcamCommentResponse(
        requested=True,
        judgement_changed=True,
        judgement_signature=current_signature,
        comment=result.comment,
    )


@router.post("/photo-comment", response_model=PhotoCommentResponse)
async def generate_photo_comment(
    request: PhotoCommentRequest,
    member: Member = Depends(verify_auth),
) -> PhotoCommentResponse:
    _ = member
    service = get_assistant_service()

    try:
        result: PhotoCommentResult = await service.generate_photo_comment(
            analysis_context={
                "status": request.status,
                "analysis_mode": request.analysis_mode,
                "confidence": request.confidence,
                "side_view": request.side_view,
                "issues": request.issues,
                "alerts": request.alerts,
                "missing_landmarks": request.missing_landmarks,
                "front": request.front,
                "side": request.side,
            },
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PhotoCommentResponse(comment=result.comment)


@router.post("/onboarding-chat", response_model=OnboardingChatResponse)
async def onboarding_chat(
    request: OnboardingChatRequest,
    member: Member = Depends(verify_auth),
    db: Session = Depends(get_db),
) -> OnboardingChatResponse:
    _ = member
    service = get_assistant_service()
    collected_fields = normalize_collected_fields(request.collected_fields)
    turn_count = count_user_turns(request.chat_history, request.user_prompt)
    missing_fields = get_missing_fields(collected_fields)

    if len(missing_fields) == 0:
        return OnboardingChatResponse(
            reply="기본 정보 확인을 마쳤습니다.",
            done=True,
            stop_reason="completed",
            collected_fields=collected_fields,
            missing_fields=[],
            turn_count=turn_count,
            max_turns=ONBOARDING_MAX_TURNS,
        )

    if turn_count > ONBOARDING_MAX_TURNS:
        return OnboardingChatResponse(
            reply="여기까지 정보를 정리할게요.",
            done=True,
            stop_reason="max_turn_reached",
            collected_fields=collected_fields,
            missing_fields=missing_fields,
            turn_count=turn_count,
            max_turns=ONBOARDING_MAX_TURNS,
        )

    try:
        result: OnboardingChatResult = await service.generate_onboarding_reply(
            user_prompt=request.user_prompt,
            context_data=build_onboarding_context(
                collected_fields=collected_fields,
                missing_fields=missing_fields,
                current_turn=turn_count,
                max_turns=ONBOARDING_MAX_TURNS,
            ),
            chat_history=request.chat_history,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    merged_fields = merge_collected_fields(collected_fields, result.extracted_fields)
    remaining_fields = get_missing_fields(merged_fields)
    done = len(remaining_fields) == 0 or turn_count >= ONBOARDING_MAX_TURNS

    if len(remaining_fields) == 0:
        stop_reason = "completed"
    elif turn_count >= ONBOARDING_MAX_TURNS:
        stop_reason = "max_turn_reached"
    else:
        stop_reason = None

    if done:
        if stop_reason == "completed":
            try:
                lifestyle_habit = (
                    db.query(UserLifestyleHabit)
                    .filter(UserLifestyleHabit.member_id == member.member_id)
                    .first()
                )
                if lifestyle_habit is None:
                    lifestyle_habit = UserLifestyleHabit(member_id=member.member_id)
                    db.add(lifestyle_habit)

                lifestyle_habit.sitting_hours_per_day = merged_fields.get("sitting_hours_per_day")
                lifestyle_habit.exercise_days_per_week = merged_fields.get("exercise_days_per_week")
                lifestyle_habit.pain_areas = merged_fields.get("pain_areas")
                lifestyle_habit.updated_at = datetime.utcnow()
                db.commit()
            except SQLAlchemyError as exc:
                db.rollback()
                raise HTTPException(status_code=503, detail="Failed to save onboarding data.") from exc

        closing_reply = "기본 정보 확인을 마쳤습니다."
        if stop_reason == "max_turn_reached":
            closing_reply = "여기까지 정보를 정리할게요."

        return OnboardingChatResponse(
            reply=closing_reply,
            done=True,
            stop_reason=stop_reason,
            collected_fields=merged_fields,
            missing_fields=remaining_fields,
            turn_count=turn_count,
            max_turns=ONBOARDING_MAX_TURNS,
        )

    next_reply = (result.reply or "").strip() or build_fallback_question(remaining_fields)

    return OnboardingChatResponse(
        reply=next_reply,
        done=False,
        stop_reason=None,
        collected_fields=merged_fields,
        missing_fields=remaining_fields,
        turn_count=turn_count,
        max_turns=ONBOARDING_MAX_TURNS,
    )
