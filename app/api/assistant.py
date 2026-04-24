from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import verify_auth
from app.core.config import settings
from app.models.models import Member
from app.services.gemini_service import (
    ChatMessage,
    GeminiAssistantService,
    OnboardingChatResult,
    PhotoCommentResult,
    WebcamCommentResult,
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


class OnboardingChatResponse(BaseModel):
    reply: str


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
) -> OnboardingChatResponse:
    _ = member
    service = get_assistant_service()

    try:
        result: OnboardingChatResult = await service.generate_onboarding_reply(
            user_prompt=request.user_prompt,
            chat_history=request.chat_history,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return OnboardingChatResponse(reply=result.reply)
