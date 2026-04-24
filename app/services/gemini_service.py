import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger("uvicorn.error")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=1000)


class WebcamCommentResult(BaseModel):
    comment: str = Field(..., min_length=1, max_length=60)


class PhotoCommentResult(BaseModel):
    comment: str = Field(..., min_length=1, max_length=120)


class OnboardingChatResult(BaseModel):
    reply: str = Field(..., min_length=1, max_length=60)


class GeminiAssistantService:
    WEBCAM_SYSTEM_PROMPT = (
        """
# Role
You are a real-time posture correction assistant.

# Response Style
Reply in casual, friendly, polite Korean.
Sound warm and simple.
Do not diagnose.
Do not use technical jargon.

# Constraints
Korean only.
About 2 to 30 Korean characters total.
No more than two sentences.
Mention only one issue and one action.
Do not include medical or overly technical terms.

# Task
Find the most important posture problem and give one action the user can do immediately.
"""
    )
    WEBCAM_USER_PROMPT = (
        "주어진 실시간 자세 분석 데이터를 바탕으로 지금 바로 고치면 좋은 점 한 가지만 짧게 말해줘."
    )

    PHOTO_SYSTEM_PROMPT = (
        """
# Role
You are a coach who identifies a person's posture and explains the analysis results.

# Response Style
Reply in casual, friendly, polite Korean.
Sound warm, clear, and easy to understand.
If there is a problem, suggest an action the user can take to improve it.
Do not diagnose.
Do not use technical jargon.

# Constraints
Korean only.
Keep the response concise and natural.
Include both positive points and problem points when relevant.
If there is a problem, include at least one practical corrective action.
About 50 to 80 Korean characters total.
Do not include medical or overly technical terms.

# Field Interpretation Rules
'status' is the overall posture status.
'analysis_mode' shows whether the analysis is full or limited.
'confidence' is the overall confidence score of the whole analysis.
'front' contains front-view posture metrics.
'front.confidence' is the confidence score for the front-view analysis.
'front.shoulder_slope' indicates shoulder tilt severity.
'front.hip_slope' indicates hip tilt severity.
'front.spine_alignment' indicates how well the spine is aligned.
'front.asymmetry_score' indicates left-right imbalance severity.
'side' contains side-view posture metrics.
'side.confidence' is the confidence score for the side-view analysis.
'side.neck_forward_angle' indicates forward-head or forward-neck tendency.
'side.forward_head_detected' tells whether forward-head posture was detected.
'issues' lists the main posture problems already detected.
'alerts' lists special warnings or extra cautions.
'missing_landmarks' lists missing body landmarks if analysis was incomplete.
'side_view' tells which side image was used.

# Reasoning Rules
Treat the 'issues' array as high-priority evidence.
Use the numeric values in 'front' and 'side' to support the explanation.
If 'alerts' is empty, do not invent extra warning messages.
If 'missing_landmarks' is empty, assume the analysis was completed normally.
If 'status' is 'warning', explain the posture as needing attention but not in an alarming way.
If 'forward_head_detected' is true, consider forward-head tendency an important issue.
        """
    )
    PHOTO_USER_PROMPT = (
        """
# Task
Identify the posture's strengths and problems, and explain the reasons for each.
# Context
        """
    )

    ONBOARDING_SYSTEM_PROMPT = (
        "너는 처음 접속한 사용자의 생활 습관을 파악하는 한국어 온보딩 코치다. "
        "매 턴 답변은 한글 기준 20글자 내외의 짧은 대화체 한 문장만 사용한다. "
        "전체 대화는 약 10턴 안에서 끝날 수 있도록 한 번에 질문 하나만 한다. "
        "앉아 있는 시간, 일이나 공부 환경, 자주 쓰는 기기, 불편 부위, 운동 습관, 수면 습관을 차례로 파악한다. "
        "아직 충분히 듣기 전에는 해결책을 길게 제안하지 말고, 자연스럽게 다음 질문으로 이어간다."
    )

    def __init__(self, api_key: str, model_name: str) -> None:
        self.api_key = api_key.strip()
        self.model_name = model_name.strip()
        self._client = None

    @property
    def enabled(self) -> bool:
        if not self.api_key:
            return False

        lowered = self.api_key.lower()
        placeholder_markers = (
            "replace-with-real",
            "your-api-key",
            "temp",
            "example",
        )
        return not any(marker in lowered for marker in placeholder_markers)

    async def generate_webcam_comment(
        self,
        analysis_context: dict[str, Any],
    ) -> WebcamCommentResult:
        return await self._generate_json_response(
            response_schema=WebcamCommentResult,
            system_prompt=self.WEBCAM_SYSTEM_PROMPT,
            chat_history=None,
            user_prompt=self.WEBCAM_USER_PROMPT,
            context_data=analysis_context,
        )

    async def generate_photo_comment(
        self,
        analysis_context: dict[str, Any],
    ) -> PhotoCommentResult:
        return await self._generate_json_response(
            response_schema=PhotoCommentResult,
            system_prompt=self.PHOTO_SYSTEM_PROMPT,
            chat_history=None,
            user_prompt=self.PHOTO_USER_PROMPT,
            context_data=analysis_context,
        )

    async def generate_onboarding_reply(
        self,
        user_prompt: str,
        chat_history: list[ChatMessage] | None = None,
    ) -> OnboardingChatResult:
        return await self._generate_json_response(
            response_schema=OnboardingChatResult,
            system_prompt=self.ONBOARDING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            chat_history=chat_history,
            context_data=None,
        )

    async def _generate_json_response(
        self,
        response_schema: type[BaseModel],
        system_prompt: str,
        user_prompt: str,
        chat_history: list[ChatMessage] | None,
        context_data: dict[str, Any] | None,
    ):
        if not self.enabled:
            raise RuntimeError("Gemini API key is not configured.")

        try:
            client = self._get_client()
            if settings.app_env.lower() == "dev":
                logger.info(
                    "Gemini request payload:\n%s",
                    self._build_debug_payload(
                        model=self.model_name,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        chat_history=chat_history or [],
                        context_data=context_data,
                        response_schema=response_schema,
                    ),
                )
            response = await client.aio.models.generate_content(
                model=self.model_name,
                contents=self._build_contents(
                    user_prompt=user_prompt,
                    chat_history=chat_history or [],
                    context_data=context_data,
                ),
                config=self._build_config(system_prompt, response_schema),
            )
        except Exception as exc:
            logger.exception("Gemini request failed.")
            raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, response_schema):
            return parsed
        if isinstance(parsed, dict):
            return response_schema.model_validate(parsed)

        text = getattr(response, "text", None)
        if text:
            try:
                return response_schema.model_validate_json(text)
            except Exception as exc:
                logger.warning("Gemini returned non-JSON response: %s", exc)

        raise RuntimeError("Gemini returned an invalid response.")

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(
                vertexai=True,
                api_key=self.api_key,
            )
        return self._client

    @staticmethod
    def _build_config(system_prompt: str, response_schema: type[BaseModel]):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=1.0,
            response_mime_type="application/json",
            response_schema=response_schema,
            max_output_tokens=200,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

    @staticmethod
    def _build_contents(
        user_prompt: str,
        chat_history: list[ChatMessage],
        context_data: dict[str, Any] | None,
    ):
        from google.genai import types

        contents: list[types.Content] = []
        for message in chat_history[-20:]:
            role = "model" if message.role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=message.content.strip())],
                )
            )

        if context_data is not None:
            context_block = json.dumps(
                context_data,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            final_text = f"[CONTEXT]\n{context_block}\n\n[USER_PROMPT]\n{user_prompt.strip()}"
        else:
            final_text = user_prompt.strip()

        contents.append(
            types.Content(role="user", parts=[types.Part(text=final_text)])
        )
        return contents

    @staticmethod
    def _build_debug_payload(
        model: str,
        system_prompt: str,
        user_prompt: str,
        chat_history: list[ChatMessage],
        context_data: dict[str, Any] | None,
        response_schema: type[BaseModel],
    ) -> str:
        payload = {
            "model": model,
            "system_instruction": system_prompt,
            "user_prompt": user_prompt,
            "chat_history": [message.model_dump() for message in chat_history],
            "context_data": context_data,
            "generation_config": {
                "temperature": 0.5,
                "response_mime_type": "application/json",
                "response_schema": response_schema.__name__,
                "max_output_tokens": 200,
                "thinking_budget": 0,
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
