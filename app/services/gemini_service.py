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


class OnboardingExtractedField(BaseModel):
    key: str = Field(..., min_length=1, max_length=100)
    value: str = Field(..., min_length=1, max_length=200)


class OnboardingChatResult(BaseModel):
    reply: str | None = Field(default=None, min_length=1, max_length=120)
    extracted_fields: list[OnboardingExtractedField] = Field(default_factory=list)


class GeminiAssistantService:
    WEBCAM_SYSTEM_PROMPT = """
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

    WEBCAM_USER_PROMPT = (
        "주어진 실시간 자세 분석 데이터를 바탕으로 지금 바로 고치면 좋은 점 한 가지만 짧게 말해줘."
    )

    PHOTO_SYSTEM_PROMPT = """
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

    PHOTO_USER_PROMPT = """
# Task
Identify the posture's strengths and problems, and explain the reasons for each.
# Context
"""

    ONBOARDING_SYSTEM_PROMPT = """
# Role
You run an onboarding conversation that collects required lifestyle information from the user.

# Task
Read the full conversation and the current user message.
Extract any required field values you can identify.
Return both:
1. the next user-facing reply
2. any extracted fields from the user's message
Always include both keys in the JSON response.
If no field was extracted, return an empty array for extracted_fields.
If a next question is needed, reply must contain that question.
If the conversation should stop, reply must contain a short closing message.

# Extraction Rules
Return extracted fields only when the user has provided a usable answer.
If the user says there is no pain, record pain_areas as 없음.
Do not invent values.
Use the required field keys exactly as provided in the context.

# Response Style
Reply in casual, friendly, polite Korean.
Use emoji.
Use one short sentence at a time.
Keep the reply concise.
Target about 10 to 30 Korean characters when asking a question.
Never exceed 60 Korean characters in the reply text.
Do not diagnose.
Do not use technical jargon.

# Conversation Rules
Ask about only one missing field at a time.
Ask only for information that is still missing.
Do not copy the example-like field questions verbatim unless there is no better natural phrasing.
Read the user's latest message and respond in a way that feels context-aware.
When the user provides an answer, briefly acknowledge or connect to what they said first.
Then add the next question at the end of the reply.
Keep the acknowledgement short and natural.
The final part of the reply should be the next question.
If all required fields are already collected, return a short closing message instead of another question.
If the current turn is the last allowed turn, return a short closing message instead of another question.
"""

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

    async def generate_webcam_comment(self, analysis_context: dict[str, Any]) -> WebcamCommentResult:
        return await self._generate_json_response(
            response_schema=WebcamCommentResult,
            system_prompt=self.WEBCAM_SYSTEM_PROMPT,
            chat_history=None,
            user_prompt=self.WEBCAM_USER_PROMPT,
            context_data=analysis_context,
        )

    async def generate_photo_comment(self, analysis_context: dict[str, Any]) -> PhotoCommentResult:
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
        context_data: dict[str, Any],
        chat_history: list[ChatMessage] | None = None,
    ) -> OnboardingChatResult:
        return await self._generate_json_response(
            response_schema=OnboardingChatResult,
            system_prompt=self.ONBOARDING_SYSTEM_PROMPT,
            chat_history=chat_history,
            user_prompt=user_prompt,
            context_data=context_data,
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
            if settings.app_env.lower() == "dev":
                logger.info("Gemini raw response:\n%s", self._build_debug_response(response))
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

        diag = self._diagnose_empty_response(response)
        logger.warning("Gemini returned an empty/invalid response. %s", diag)
        raise RuntimeError(f"Gemini returned an invalid response. {diag}")

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

        contents.append(types.Content(role="user", parts=[types.Part(text=final_text)]))
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
                "temperature": 1.0,
                "response_mime_type": "application/json",
                "response_schema": response_schema.__name__,
                "max_output_tokens": 200,
                "thinking_budget": 0,
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _build_debug_response(response: Any) -> str:
        candidates = getattr(response, "candidates", None) or []
        candidate_summary = [
            {
                "finish_reason": str(getattr(c, "finish_reason", None)),
                "safety_ratings": [
                    {
                        "category": str(getattr(r, "category", None)),
                        "probability": str(getattr(r, "probability", None)),
                        "blocked": getattr(r, "blocked", None),
                    }
                    for r in (getattr(c, "safety_ratings", None) or [])
                ],
                "content_parts": [
                    getattr(p, "text", None)
                    for p in (getattr(getattr(c, "content", None), "parts", None) or [])
                ],
            }
            for c in candidates
        ]
        payload = {
            "text": getattr(response, "text", None),
            "parsed": getattr(response, "parsed", None),
            "prompt_feedback": getattr(response, "prompt_feedback", None),
            "usage_metadata": getattr(response, "usage_metadata", None),
            "candidates": candidate_summary,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _diagnose_empty_response(response: Any) -> str:
        prompt_feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason:
            return f"prompt_blocked={block_reason}"

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return "no_candidates"

        reasons = [str(getattr(c, "finish_reason", None)) for c in candidates]
        usage = getattr(response, "usage_metadata", None)
        usage_str = ""
        if usage is not None:
            usage_str = (
                f" input={getattr(usage, 'prompt_token_count', None)}"
                f" output={getattr(usage, 'candidates_token_count', None)}"
                f" total={getattr(usage, 'total_token_count', None)}"
            )
        return f"finish_reasons={reasons}{usage_str}"
