import base64
import hashlib
import hmac
import json
from typing import Any


class AnalysisTokenService:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode("utf-8")

    def dumps(self, payload: dict[str, Any]) -> str:
        body = self._serialize(payload)
        signature = hmac.new(self.secret_key, body.encode("utf-8"), hashlib.sha256).hexdigest()
        encoded_body = self._urlsafe_b64encode(body.encode("utf-8"))
        return f"{encoded_body}.{signature}"

    def loads(self, token: str) -> dict[str, Any]:
        try:
            encoded_body, signature = token.split(".", 1)
        except ValueError as exc:
            raise ValueError("Invalid save token format.") from exc

        body_bytes = self._urlsafe_b64decode(encoded_body)
        expected_signature = hmac.new(
            self.secret_key,
            body_bytes,
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_signature):
            raise ValueError("Invalid save token signature.")

        return json.loads(body_bytes.decode("utf-8"))

    @staticmethod
    def _serialize(payload: dict[str, Any]) -> str:
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @staticmethod
    def _urlsafe_b64encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

    @staticmethod
    def _urlsafe_b64decode(data: str) -> bytes:
        padding = "=" * (-len(data) % 4)
        return base64.urlsafe_b64decode(data + padding)
