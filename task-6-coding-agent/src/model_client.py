"""Client for the dedicated local Qwen2.5-Coder Ollama endpoint."""

from __future__ import annotations

import os
from typing import Any

import requests


class LocalQwenClient:
    def __init__(self, api_url: str | None = None, model: str | None = None, timeout: int = 180) -> None:
        self.api_url = api_url or os.getenv("CODER_API_URL", "http://127.0.0.1:11435/api/chat")
        self.model = model or os.getenv("CODER_MODEL", "qwen2.5-coder:7b-instruct")
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 500) -> str:
        response = requests.post(
            self.api_url,
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.05,
                    "num_predict": max_tokens,
                    "num_ctx": 8192,
                    "repeat_penalty": 1.05,
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        content = payload.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"invalid local model response: {payload}")
        return content.strip()
