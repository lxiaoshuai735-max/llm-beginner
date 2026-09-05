"""Faithful extractive answer generation with an optional local HTTP model."""

from __future__ import annotations

import os
import re

import requests


def _terms(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", str(text))
    return {compact[index : index + 2] for index in range(max(0, len(compact) - 1))}


class Generator:
    def __init__(
        self,
        api_url: str | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        self.api_url = api_url or os.getenv("RAG_API_URL")
        self.model = model or os.getenv("RAG_MODEL", "qwen2.5:7b-instruct")
        inferred = "ollama" if self.api_url and "/api/chat" in self.api_url else "generic"
        self.provider = (provider or os.getenv("RAG_PROVIDER") or inferred).lower()
        self.timeout = int(os.getenv("RAG_API_TIMEOUT", "180"))

    def build_prompt(self, query: str, sources: list[dict]) -> str:
        context = "\n\n".join(
            f"[资料{index}] {item['text']}" for index, item in enumerate(sources, 1)
        )
        return (
            "请只依据给定资料回答问题；资料不足时明确说不知道，不要补充资料外事实。\n\n"
            f"{context}\n\n问题：{query}\n回答："
        )

    def generate(self, query: str, sources: list[dict]) -> str:
        if self.api_url:
            prompt = self.build_prompt(query, sources)
            if self.provider == "ollama":
                request_body = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是严谨的检索问答助手。只依据用户提供的资料作答，并保留资料编号引用。",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.2, "num_ctx": 4096},
                }
            else:
                request_body = {"prompt": prompt}
            response = requests.post(self.api_url, json=request_body, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            if self.provider == "ollama":
                answer = (payload.get("message") or {}).get("content")
            else:
                answer = payload.get("answer") or payload.get("response")
            answer = str(answer or "").strip()
            if not answer:
                raise RuntimeError(f"empty response from {self.provider} model service")
            return answer
        query_terms = _terms(query)
        sentences = []
        for source in sources:
            for sentence in re.split(r"(?<=[。！？；])|\n+", source["text"]):
                sentence = sentence.strip()
                if len(sentence) < 8:
                    continue
                overlap = len(query_terms & _terms(sentence))
                sentences.append((overlap, len(sentence), sentence))
        sentences.sort(key=lambda item: (-item[0], item[1]))
        selected = []
        for _, _, sentence in sentences:
            if sentence not in selected:
                selected.append(sentence)
            if len(selected) == 3:
                break
        return "根据检索资料，" + "".join(selected) if selected else "资料不足，无法回答。"
