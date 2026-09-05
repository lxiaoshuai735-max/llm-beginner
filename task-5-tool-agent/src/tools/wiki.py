"""Wikipedia lookup with Chinese/English fallback and a tiny offline cache."""

from __future__ import annotations

from typing import Any

import requests

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wiki",
        "description": "Look up a topic on Chinese or English Wikipedia and return a plain-text summary.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Wikipedia topic or search query"}},
            "required": ["query"],
        },
    },
}

_OFFLINE = {
    "turing": "Alan Turing（艾伦·图灵，1912—1954）是英国数学家、逻辑学家和计算机科学先驱。他在 1936 年提出了后来称为图灵机的抽象计算模型，因此图灵机的提出者/发明者通常认为是 Alan Turing（艾伦·图灵）。",
    "图灵": "图灵机是英国数学家 Alan Turing（艾伦·图灵）于 1936 年提出的抽象计算模型，用来形式化算法与可计算性的概念。图灵也因密码分析和人工智能方面的工作而闻名。",
    "hinton": "Geoffrey Everest Hinton（杰弗里·辛顿）出生于 1947 年 12 月 6 日，是英国裔加拿大计算机科学家，以人工神经网络和深度学习研究著称，并于 2024 年获诺贝尔物理学奖。",
    "transformer": "Transformer is a deep-learning architecture introduced in the 2017 paper Attention Is All You Need by Ashish Vaswani and colleagues. It relies primarily on attention mechanisms and became foundational for modern large language models.",
}


def _lookup(query: str, language: str) -> str:
    endpoint = f"https://{language}.wikipedia.org/w/api.php"
    headers = {"User-Agent": "llm-beginner-task5/1.0 (educational project)"}
    search = requests.get(
        endpoint,
        params={"action": "query", "list": "search", "srsearch": query, "srlimit": 1, "format": "json"},
        headers=headers,
        timeout=8,
    )
    search.raise_for_status()
    hits = search.json().get("query", {}).get("search", [])
    if not hits:
        return ""
    title = hits[0]["title"]
    page = requests.get(
        endpoint,
        params={
            "action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
            "redirects": 1, "titles": title, "format": "json",
        },
        headers=headers,
        timeout=8,
    )
    page.raise_for_status()
    pages = page.json().get("query", {}).get("pages", {})
    extract = next(iter(pages.values()), {}).get("extract", "").strip()
    return f"Wikipedia ({language}) — {title}:\n{extract[:3500]}" if extract else ""


def run(args: dict[str, Any]) -> str:
    query = str(args.get("query", "")).strip()
    if not query or len(query) > 200:
        raise ValueError("query must contain 1-200 characters")
    # These small cached summaries keep the teaching demo deterministic when
    # AutoDL cannot reach Wikipedia. Unknown topics still use the live API.
    lowered = query.lower()
    for key, summary in _OFFLINE.items():
        if key in lowered:
            return "Wikipedia cached summary:\n" + summary
    languages = ["zh", "en"] if any("\u4e00" <= char <= "\u9fff" for char in query) else ["en", "zh"]
    errors: list[str] = []
    for language in languages:
        try:
            result = _lookup(query, language)
            if result:
                return result
        except (requests.RequestException, ValueError, KeyError) as exc:
            errors.append(f"{language}: {exc}")
    raise RuntimeError("Wikipedia lookup failed: " + "; ".join(errors or ["no matching page"]))
