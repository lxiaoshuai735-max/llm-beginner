"""Wikipedia lookup with Chinese/English language fallback."""

from __future__ import annotations

from typing import Any

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

def _lookup(query: str, language: str) -> str:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("the requests package is required for Wikipedia lookup") from exc

    endpoint = f"https://{language}.wikipedia.org/w/api.php"
    headers = {"User-Agent": "llm-beginner-task5/1.0 (educational project)"}
    try:
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
    except requests.RequestException as exc:
        raise RuntimeError(f"Wikipedia {language} request failed: {exc}") from exc
    pages = page.json().get("query", {}).get("pages", {})
    extract = next(iter(pages.values()), {}).get("extract", "").strip()
    return f"Wikipedia ({language}) — {title}:\n{extract[:3500]}" if extract else ""


def run(args: dict[str, Any]) -> str:
    query = str(args.get("query", "")).strip()
    if not query or len(query) > 200:
        raise ValueError("query must contain 1-200 characters")
    languages = ["zh", "en"] if any("\u4e00" <= char <= "\u9fff" for char in query) else ["en", "zh"]
    errors: list[str] = []
    for language in languages:
        try:
            result = _lookup(query, language)
            if result:
                return result
        except (RuntimeError, ValueError, KeyError) as exc:
            errors.append(f"{language}: {exc}")
    raise RuntimeError("Wikipedia lookup failed: " + "; ".join(errors or ["no matching page"]))
