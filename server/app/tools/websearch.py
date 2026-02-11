"""
Web search tool - uses DuckDuckGo's HTML endpoint (no API key required).

why DuckDuckGo over Google/Bing APIs: this is a portfolio project meant
to run out-of-the-box with zero paid API keys. DDG's lite HTML endpoint
is free, has no auth, and is good enough for "quick fact lookup" - not
meant to be a production search integration.
"""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information the model wouldn't know (news, facts, prices, etc).",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
}


def web_search(query: str) -> str:
    try:
        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (voice-operations-agent demo)"},
            timeout=6.0,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for result in soup.select(".result__snippet")[:3]:
            text = result.get_text(strip=True)
            if text:
                results.append(text)
        if not results:
            return f"No web results found for '{query}'."
        return " | ".join(results)
    except Exception as exc:  # noqa: BLE001
        return f"Web search failed: {exc}"
