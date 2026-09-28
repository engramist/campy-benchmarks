"""
campy-benchmarks / records.py
Helpers for the per-probe records every suite now returns under `details`,
so a result file shows WHICH probes passed and why, not just an aggregate.
"""

from __future__ import annotations

from typing import Any, Dict, List

SNIPPET_CHARS = 240


def snippet(text: Any, n: int = SNIPPET_CHARS) -> str:
    s = text if isinstance(text, str) else str(text)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def result_texts(res: Any) -> List[str]:
    """Text of each hit in a current_truth-style response. The daemon returns
    message text under `text_raw`; the mock uses `content`/`text`."""
    if not isinstance(res, dict):
        return []
    out = []
    for r in res.get("results", []) or []:
        if isinstance(r, dict):
            t = r.get("content") or r.get("text_raw") or r.get("text") or ""
            if t:
                out.append(str(t))
    return out


def summarize_bundle(ctx: Any, max_items: int = 8) -> Dict[str, Any]:
    """Compact, bounded view of a compile_context response: section types,
    item counts, token estimate, and the first few items' text."""
    bundle = ctx.get("bundle", {}) if isinstance(ctx, dict) else {}
    sections = bundle.get("sections", []) or []
    items: List[str] = []
    summary = []
    for sec in sections:
        content = sec.get("content", []) if isinstance(sec, dict) else []
        content = content if isinstance(content, list) else [content]
        summary.append({"type": sec.get("section_type") if isinstance(sec, dict) else None,
                        "items": len(content)})
        for item in content:
            if len(items) >= max_items:
                break
            if isinstance(item, dict):
                item = item.get("text") or item.get("content") or item.get("text_raw") or item
            items.append(snippet(item))
    return {
        "total_token_estimate": bundle.get("total_token_estimate"),
        "sections": summary,
        "top_items": items,
    }
