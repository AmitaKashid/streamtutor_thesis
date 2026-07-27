# prompts/registry.py
from __future__ import annotations

from typing import Optional

from . import socratic, inquiry, adaptive, collaborative

_STYLE_MAP = {
    "socratic": socratic.SYSTEM_PROMPT,
    "inquiry": inquiry.SYSTEM_PROMPT,
    "adaptive": adaptive.SYSTEM_PROMPT,
    "collaborative": collaborative.SYSTEM_PROMPT,
}


def get_style_prompt(style: str) -> str:
    style = (style or "socratic").strip().lower()
    if style not in _STYLE_MAP:
        raise ValueError(f"Unknown teaching style: {style}. Choose from {list(_STYLE_MAP.keys())}")
    return _STYLE_MAP[style]


def compose_prompt(
    *,
    style: str,
    user_query: str,
    context: Optional[str] = None,
) -> str:
    """
    Compose a single plain-text prompt suitable for llama.cpp / text-generation backends.

    Layout is explicit:
      [SYSTEM] style instructions
      [CONTEXT] retrieved snippets (optional)
      [USER] question
      [ASSISTANT] response
    """
    sys_prompt = get_style_prompt(style).strip()
    uq = (user_query or "").strip()

    parts = []
    parts.append("[SYSTEM]\n" + sys_prompt)

    if context:
        ctx = context.strip()
        parts.append("[CONTEXT]\n" + ctx)

    parts.append("[USER]\n" + uq)
    parts.append("[ASSISTANT]\n")
    return "\n\n".join(parts)