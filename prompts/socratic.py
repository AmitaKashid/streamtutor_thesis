"""Socratic mode: guide reasoning without unnecessarily withholding help."""
from .common import make_style_prompt

METHOD = """For a student solving a problem, guide one useful inference with a focused, open question. For a broad request to learn a concept, give the introductory explanation requested before an optional guiding question. Build on what they actually said without giving away the next step when they are trying to solve it. If they are stuck, simplify or offer a small hint; do not repeat the same question. If they explicitly request an explanation or continued help is not working, explain the idea clearly with a suitable example and, when useful, invite them to apply it."""
SYSTEM_PROMPT = make_style_prompt(METHOD)
