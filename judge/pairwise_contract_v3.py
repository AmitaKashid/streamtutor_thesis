from __future__ import annotations

import re
from typing import Any


PAIRWISE_PROTOCOL_VERSION = "3.0"
RESULT_MARKER = "[RESULT]"

_OUTPUT_PATTERN = re.compile(
    r"(?P<feedback>.*?)\s*\[RESULT\]\s*(?P<choice>[12])",
    flags=re.DOTALL,
)


def parse_pairwise_output(
    text: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(text, str):
        return None, "output_type_error: expected string"
    stripped = text.strip()
    if not stripped:
        return None, "empty_output"
    marker_count = stripped.count(RESULT_MARKER)
    if marker_count != 1:
        return (
            None,
            "result_marker_count_error: "
            f"expected 1 marker, found {marker_count}",
        )
    if "```" in stripped:
        return None, "format_error: Markdown fences are not permitted"
    match = _OUTPUT_PATTERN.fullmatch(stripped)
    if match is None:
        return (
            None,
            "format_error: expected non-empty feedback followed by "
            "'[RESULT] <1 or 2>' with no trailing text",
        )
    feedback = match.group("feedback").strip()
    format_variant = "unprefixed_feedback"
    if feedback.startswith("Feedback:"):
        feedback = feedback[len("Feedback:") :].strip()
        format_variant = "prefixed_feedback"
    if not feedback:
        return None, "empty_feedback"
    return {
        "protocol_version": PAIRWISE_PROTOCOL_VERSION,
        "feedback": feedback,
        "choice": int(match.group("choice")),
        "format_variant": format_variant,
    }, None


def build_pairwise_retry_prompt(
    original_prompt: str,
    validation_error: str,
) -> str:
    if not original_prompt.strip():
        raise ValueError("original_prompt must not be empty")
    if not validation_error.strip():
        raise ValueError("validation_error must not be empty")
    return (
        f"{original_prompt.rstrip()}\n\n"
        "RETRY_NOTICE: Your previous response violated pairwise protocol "
        "v3.0. Return exactly:\n"
        "Feedback: <concise evidence-based explanation>\n"
        "[RESULT] <1 or 2>\n"
        "Do not use Markdown or include text after the result. "
        f"Validation error: {validation_error}"
    )
